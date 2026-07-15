from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import stft, welch


ROOT = Path("results/stiffness_four_mechanism_comparison")
OUT_DIR = ROOT / "_comparison"
OUT_BASE = OUT_DIR / "stiffness_response_frequency_timefrequency"
G = 9.81

PSD_WINDOW_M = (180.0, 330.0)
STFT_WINDOW_M = (140.0, 380.0)
BANDS_HZ = (
    ("0.5-5", 0.5, 5.0),
    ("5-15", 5.0, 15.0),
    ("15-40", 15.0, 40.0),
    ("40-100", 40.0, 100.0),
)


@dataclass(frozen=True)
class CaseStyle:
    key: str
    folder_token: str
    label: str
    color: str


CASES = (
    CaseStyle("baseline", "case_00_baseline", "baseline", "#4D4D4D"),
    CaseStyle("fastener_failure_local", "case_01_fastener_failure_local", "fastener loss", "#0072B2"),
    CaseStyle("sleeper_void_local", "case_02_sleeper_void_local", "sleeper void", "#009E73"),
    CaseStyle("ballast_softening_100m", "case_03_ballast_softening_100m", "ballast softening", "#D55E00"),
    CaseStyle("subgrade_weakening_100m", "case_04_subgrade_weakening_100m", "subgrade weakening", "#CC79A7"),
)


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7.0,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    }
)


def find_result_file(token: str) -> Path:
    matches = sorted(ROOT.glob(f"*{token}*/files/simulation_result.npz"))
    if not matches:
        raise FileNotFoundError(f"No simulation_result.npz found for {token}")
    return matches[-1]


def load_case(style: CaseStyle) -> dict[str, np.ndarray | float | str]:
    path = find_result_file(style.folder_token)
    data = np.load(path, allow_pickle=True)
    accel = np.asarray(data["A"], dtype=float)
    distance = np.asarray(data["Track_rel_mileage_m"], dtype=float)
    dt = float(data["dt"])

    def col(index: int) -> np.ndarray:
        return accel[:, index] if accel.shape[1] > index else np.zeros(accel.shape[0])

    return {
        "key": style.key,
        "label": style.label,
        "color": style.color,
        "path": str(path),
        "dt": dt,
        "distance_m": distance,
        "carbody_az_g": col(1) / G,
        "carbody_ay_g": col(0) / G,
        "bogie_az_g": 0.5 * (col(6) + col(11)) / G,
        "bogie_ay_g": 0.5 * (col(5) + col(10)) / G,
    }


def local_mask(distance: np.ndarray, window_m: tuple[float, float]) -> np.ndarray:
    return (distance >= window_m[0]) & (distance <= window_m[1])


def welch_psd(signal: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    nperseg = min(1024, max(64, signal.size // 2))
    return welch(
        signal,
        fs=1.0 / dt,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
    )


def integrate_band(freq: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    mask = (freq >= lo) & (freq < hi)
    if np.count_nonzero(mask) < 2:
        return 0.0
    return float(np.trapz(psd[mask], freq[mask]))


def spectral_centroid(freq: np.ndarray, psd: np.ndarray, lo: float = 0.5, hi: float = 100.0) -> float:
    mask = (freq >= lo) & (freq <= hi)
    denom = float(np.trapz(psd[mask], freq[mask]))
    if denom <= 0.0:
        return float("nan")
    return float(np.trapz(freq[mask] * psd[mask], freq[mask]) / denom)


def compute_psd_features(cases: list[dict[str, np.ndarray | float | str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    summary = []
    baseline_power: dict[tuple[str, str], float] = {}

    for case in cases:
        distance = np.asarray(case["distance_m"], dtype=float)
        dt = float(case["dt"])
        mask = local_mask(distance, PSD_WINDOW_M)
        for channel in ("carbody_az_g", "carbody_ay_g", "bogie_az_g", "bogie_ay_g"):
            freq, psd = welch_psd(np.asarray(case[channel], dtype=float)[mask], dt)
            centroid = spectral_centroid(freq, psd)
            for band, lo, hi in BANDS_HZ:
                power = integrate_band(freq, psd, lo, hi)
                if case["key"] == "baseline":
                    baseline_power[(channel, band)] = power
                base = baseline_power.get((channel, band), np.nan)
                delta_pct = np.nan if base == 0 or not np.isfinite(base) else (power / base - 1.0) * 100.0
                rows.append(
                    {
                        "case": case["key"],
                        "channel": channel,
                        "band_hz": band,
                        "power_g2": power,
                        "delta_vs_baseline_pct": delta_pct,
                        "centroid_hz": centroid,
                    }
                )

            if channel == "bogie_az_g" and case["key"] != "baseline":
                case_summary = {"case": case["key"], "bogie_az_centroid_hz": centroid}
                for band, lo, hi in BANDS_HZ:
                    power = integrate_band(freq, psd, lo, hi)
                    base = baseline_power.get((channel, band), np.nan)
                    label = band.replace(".", "p").replace("-", "_")
                    case_summary[f"bogie_az_{label}_power_delta_pct"] = (
                        np.nan if base == 0 or not np.isfinite(base) else (power / base - 1.0) * 100.0
                    )
                summary.append(case_summary)

    return pd.DataFrame(rows), pd.DataFrame(summary)


def differential_stft(
    case: dict[str, np.ndarray | float | str],
    baseline: dict[str, np.ndarray | float | str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distance = np.asarray(case["distance_m"], dtype=float)
    dt = float(case["dt"])
    mask = local_mask(distance, STFT_WINDOW_M)
    case_signal = np.asarray(case["bogie_az_g"], dtype=float)[mask]
    base_signal = np.asarray(baseline["bogie_az_g"], dtype=float)[mask]
    n = min(case_signal.size, base_signal.size)
    diff = case_signal[:n] - base_signal[:n]
    local_distance = distance[mask][:n]
    local_time = np.arange(n) * dt

    nperseg = min(256, max(64, n // 4))
    freq, t_seg, zxx = stft(
        diff,
        fs=1.0 / dt,
        window="hann",
        nperseg=nperseg,
        noverlap=int(nperseg * 0.875),
        detrend=False,
        boundary=None,
        padded=False,
    )
    mileage = np.interp(t_seg, local_time, local_distance)
    magnitude = np.abs(zxx)
    return mileage, freq, magnitude


def panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )


def draw_figure(cases: list[dict[str, np.ndarray | float | str]], features: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = next(c for c in cases if c["key"] == "baseline")
    nonbaseline = [c for c in cases if c["key"] != "baseline"]

    fig = plt.figure(figsize=(7.35, 5.65), constrained_layout=False)
    gs = fig.add_gridspec(
        3,
        4,
        left=0.075,
        right=0.982,
        bottom=0.075,
        top=0.835,
        wspace=0.42,
        hspace=0.62,
    )
    ax_a = fig.add_subplot(gs[0, :2])
    ax_b = fig.add_subplot(gs[0, 2:])
    stft_axes = [fig.add_subplot(gs[1 + i // 2, (i % 2) * 2 : (i % 2) * 2 + 2]) for i in range(4)]

    # Panel a: local bogie vertical acceleration PSD. The legend is outside the
    # data rectangle to avoid colliding with the panel title and annotations.
    psd_records = []
    legend_handles = []
    legend_labels = []
    for case in cases:
        mask = local_mask(np.asarray(case["distance_m"], dtype=float), PSD_WINDOW_M)
        freq, psd = welch_psd(np.asarray(case["bogie_az_g"], dtype=float)[mask], float(case["dt"]))
        keep = (freq >= 0.5) & (freq <= 100.0)
        line = ax_a.plot(freq[keep], psd[keep], lw=1.05, color=str(case["color"]), label=str(case["label"]))[0]
        legend_handles.append(line)
        legend_labels.append(str(case["label"]))
        psd_records.extend(
            {
                "case": case["key"],
                "frequency_hz": float(f),
                "bogie_az_psd_g2_per_hz": float(p),
            }
            for f, p in zip(freq[keep], psd[keep])
        )
    ax_a.set_yscale("log")
    ax_a.set_xlim(0, 100)
    ax_a.set_xlabel("Frequency (Hz)")
    ax_a.set_ylabel(r"PSD of bogie $a_z$ (g$^2$ Hz$^{-1}$)")
    ax_a.set_title("Local frequency response", loc="left", pad=6, fontsize=8.2)
    ax_a.text(
        0.98,
        0.97,
        f"Welch PSD, {PSD_WINDOW_M[0]:.0f}-{PSD_WINDOW_M[1]:.0f} m",
        transform=ax_a.transAxes,
        ha="right",
        va="top",
        fontsize=6.4,
        color="#555555",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.5},
    )
    panel_label(ax_a, "a", y=1.07)

    # Panel b: band-wise changes relative to baseline.
    band_order = [b[0] for b in BANDS_HZ]
    x = np.arange(len(band_order))
    width = 0.18
    for i, case in enumerate(nonbaseline):
        vals = []
        for band in band_order:
            part = features[
                (features["case"] == case["key"])
                & (features["channel"] == "bogie_az_g")
                & (features["band_hz"] == band)
            ]
            vals.append(float(part["delta_vs_baseline_pct"].iloc[0]))
        ax_b.bar(x + (i - 1.5) * width, vals, width=width, color=str(case["color"]), label=str(case["label"]))
    ax_b.axhline(0.0, color="#555555", lw=0.7)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([f"{b}\nHz" for b in band_order])
    ax_b.set_ylabel(r"$\Delta$ band power vs baseline (%)")
    ax_b.set_title("Frequency-band separation", loc="left", pad=7, fontsize=8.2)
    ax_b.legend(ncol=2, loc="upper left", fontsize=6.2, columnspacing=0.8, handlelength=1.2)
    panel_label(ax_b, "b", y=1.08)

    vmax = 0.0
    stft_payload = []
    for case in nonbaseline:
        mileage, freq, mag = differential_stft(case, baseline)
        keep = freq <= 100.0
        mag_keep = mag[keep, :]
        vmax = max(vmax, float(np.nanpercentile(mag_keep, 99.0)))
        stft_payload.append((case, mileage, freq[keep], mag_keep))
    vmax = max(vmax, 1e-5)

    for ax, (case, mileage, freq, mag) in zip(stft_axes, stft_payload):
        mesh = ax.pcolormesh(
            mileage,
            freq,
            mag,
            shading="auto",
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
        )
        ax.axvspan(200.0, 300.0, color="white", alpha=0.11, lw=0)
        ax.set_ylim(0, 100)
        ax.set_xlim(STFT_WINDOW_M)
        ax.set_title(str(case["label"]), loc="left", fontsize=7.3, pad=4)
        ax.set_xlabel("Mileage (m)")
        ax.set_ylabel("Frequency (Hz)")
    panel_label(stft_axes[0], "c", y=1.06)
    cbar = fig.colorbar(mesh, ax=stft_axes, orientation="vertical", fraction=0.025, pad=0.012)
    cbar.set_label(r"|STFT $\Delta$bogie $a_z$| (g)", fontsize=7)
    cbar.ax.tick_params(labelsize=6.2)

    fig.suptitle(
        "Stiffness defects show distinct spectral and time-frequency fingerprints",
        x=0.075,
        y=0.987,
        ha="left",
        fontsize=9.2,
        fontweight="bold",
    )
    fig.text(
        0.075,
        0.955,
        "PSD uses the local response window; time-frequency maps show differential bogie vertical acceleration relative to baseline.",
        ha="left",
        va="top",
        fontsize=6.6,
        color="#555555",
    )
    fig.legend(
        legend_handles,
        legend_labels,
        ncol=5,
        loc="upper left",
        bbox_to_anchor=(0.075, 0.925),
        bbox_transform=fig.transFigure,
        columnspacing=0.95,
        handlelength=1.55,
        handletextpad=0.38,
        borderaxespad=0.0,
        fontsize=6.3,
        frameon=False,
    )

    fig.savefig(OUT_BASE.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(OUT_BASE.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT_BASE.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(OUT_BASE.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(psd_records).to_csv(OUT_DIR / "stiffness_response_psd_source_data.csv", index=False)


def main() -> None:
    cases = [load_case(style) for style in CASES]
    features, summary = compute_psd_features(cases)
    features.to_csv(OUT_DIR / "stiffness_response_frequency_features.csv", index=False)
    features.to_json(OUT_DIR / "stiffness_response_frequency_features.json", orient="records", indent=2)
    summary.to_csv(OUT_DIR / "stiffness_response_feature_summary.csv", index=False)
    summary.to_json(OUT_DIR / "stiffness_response_feature_summary.json", orient="records", indent=2)
    draw_figure(cases, features)
    print(f"Wrote {OUT_BASE.with_suffix('.png')}")


if __name__ == "__main__":
    main()
