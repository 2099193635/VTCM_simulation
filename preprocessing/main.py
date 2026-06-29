from __future__ import annotations

from pathlib import Path

from alignment_pre import (
	clip_by_mileage_range,
	flip_dynamic_alignment_irregularity,
	get_alignment_range,
	plot_alignment_before_after,
	save_aligned_results,
)
from dynamic_pre import plot_dynamic_irregularity_psd, process_dynamic_file
from ledger_pre import clip_ledger_by_mileage_range, process_ledger_file
from static_irre_pre import export_for_irregularity_external_import, process_static_files


def run_preprocessing_pipeline() -> dict:
	base = Path(r"preprocessing")

	# 绘图显示的实际里程范围（km，可调）；填 None 时显示完整/自动聚焦范围。
	plot_mileage_start = 273.8
	plot_mileage_end = 274.6

	# 1) 台账处理 -> 生成与 configs 同结构的曲线/坡度参数
	ledger_raw = base / "台账" / "京包客专.xls"
	ledger_out = base / "台账" / "处理后"
	curve_df, gradient_df = process_ledger_file(ledger_raw, ledger_out)

	# 2) 动检处理 -> 统一绝对里程
	dyn_raw = base / "动检数据" / "呼局" / "20210416" / "原始文件" / "动检上行20210416-238-363.txt"
	dyn_out = base / "动检数据" / "呼局" / "20210416" / "处理后" / "动检上行20210416-238-363.processed.csv"
	dynamic_df = process_dynamic_file(dyn_raw, dyn_out)

	# 3) 静检处理 -> 先拼合原始目录下多个 jdt，再绘图与导出（绝对里程）
	static_raw_dir = base / "静检数据" / "呼局" / "20210416" / "原始文件"
	static_raw_files = sorted(static_raw_dir.glob("*.jdt"))
	if not static_raw_files:
		raise FileNotFoundError(f"未在目录中找到静检 jdt 文件: {static_raw_dir}")

	# 依据文件名末段自动生成拼合前缀，例如 271-278
	ranges = []
	for p in static_raw_files:
		name = p.stem
		parts = name.split("-")
		if len(parts) >= 2:
			try:
				ranges.extend([int(parts[-2]), int(parts[-1])])
			except Exception:
				pass
	if ranges:
		static_prefix = f"静检上行20210416-{min(ranges)}-{max(ranges)}.merged"
	else:
		static_prefix = "静检上行20210416.merged"

	static_processed_dir = base / "静检数据" / "呼局" / "20210416" / "处理后"
	static_plot_dir = base / "静检数据" / "呼局" / "20210416" / "原始文件" / "绘图检测"
	static_result = process_static_files(
		raw_files=static_raw_files,
		processed_dir=static_processed_dir,
		plot_dir=static_plot_dir,
		file_prefix=static_prefix,
		save_tidy_csv=True,
		plot_pct_start=0.0,
		plot_pct_end=1.0,
		plot_mileage_start=plot_mileage_start,
		plot_mileage_end=plot_mileage_end,
	)
	static_df = static_result["tidy_df"]

	# 4) 暂不修正里程；仅将动检轨向不平顺反向，再与静检基准绘图
	dynamic_aligned, flipped_dynamic_channels = flip_dynamic_alignment_irregularity(dynamic_df)
	static_aligned = static_df.copy()
	correction_summary = {
		"alignment_basis": "static",
		"correction_mode": "dynamic_alignment_sign_flip_only",
		"dynamic_shift_km": 0.0,
		"flipped_dynamic_channels": flipped_dynamic_channels,
	}

	# 4.1) 按最小公共里程范围裁剪（默认即静检范围与动检交集）
	x0, x1 = get_alignment_range(dynamic_aligned, static_aligned, prefer_static=True)
	dynamic_aligned = clip_by_mileage_range(dynamic_aligned, x0, x1, mileage_col="里程")
	static_aligned = clip_by_mileage_range(static_aligned, x0, x1, mileage_col="里程")
	curve_df, gradient_df = clip_ledger_by_mileage_range(curve_df, gradient_df, x0, x1)

	# 保存台账裁剪后结果（同配置结构）
	curve_df.to_csv(ledger_out / "curve_parameters.csv", index=False, encoding="utf-8-sig")
	gradient_df.to_csv(ledger_out / "gradient_parameters.csv", index=False, encoding="utf-8-sig")

	# 4.2) 绘制 7 种动检不平顺空间 PSD
	dynamic_psd_dir = base / "动检数据" / "呼局" / "20210416" / "处理后" / "psd"
	dynamic_psd = plot_dynamic_irregularity_psd(
		dynamic_aligned,
		dynamic_psd_dir,
		file_prefix="动检上行20210416-238-363.aligned",
	)

	align_plot_dir = base / "静检数据" / "呼局" / "20210416" / "原始文件" / "绘图检测" / "alignment"
	align_plot_files = plot_alignment_before_after(
		dynamic_before=dynamic_df,
		dynamic_after=dynamic_aligned,
		static_df=static_aligned,
		out_dir=align_plot_dir,
		mileage_start=plot_mileage_start,
		mileage_end=plot_mileage_end,
	)

	# 5) 保存对齐后的动静数据（处理后目录）
	dyn_aligned_out = base / "动检数据" / "呼局" / "20210416" / "处理后" / "动检上行20210416-238-363.aligned.csv"
	sta_aligned_out = base / "静检数据" / "呼局" / "20210416" / "处理后" / f"{static_prefix}.aligned.csv"
	save_aligned_results(dynamic_aligned, static_aligned, dyn_aligned_out, sta_aligned_out)

	# 5.0) 保存动检数据的横向加速度和垂向加速度
	dyn_acceleration_out = base / "动检数据" / "呼局" / "20210416" / "处理后" / "动检上行20210416-238-363.acceleration.csv"
	acceleration_cols = ['里程', '横向加速度(g)', '垂向加速度(g)']
	if all(col in dynamic_aligned.columns for col in acceleration_cols):
		dynamic_aligned[acceleration_cols].to_csv(dyn_acceleration_out, index=False, encoding="utf-8-sig")
		print(f"加速度数据已保存至: {dyn_acceleration_out}")
	else:
		print(f"警告: 动检数据缺少加速度列。可用列: {dynamic_aligned.columns.tolist()}")
	
	# 5.1) 静检外部导入文件（绝对里程，且已按最小范围裁剪）
	static_external_aligned = export_for_irregularity_external_import(
		static_aligned,
		base / "静检数据" / "呼局" / "20210416" / "处理后" / f"{static_prefix}.aligned.external",
		file_prefix=f"{static_prefix}.aligned",
		use_relative_mileage=False,
	)

	return {
		"ledger_curve_rows": len(curve_df),
		"ledger_gradient_rows": len(gradient_df),
		"dynamic_rows": len(dynamic_aligned),
		"static_rows": len(static_aligned),
		"align_range_km": [x0, x1],
		"dynamic_shift_km": correction_summary["dynamic_shift_km"],
		"correction_summary": correction_summary,
		"align_plots": [str(p) for p in align_plot_files],
		"dynamic_psd": dynamic_psd,
		"static_external": static_external_aligned,
	}


if __name__ == "__main__":
	summary = run_preprocessing_pipeline()
	print("Preprocessing completed.")
	print(summary)

