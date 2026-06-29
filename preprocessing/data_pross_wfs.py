import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt

def wavelet_filter_by_wavelength(signal: np.ndarray, dx: float, cutoff_wavelength: float = 50.0):
    """
    通过波长滤波，滤去大于cutoff_wavelength的长波信息。
    dx: 采样间隔（单位：米）
    cutoff_wavelength: 截止波长（单位：米），滤去大于该波长的成分
    """
    wavelet = 'db4'
    max_level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)
    coeffs = pywt.wavedec(signal, wavelet, level=max_level)
    # 计算每层对应的波长
    for i in range(len(coeffs)):
        wavelength = 2 ** i * dx
        if wavelength > cutoff_wavelength:
            coeffs[i] = np.zeros_like(coeffs[i])
    filtered_signal = pywt.waverec(coeffs, wavelet)
    filtered_signal = filtered_signal[:len(signal)]
    return filtered_signal


def data_processing_wfs(dynamic_file_path: str, static_file_path: str) -> pd.DataFrame:
    """
    处理WFS动检数据文件，提取里程，速度，七项不平顺数据，横向、垂向加速度,
    处理WFS静检数据文件，提取里程、左右高低、左右轨向。
    根据两个文件的最小里程范围对两个文件的数据进行裁剪，并对齐里程。
    参数:
        dynamic_file_path: WFS动检数据文件路径
        static_file_path: WFS静检数据文件路径
    返回:
        处理后的DataFrame
    """

    # 1) 读取动检数据
    dynamic_df = pd.read_csv(dynamic_file_path, encoding="GBK")
    dynamic_df.columns = [str(c).strip() for c in dynamic_df.columns]

    # 2) 读取静检数据
    static_df = pd.read_csv(static_file_path, encoding="UTF-8")
    static_df.columns = [str(c).strip() for c in static_df.columns]

    # 3) 提取动检数据里程、速度、七项不平顺数据、横向加速度、垂向加速度
    col_name_dynamic = ["里程", "左高低(mm)", "右高低(mm)", "左轨向(mm)", "右轨向(mm)", "三角坑(mm)", "轨距(mm)", "超高(mm)", "水平(mm)", "横向加速度(g)", "垂向加速度(g)"]
    dynamic_df = dynamic_df[col_name_dynamic]

    #）提取静检数据里程、左右高低、左右轨向
    col_name_static = ["里程", "左高低", "右高低", "左轨向", "右轨向", "超高差值", "轨距差值"]
    static_df = static_df[col_name_static]
    # 4) 按最小公共里程范围裁剪（默认即静检范围与动检交集）
    x0 = max(dynamic_df["里程"].min(), static_df["里程"].min())
    x1 = min(dynamic_df["里程"].max(), static_df["里程"].max())

    return dynamic_df[(dynamic_df["里程"] >= x0) & (dynamic_df["里程"] <= x1)], static_df[(static_df["里程"] >= x0) & (static_df["里程"] <= x1)]

def irre_plot(dynamic_df: pd.DataFrame, static_df: pd.DataFrame, x0: float, x1: float) -> None:
    """
    绘制动检和静检数据的里程-不平顺关系图，图片中文字均为英文，Times New Roman。
    参数:
        dynamic_df: 处理后的动检数据DataFrame
        static_df: 处理后的静检数据DataFrame
        x0: 对齐后的最小里程
        x1: 对齐后的最大里程
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dynamic_df["里程"], dynamic_df["左高低(mm)"], label="dynamic vertical irregularity (mm)", color="blue")
    plt.plot(static_df["里程"], static_df["左高低"], label="static vertical irregularity (mm)", color="orange")
    plt.xlim(x0, x1)
    plt.xlabel("mileage (km)", fontname="Times New Roman", fontsize=12)
    plt.ylabel("irregularity index", fontname="Times New Roman", fontsize=12)
    plt.title("WFS Dynamic and Static Irregularity Comparison", fontname="Times New Roman", fontsize=14)
    plt.legend()
    plt.grid()
    plt.savefig("wfs_irre_plot.png")





if __name__ == "__main__":
    dynamic_file = "preprocessing/动检数据/五峰山大桥梁端/20220604/原始文件/20220604.csv"
    static_file = "preprocessing/静检数据/五峰山大桥梁端/20220605/原始文件/20220605.csv"
    
    dynamic_aligned, static_aligned = data_processing_wfs(dynamic_file, static_file)
    static_aligned["左高低"] = wavelet_filter_by_wavelength(static_aligned["左高低"].values, dx=0.25, cutoff_wavelength=100.0)
    irre_plot(dynamic_aligned, static_aligned, x0=dynamic_aligned["里程"].min(), x1=dynamic_aligned["里程"].max())