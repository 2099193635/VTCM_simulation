import numpy as np
from scipy.interpolate import CubicSpline
from dataclasses import dataclass

@dataclass
class ContactGeometryInfo:
    """轮轨接触几何资产包：装载所有预处理后的型面、曲率与接触角数据"""
    RrL_ql: np.ndarray; RrR_ql: np.ndarray
    gauge: float; wdis: float; Rwo: float; gdis: float; zplot: float; hrr: float
    # 这里为了简便，省略了所有字段的类型提示（实际使用中，可以直接作为一个普通类或字典返回）
    # 在下面的函数中，我们将把所有需要返回的属性动态绑定到这个对象上
    
class WheelRailContactProcessor:
    """轮轨接触几何前处理引擎"""
    
    @staticmethod
    def process_pre_information(rail_data: np.ndarray, wheel_data: np.ndarray):
        """
        处理轮轨原始廓形数据，生成接触力学计算所需的全套几何参量
        输入:
            rail_data: 钢轨原始坐标点 (2 x N 数组，单位 mm)
            wheel_data: 车轮原始坐标点 (2 x M 数组，单位 mm)
        返回:
            info: 包含所有几何信息的综合数据对象
        """
        if rail_data.shape[0] != 2:
            rail_data = rail_data.T
        if wheel_data.shape[0] != 2:
            wheel_data = wheel_data.T
        # 1. 数据单位转换 (mm -> m)
        rail = rail_data.copy() / 1000.0
        wheel = wheel_data.copy() / 1000.0
        
        info = ContactGeometryInfo(None, None, 0, 0, 0, 0, 0, 0)
        
        # ==========================================
        # PART 1: 计算钢轨与车轮的表面曲率半径 (基于三次样条的精确二阶导数)
        # ==========================================
        # 钢轨轨面曲率半径计算
        # 使用自然边界条件的三次样条插值
        rail_spline = CubicSpline(rail[0, :], rail[1, :], bc_type='natural')
        drL = rail_spline.derivative(1)(rail[0, :])     # 一阶导数
        ddrL = rail_spline.derivative(2)(rail[0, :])    # 二阶导数
        # 曲率半径公式: R = (1 + y'^2)^1.5 / |y''|
        # 加上 1e-12 防止直线段二阶导数为 0 导致除零溢出
        info.RrL_ql = ((1 + drL**2)**1.5) / (np.abs(ddrL) + 1e-12)
        info.RrR_ql = np.flip(info.RrL_ql)
        
        # 车轮踏面曲率半径计算
        wheel_spline = CubicSpline(wheel[0, :], wheel[1, :], bc_type='natural')
        dwR = wheel_spline.derivative(1)(wheel[0, :])
        ddwR = wheel_spline.derivative(2)(wheel[0, :])
        RwR_ql = ((1 + dwR**2)**1.5) / (np.abs(ddwR) + 1e-12)
        RwL_ql = np.flip(RwR_ql)

        # ==========================================
        # PART 2: 定义轮对及钢轨结构基础参量
        # ==========================================
        info.gauge = 1435.0 / 1000.0    # 标准轨距
        info.wdis = 1493.0 / 1000.0     # 接触点距离（对应轮背距1353）
        info.Rwo = 0.43                 # 车轮名义滚动圆半径 (m)
        info.gdis = 16.0 / 1000.0       # 轨距测量点向下偏移量
        info.zplot = info.Rwo + 0.1
        info.hrr = 94.53 / 1000.0       # 轨顶至扭转中心距离
        
        # ==========================================
        # PART 3: 车轮廓形处理 (自定坐标系转换与切角计算)
        # ==========================================
        # 右车轮廓形平移 (横向加上半轮距，垂向减去名义半径)
        WprofR_y = wheel[0, :] + info.wdis / 2.0
        WprofR_z = wheel[1, :] - info.Rwo
        # 左车轮为右车轮的镜像翻转
        WprofL_y = -np.flip(WprofR_y)
        WprofL_z = np.flip(WprofR_z)
        
        # 存储廓形对应的滚动圆半径 (深度取反即为实际半径)
        RwL = -WprofL_z
        RwR = -WprofR_z
        
        # 计算切线斜率 (注意 Python 插值要求 X 必须单调递增)
        # 确保 WprofL_y 是严格递增的
        spline_L = CubicSpline(WprofL_y, WprofL_z, bc_type='natural')
        dL = spline_L.derivative(1)(WprofL_y)
        
        spline_R = CubicSpline(WprofR_y, WprofR_z, bc_type='natural')
        dR = -np.flip(dL)  # 物理对称性：右侧斜率等于左侧翻转后取反
        
        # 计算接触角 (切线斜率的反正切的绝对值)
        detR = np.abs(np.arctan(spline_R.derivative(1)(WprofR_y)))
        detL = np.flip(detR)
        
        # ==========================================
        # PART 4: 廓形分割 (应对两点接触：踏面区 vs 轮缘区)
        # ==========================================
        Split_Number = 2344
        # Python 切片与 MATLAB 的 1-based 索引转换
        # 轮缘区 (Number2)
        info.WprofR2_y = WprofR_y[:Split_Number]
        info.WprofR2_z = WprofR_z[:Split_Number]
        info.WprofL2_y = WprofL_y[-Split_Number:]
        info.WprofL2_z = WprofL_z[-Split_Number:]
        
        info.RwR2 = RwR[:Split_Number]
        info.RwL2 = RwL[-Split_Number:]
        info.dR2 = dR[:Split_Number]
        info.dL2 = dL[-Split_Number:]
        info.detR2 = detR[:Split_Number]
        info.detL2 = detL[-Split_Number:]
        info.RwR_ql2 = RwR_ql[:Split_Number]
        info.RwL_ql2 = RwL_ql[-Split_Number:]
        
        # 踏面区 (Number1)
        info.WprofR_y = WprofR_y[Split_Number:]
        info.WprofR_z = WprofR_z[Split_Number:]
        info.WprofL_y = WprofL_y[:-Split_Number]
        info.WprofL_z = WprofL_z[:-Split_Number]
        
        info.RwR = RwR[Split_Number:]
        info.RwL = RwL[:-Split_Number]
        info.dR = dR[Split_Number:]
        info.dL = dL[:-Split_Number]
        info.detR = detR[Split_Number:]
        info.detL = detL[:-Split_Number]
        info.RwR_ql = RwR_ql[Split_Number:]
        info.RwL_ql = RwL_ql[:-Split_Number]

        # ==========================================
        # PART 5: 钢轨廓形处理与坐标系旋转 (轨底坡补偿)
        # ==========================================
        cant = np.arctan(1.0 / 40.0)  # 轨底坡 1:40
        # 构建旋转矩阵
        RotateL = np.array([
            [np.cos(cant),  np.sin(cant)],
            [-np.sin(cant), np.cos(cant)]
        ])
        
        # 旋转钢轨坐标系
        rprofL = RotateL @ rail
        rprofL_y = rprofL[0, :]
        rprofL_z = rprofL[1, :] - np.max(rprofL[1, :]) + info.gdis  # 对齐轨面最高点
        
        # 确定钢轨顶面中心及轨距测量点
        mid_idx = len(rprofL_z) // 2
        rprofLs_y = rprofL_y[mid_idx:]
        rprofLs_z = rprofL_z[mid_idx:]
        
        # 寻找轨距计算基准点 (Z 绝对值最小处)
        locg = np.argmin(np.abs(rprofLs_z))
        info.locg = locg
        info.rprofLs_y = rprofLs_y
        
        # 归一化：横坐标减去基准点并扣除半轨距，纵坐标扣除参考高度
        rprofL_y = rprofL_y - rprofLs_y[locg] - info.gauge / 2.0
        rprofL_z = rprofL_z - info.zplot
        
        rprofR_y = -np.flip(rprofL_y)
        rprofR_z = np.flip(rprofL_z)
        
        # 挂载处理完毕的钢轨信息
        info.cant = cant
        info.rprofL_y = rprofL_y; info.rprofL_z = rprofL_z
        info.rprofR_y = rprofR_y; info.rprofR_z = rprofR_z
        info.rail = rail  # 保存原始归一化轨面供后续插值查找使用
        
        # (可选预留变量：ppdL, ppdR，原版用于直接求函数值，这里我们已离散化)
        info.ppdL = spline_L
        info.ppdR = spline_R

        return info