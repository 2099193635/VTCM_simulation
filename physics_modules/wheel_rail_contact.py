import numpy as np



class WheelRailInteraction:
    """轮轨两点接触空间几何与接触力学联合求解器 (迹线法)"""
    
    def __init__(self, geom_info, veh_params):
        """
        传入我们在预处理阶段生成的 ContactGeometryInfo 对象
        """
        self.P0 = veh_params.P0
        self.info = geom_info
        # self.detZwj0 = 84.2625739116535 / 1000.0  # 初始状态轮轨距离 (m)
        self.detZwj0 = self._calculate_static_equilibrium()
        print(f" -> [接触力学初始化] 自动寻优完成! 轮轨静态参考距离 detZwj0 = {self.detZwj0 * 1000:.6f} mm")
        
    def _calculate_static_equilibrium(self):
        """
        [自适应落车寻优]
        在静态理想位置 (Yw=0, Zw=0, Yaw=0) 下，寻找轮轨几何最小距离，
        并结合静轮重 P0 与 Hertz 理论，反推精确的静态初始接触距离常数。
        """
        info = self.info
        
        # 1. 提取钢轨静态位置
        mid_L = len(info.rprofL_y) // 2
        zbo_L = np.array([info.rprofL_y[mid_L], info.rprofL_z[mid_L]])
        To_L = zbo_L - np.array([info.hrr * np.sin(info.cant), info.hrr * np.cos(info.cant)])
        
        d_rprofL_y = (info.rprofL_y - To_L[0]) * 1.0 + To_L[0]
        d_rprofL_z = (info.rprofL_z - To_L[1]) * 1.0 + To_L[1]

        # ========================================================
        # 🌟 核心修复：静态空间迹线的完美数学退化
        # 当 phiw=0, psiw=0, Yw=0 时，lx=0, ly=1, lz=0。
        # 动态迹线公式完美退化为纯二维廓形：
        # ========================================================
        yjxL = info.WprofL_y
        zjxL = -info.RwL

        # 3. 寻找纯几何的最小距离 (不考虑变形)
        rail_z_interp = np.interp(yjxL, d_rprofL_y, d_rprofL_z)
        distance = zjxL - rail_z_interp
        min_idx = np.argmin(distance)
        
        # 此时算出来的 minL_nominal 将绝对等于您观察到的 0.084... 米！
        minL_nominal = distance[min_idx]  

        # 4. 提取该接触点的曲率参数，计算静态 Hertz 刚度 Gwr
        det_d = info.detL[min_idx]
        Rw_c = info.RwL[min_idx]
        Ry1 = info.RwL_ql[min_idx]
        
        locr = np.argmin(np.abs(d_rprofL_y - yjxL[min_idx]))
        Ry2 = info.RrL_ql[locr]

        # 转换为一维数组以适应 ContactMechanics
        Gwr = ContactMechanics.calc_Gwr(np.array([Ry1]), np.array([Ry2]), np.array([det_d]), Rw_c)[0]

        # 5. 反推赫兹弹性静压缩量 delta_0
        PW_static = self.P0 / np.cos(det_d)
        detZc_static = Gwr * (PW_static ** (2/3))
        delta_0 = detZc_static * np.cos(det_d)
        
        return minL_nominal + delta_0
    
    def calculate_two_point_contact(self, nw, vc, omg,
                                    Zw, Yw, phiw, psiw,
                                    LRKX_Y, LRKX_Z, thetaL,
                                    RRKX_Y, RRKX_Z, thetaR,
                                    VXwo, VYwo, VZwo, Vwphi, Vwbeta, Vwpsi,
                                    VrkxY_L, VrkxZ_L, VrkxO_L,
                                    VrkxY_R, VrkxZ_R, VrkxO_R,
                                    Irrez_L, Irrey_L, VIrrez_L, VIrrey_L,
                                    Irrez_R, Irrey_R, VIrrez_R, VIrrey_R):
        """
        针对单个轮对 (nw) 的空间两点接触寻优与接触力计算。
        为了极致性能，抛弃了极其耗时的 spline 动态插值，改用高密度网格的极速求小值。
        """
        # --- 0. 坐标系对齐与符号转换 (与 MATLAB 保持绝对一致) ---
        Zw = -Zw
        RRKX_Z = -RRKX_Z; LRKX_Z = -LRKX_Z
        Irrez_L = -Irrez_L; Irrez_R = -Irrez_R
        
        info = self.info
        
        # --- 1. 钢轨动态位移与扭转 (Dynamic Rail Transformation) ---
        mid_L, mid_R = len(info.rprofL_y) // 2, len(info.rprofR_y) // 2
        zbo_L = np.array([info.rprofL_y[mid_L], info.rprofL_z[mid_L]])
        zbo_R = np.array([info.rprofR_y[mid_R], info.rprofR_z[mid_R]])
        
        To_L = zbo_L - np.array([info.hrr * np.sin(info.cant), info.hrr * np.cos(info.cant)])
        To_R = zbo_R - np.array([-info.hrr * np.sin(info.cant), info.hrr * np.cos(info.cant)])
        
        cos_tL, sin_tL = np.cos(thetaL), np.sin(thetaL)
        d_rprofL_y = (info.rprofL_y - To_L[0]) * cos_tL + (info.rprofL_z - To_L[1]) * sin_tL + To_L[0] + LRKX_Y + Irrey_L
        d_rprofL_z = -(info.rprofL_y - To_L[0]) * sin_tL + (info.rprofL_z - To_L[1]) * cos_tL + To_L[1] + LRKX_Z + Irrez_L

        cos_tR, sin_tR = np.cos(thetaR), np.sin(thetaR)
        d_rprofR_y = (info.rprofR_y - To_R[0]) * cos_tR + (info.rprofR_z - To_R[1]) * sin_tR + To_R[0] + RRKX_Y + Irrey_R
        d_rprofR_z = -(info.rprofR_y - To_R[0]) * sin_tR + (info.rprofR_z - To_R[1]) * cos_tR + To_R[1] + RRKX_Z + Irrez_R

        # --- 2. 构造车轮空间迹线 (Space Trace Generation) ---
        lx = -np.cos(phiw) * np.sin(psiw)
        ly = np.cos(phiw) * np.cos(psiw)
        lz = np.sin(phiw)
        
        def build_trace(Wprof_y, Rw, dW):
            """内部闭包：快速生成踏面/轮缘的迹线"""
            xB, yB, zB = lx * Wprof_y, ly * Wprof_y + Yw, lz * Wprof_y
            m_val = np.sqrt(np.maximum(1 - lx**2 * (1 + dW**2), 1e-12))
            denom = 1 - lx**2 + 1e-12
            xjx = xB - lx * Rw * dW
            yjx = yB + Rw / denom * (lx**2 * ly * dW - lz * m_val)
            zjx = zB + Rw / denom * (lx**2 * lz * dW + ly * m_val)
            return xjx, yjx, -zjx  # zjx 翻转回自定坐标系

        xjxL, yjxL, zjxL = build_trace(info.WprofL_y, info.RwL, info.dL)
        xjxR, yjxR, zjxR = build_trace(info.WprofR_y, info.RwR, info.dR)
        xjxL2, yjxL2, zjxL2 = build_trace(info.WprofL2_y, info.RwL2, info.dL2)
        xjxR2, yjxR2, zjxR2 = build_trace(info.WprofR2_y, info.RwR2, info.dR2)

        # --- 3. 极速接触点寻优 ---
        def find_contact_point(yjx, zjx, rprof_y, rprof_z):
            if rprof_y[-1] < rprof_y[0]:
                r_y, r_z = rprof_y[::-1], rprof_z[::-1]
            else:
                r_y, r_z = rprof_y, rprof_z
            valid_mask = (yjx >= r_y[0]) & (yjx <= r_y[-1])
            if not np.any(valid_mask):
                return 0, 0, False
            rail_z_interp = np.interp(yjx[valid_mask], r_y, r_z)
            distance = zjx[valid_mask] - rail_z_interp
            min_idx_valid = np.argmin(distance)
            return distance[min_idx_valid], np.where(valid_mask)[0][min_idx_valid], True

        minL, locwL, valid_L = find_contact_point(yjxL, zjxL, d_rprofL_y, d_rprofL_z)
        minR, locwR, valid_R = find_contact_point(yjxR, zjxR, d_rprofR_y, d_rprofR_z)
        minL2, locwL2, valid_L2 = find_contact_point(yjxL2, zjxL2, d_rprofL_y, d_rprofL_z)
        minR2, locwR2, valid_R2 = find_contact_point(yjxR2, zjxR2, d_rprofR_y, d_rprofR_z)

        # --- 4. 核心力学解算器 (高内聚闭包) ---
        # 共用角速度
        Wwx = Vwphi * np.cos(psiw) - (-omg + Vwbeta) * np.cos(phiw) * np.sin(psiw)
        Wwy = Vwphi * np.sin(psiw) + (-omg + Vwbeta) * np.cos(phiw) * np.cos(psiw)
        Wwz = (-omg + Vwbeta) * np.sin(phiw) + Vwpsi

        def evaluate_patch(valid, min_dist, locw, yjx, xjx, zjx, d_rprof_y,
                           Rw_arr, det_arr, Rw_ql_arr, Rr_ql_arr, Wprof_y, is_left):
            """计算单个接触斑的广义力与几何参量"""
            if not valid:
                return (0.,)*10 + (1, 0., 0., 0., 0., 0.) # 返回空数据
                
            det_d, Rw_c, Ry1 = det_arr[locw], Rw_arr[locw], Rw_ql_arr[locw]
            locr = np.argmin(np.abs(d_rprof_y - yjx[locw]))
            Ry2 = Rr_ql_arr[locr]

            detZ = (-Zw - (min_dist - self.detZwj0))
            if detZ <= 0:
                return (0.,)*10 + (1, 0., 0., 0., 0., 0.)

            sign_L = 1 if is_left else -1
            detZc = detZ / np.cos(det_d + sign_L * phiw)

            # 调用 Hertz 接触模块
            Gwr = ContactMechanics.calc_Gwr(Ry1, Ry2, det_d, Rw_c)
            PW = -(detZc / Gwr)**1.5

            # 坐标转换矩阵 (接触斑坐标系 -> 绝对坐标系)
            sin_psi, cos_psi = np.sin(psiw), np.cos(psiw)
            ang = phiw + sign_L * det_d
            sin_ang, cos_ang = np.sin(ang), np.cos(ang)
            
            Te = np.array([
                [cos_psi, sin_psi, 0],
                [-cos_ang * sin_psi, cos_ang * cos_psi, sign_L * sin_ang],
                [sin_ang * sin_psi, -sin_ang * cos_psi, cos_ang]
            ])
            zh = Te.T

            # 法向力分量
            NX, NY, NZ = zh[0, 2] * PW, zh[1, 2] * PW, -zh[2, 2] * PW

            # 相对速度提取 (利用迹线坐标)
            Rwx, Rwy, Rwz = xjx[locw], yjx[locw] - Yw, zjx[locw]  
            
            VrkxY = VrkxY_L if is_left else VrkxY_R
            VrkxZ = VrkxZ_L if is_left else VrkxZ_R
            VIrrey = VIrrey_L if is_left else VIrrey_R
            VIrrez = VIrrez_L if is_left else VIrrez_R
            VrkxO = VrkxO_L if is_left else VrkxO_R

            detVx = VXwo + (Wwy * Rwz - Wwz * Rwy)
            detVy = VYwo + (Wwz * Rwx - Wwx * Rwz) - (VrkxY + VIrrey)
            detVz = VZwo + (Wwx * Rwy - Wwy * Rwx) - (VrkxZ + VIrrez)

            detVx_b, detVy_b, detVz_b = Te @ np.array([detVx, detVy, detVz])

            # 蠕滑率计算
            Rreal = 0.4298267
            V_ref = 0.5 * (vc + vc * Rw_c * cos_psi / Rreal)
            ksix, ksiy = detVx_b / V_ref, detVy_b / V_ref

            detW3 = (Wwx - VrkxO) * Te[2, 0] + Wwy * Te[2, 1] + Wwz * Te[2, 2]
            ksisp = detW3 / V_ref

            # 调用蠕滑力学模块
            FX, FY, MZ = ContactMechanics.calc_creep_forces(ksix, ksiy, ksisp, Rw_c, Ry1, Ry2, abs(PW))

            # 切向力转换回全局坐标
            FL_jd = zh @ np.array([FX, FY, 0])
            ML_jd = zh @ np.array([0, 0, MZ])

            # 导出几何量
            hr = info.hrr + info.rail[1, locr]
            e = -info.rail[0, locr]
            
            wdis_half = info.wdis / 2
            wpoint = Wprof_y[locw] + (wdis_half if is_left else -wdis_half)
            rpoint = info.rprofL_y[locr] + info.rprofLs_y[info.locg] + info.gauge/2 if is_left \
                     else info.rprofR_y[locr] - info.rprofLs_y[info.locg] - info.gauge/2

            return (FL_jd[0], FL_jd[1], FL_jd[2], NX, NY, NZ, 
                    ML_jd[1], ML_jd[2], PW, FY, 
                    2, hr, e, Rw_c, wpoint, rpoint)

        # ---------------- 5. 组装四点力学向量 ----------------
        res_L1 = evaluate_patch(valid_L, minL, locwL, yjxL, xjxL, zjxL, d_rprofL_y, info.RwL, info.detL, info.RwL_ql, info.RrL_ql, info.WprofL_y, True)
        res_R1 = evaluate_patch(valid_R, minR, locwR, yjxR, xjxR, zjxR, d_rprofR_y, info.RwR, info.detR, info.RwR_ql, info.RrR_ql, info.WprofR_y, False)
        
        res_L2 = evaluate_patch(valid_L2, minL2, locwL2, yjxL2, xjxL2, zjxL2, d_rprofL_y, info.RwL2, info.detL2, info.RwL_ql2, info.RrL_ql, info.WprofL2_y, True)
        res_R2 = evaluate_patch(valid_R2, minR2, locwR2, yjxR2, xjxR2, zjxR2, d_rprofR_y, info.RwR2, info.detR2, info.RwR_ql2, info.RrR_ql, info.WprofR2_y, False)

        # 接触力叠加 (法向 + 蠕滑)
        FNx_L, FNy_L, FNz_L = res_L1[0]+res_L1[3], res_L1[1]+res_L1[4], res_L1[2]+res_L1[5]
        FNx_R, FNy_R, FNz_R = res_R1[0]+res_R1[3], res_R1[1]+res_R1[4], res_R1[2]+res_R1[5]
        FNx_L2, FNy_L2, FNz_L2 = res_L2[0]+res_L2[3], res_L2[1]+res_L2[4], res_L2[2]+res_L2[5]
        FNx_R2, FNy_R2, FNz_R2 = res_R2[0]+res_R2[3], res_R2[1]+res_R2[4], res_R2[2]+res_R2[5]

        # 计算触点力臂 a0
        a0 = abs(info.WprofR_y[locwR] - info.WprofL_y[locwL]) / 2 if valid_L and valid_R else 0
        a02 = abs(info.WprofR2_y[locwR2] - info.WprofL2_y[locwL2]) / 2 if valid_L2 and valid_R2 else 0

        # 返回全部计算结果字典，完美映射原版 MATLAB 变量
        return {
            'FNx_L': FNx_L, 'FNy_L': FNy_L, 'FNz_L': FNz_L,
            'FNx_R': FNx_R, 'FNy_R': FNy_R, 'FNz_R': FNz_R,
            'FNx_L2': FNx_L2, 'FNy_L2': FNy_L2, 'FNz_L2': FNz_L2,
            'FNx_R2': FNx_R2, 'FNy_R2': FNy_R2, 'FNz_R2': FNz_R2,
            'MLy': res_L1[6] + res_L2[6], 'MLz': res_L1[7] + res_L2[7],
            'MRy': res_R1[6] + res_R2[6], 'MRz': res_R1[7] + res_R2[7],
            'eL': res_L1[12], 'hrL': res_L1[11], 'eR': res_R1[12], 'hrR': res_R1[11],
            'eL2': res_L2[12], 'hrL2': res_L2[11], 'eR2': res_R2[12], 'hrR2': res_R2[11],
            'a0': a0, 'a02': a02,
            'rL': res_L1[13], 'rR': res_R1[13], 'rL2': res_L2[13], 'rR2': res_R2[13],
            'CreepForce_L': res_L1[9], 'CreepForce_R': res_R1[9],
            'CreepForce_L2': res_L2[9], 'CreepForce_R2': res_R2[9],
            'Ny_L': res_L1[4], 'Ny_R': res_R1[4],
            'Ny_L2': res_L2[4], 'Ny_R2': res_R2[4],
            'caseL': res_L1[10], 'caseR': res_R1[10],
            'wpointL1': res_L1[14], 'rpointL1': res_L1[15],
            'wpointR1': res_R1[14], 'rpointR1': res_R1[15]
            # ... 以及你需要的记录项可以轻松添加进这个字典
        }

class ContactMechanics:
    """
    轮轨接触力学底层计算引擎
    包含:
    1. Hertz 接触斑非线性压缩参数计算 (Gwr)
    2. Kalker 线性蠕滑力 + Shen-Hedrick-Elkins 非线性修正
    """

    # ================= 预加载静态查表数据 =================
    
    # 1. Fun_Contact2: Hertz 压缩参数 G8 查表
    _G8_table = np.array([
        [0.5, 0.08073], [1, 0.12126], [1.5, 0.15152], [2, 0.17923], [3, 0.22286], [4, 0.26277], 
        [6, 0.3274], [8, 0.38105], [10, 0.42838], [18, 0.57419], [20, 0.60425], [25, 0.67295], 
        [30, 0.72627], [35, 0.7753], [40, 0.81781], [45, 0.85476], [50, 0.88662], [55, 0.91388], 
        [60, 0.93758], [65, 0.95651], [70, 0.97197], [75, 0.98407], [80, 0.99298], [85, 0.99836], 
        [90, 1], [95, 0.99836], [100, 0.99298], [105, 0.98407], [110, 0.97197], [115, 0.95651],
        [120, 0.93758], [125, 0.91388], [130, 0.88662], [135, 0.85476], [140, 0.81781], [145, 0.7753], 
        [150, 0.72627], [155, 0.67295], [160, 0.60425], [162, 0.57419], [170, 0.42838], [172, 0.38105], 
        [174, 0.3274], [176, 0.26277], [177, 0.22286], [178, 0.17923], [178.5, 0.15252], [179, 0.12126], [179.5, 0.08073]
    ])
    _G8_x = _G8_table[:, 0]
    _G8_y = _G8_table[:, 1]

    # 2. Fun_creep_t: 椭圆积分参数查表
    _beta_table_x = np.array([0, 10, 20, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90])
    _beta_table_m = np.array([10000, 6.612, 3.778, 2.731, 2.397, 2.13, 1.926, 1.754, 1.611, 1.486, 1.378, 1.284, 1.202, 1.128, 1.061, 1])
    _beta_table_n = np.array([0, 0.319, 0.408, 0.493, 0.53, 0.567, 0.604, 0.641, 0.678, 0.717, 0.759, 0.802, 0.846, 0.893, 0.944, 1])

    # 3. Kalker 系数查表 (必须保证 x 轴升序！)
    _Cij_table1 = np.array([
        [0.1, 3.618, 2.522, 0.5246, 8.964], [0.2, 3.658, 2.636, 0.6442, 4.548], [0.3, 3.712, 2.762, 0.7498, 3.112],
        [0.4, 3.788, 2.9, 0.8538, 2.41], [0.5, 3.862, 3.036, 0.9572, 1.988], [0.6, 3.958, 3.174, 1.06, 1.716],
        [0.7, 4.042, 3.32, 1.17, 1.52], [0.8, 4.138, 3.458, 1.28, 1.38], [0.9, 4.232, 3.596, 1.39, 1.27], [1.0, 4.336, 3.732, 1.502, 1.184]
    ])
    
    # ⚠️ 极其关键的修复：MATLAB 原表是从 1.0 降到 0.1，Python 必须用 np.flipud 翻转成升序！
    _Cij_table2_raw = np.array([
        [1, 4.336, 3.732, 1.502, 1.184], [0.9, 4.436, 3.88, 1.626, 1.1], [0.8, 4.572, 4.07, 1.788, 1.0228],
        [0.7, 4.748, 4.302, 1.996, 0.9424], [0.6, 4.984, 4.608, 2.284, 0.8638], [0.5, 5.302, 5.032, 2.688, 0.7852],
        [0.4, 5.77, 5.646, 3.332, 0.7074], [0.3, 6.54, 6.622, 4.458, 0.6284], [0.2, 7.988, 8.47, 6.882, 0.549], [0.1, 11.94, 13.44, 15.28, 0.4664]
    ])
    _Cij_table2 = np.flipud(_Cij_table2_raw)

    @classmethod
    def calc_Gwr(cls, Rwx: np.ndarray, Rrx: np.ndarray, lanm: np.ndarray, r: float) -> np.ndarray:
        """
        计算 Hertz 接触斑非线性压缩参数 (支持标量或一维数组输入)
        """
        v = 0.3
        E = 2.06e11
        
        # 保护机制：处理曲率半径无穷大的情况
        inv_Rwx = np.where(np.isinf(Rwx), 0.0, 1.0 / Rwx)
        inv_Rrx = np.where(np.isinf(Rrx), 0.0, 1.0 / Rrx)
        
        A = np.cos(lanm) / (2 * r)
        B = 0.5 * (inv_Rwx + inv_Rrx)
        
        # 保护反余弦内部不越界 [-1, 1]
        inner = np.clip((B - A) / (B + A + 1e-12), -1.0, 1.0)
        beta = np.rad2deg(np.arccos(inner))
        
        G8 = np.interp(beta, cls._G8_x, cls._G8_y)
        Gwr = G8 * ((1.5 * (1 - v**2) / E)**2 * (A + B))**(1/3)
        return Gwr

    @classmethod
    def calc_creep_forces(cls, ksix, ksiy, ksisp, Rw, rw, rr, N):
        """
        计算非线性蠕滑力 (支持标量或数组)
        :param N: 法向接触力，如果 N=0，将返回全零的蠕滑力
        """
        # 如果输入是标量且 N==0，直接快速返回
        if np.isscalar(N) and N == 0:
            return 0.0, 0.0, 0.0

        # ⚠️ 关键修复：在转换为数组前记录输入是否为标量
        is_scalar_input = np.isscalar(N)

        # 确保输入为 NumPy 数组，以实现向量化计算
        ksix, ksiy, ksisp, Rw, rw, rr, N = map(np.atleast_1d, (ksix, ksiy, ksisp, Rw, rw, rr, N))
        
        v = 0.25
        f = 0.3
        E = 2.06e11
        G = E / (2 * (1 + v))
        
        # 掩码：仅在法向力 > 0 的接触点进行计算
        mask = N > 0
        if not np.any(mask):
            res = np.zeros_like(N)
            return (res.copy(), res.copy(), res.copy()) if res.size > 1 else (0.0, 0.0, 0.0)

        FX, FY, MZ = np.zeros_like(N), np.zeros_like(N), np.zeros_like(N)
        
        # 为了代码简洁，提取有效元素
        _ksix, _ksiy, _ksisp = ksix[mask], ksiy[mask], ksisp[mask]
        _Rw, _rw, _rr, _N = Rw[mask], rw[mask], rr[mask], N[mask]

        inv_Rw = np.where(np.isinf(_Rw), 0.0, 1.0 / _Rw)
        inv_rw = np.where(np.isinf(_rw), 0.0, 1.0 / _rw)
        inv_rr = np.where(np.isinf(_rr), 0.0, 1.0 / _rr)

        rho = 1.0 / (0.25 * (inv_Rw + inv_rw + inv_rr) + 1e-12)
        inner = np.clip(0.25 * rho * np.abs(inv_Rw - inv_rw - inv_rr), -1.0, 1.0)
        beta = np.rad2deg(np.arccos(inner))

        m = np.interp(beta, cls._beta_table_x, cls._beta_table_m)
        n = np.interp(beta, cls._beta_table_x, cls._beta_table_n)

        # 判断椭圆半轴条件
        ratio_cond = (rho * inv_Rw) <= 2
        ae_coeff = np.where(ratio_cond, 0.1506 * m, 0.1506 * n)
        be_coeff = np.where(ratio_cond, 0.1506 * n, 0.1506 * m)

        factor = (_N * _Rw)**(1/3)
        a = ae_coeff * (rho * inv_Rw)**(1/3) * 1e-3 * factor
        b = be_coeff * (rho * inv_Rw)**(1/3) * 1e-3 * factor
        ab = a * b

        judge_cond = b > a
        g = np.where(judge_cond, a / b, b / a)
        an = np.log(16.0 / (g**2 + 1e-12))

        # --- 初始化 Cij ---
        C11, C22, C23, C33 = np.zeros_like(g), np.zeros_like(g), np.zeros_like(g), np.zeros_like(g)

        # ---------------- 分支 1: judge == 1 (b > a) ----------------
        m1 = judge_cond
        if np.any(m1):
            g1 = g[m1]
            an1 = an[m1]
            # sub-branch 1: g < 0.1
            m1_small = g1 < 0.1
            if np.any(m1_small):
                C11[np.flatnonzero(m1)[m1_small]] = np.pi**2 / (4 * (1 - v))
                C22[np.flatnonzero(m1)[m1_small]] = np.pi**2 / 4
                C23[np.flatnonzero(m1)[m1_small]] = np.pi * np.sqrt(g1[m1_small]) / (3 * (1 - v)) * (1 + v * (0.5 * an1[m1_small] + np.log(4) - 5))
                C33[np.flatnonzero(m1)[m1_small]] = np.pi**2 / (16 * (1 - v) * g1[m1_small])
            
            # sub-branch 2: g >= 0.1 (查表 1)
            m1_large = ~m1_small
            if np.any(m1_large):
                g_eval = g1[m1_large]
                C11[np.flatnonzero(m1)[m1_large]] = np.interp(g_eval, cls._Cij_table1[:, 0], cls._Cij_table1[:, 1])
                C22[np.flatnonzero(m1)[m1_large]] = np.interp(g_eval, cls._Cij_table1[:, 0], cls._Cij_table1[:, 2])
                C23[np.flatnonzero(m1)[m1_large]] = np.interp(g_eval, cls._Cij_table1[:, 0], cls._Cij_table1[:, 3])
                C33[np.flatnonzero(m1)[m1_large]] = np.interp(g_eval, cls._Cij_table1[:, 0], cls._Cij_table1[:, 4])

        # ---------------- 分支 2: judge == 2 (b <= a) ----------------
        m2 = ~judge_cond
        if np.any(m2):
            g2 = g[m2]
            an2 = an[m2]
            # sub-branch 1: g < 0.1
            m2_small = g2 < 0.1
            if np.any(m2_small):
                an_val = an2[m2_small]
                g_val = g2[m2_small]
                denom1 = an_val - 2 * v
                denom2 = (1 - v) * an_val + 2 * v
                denom3 = (1 - v) * an_val - 2 + 4 * v
                
                C11[np.flatnonzero(m2)[m2_small]] = 2 * np.pi / (denom1 * g_val) * (1 + (3 - np.log(4)) / denom1)
                C22[np.flatnonzero(m2)[m2_small]] = (2 * np.pi / g_val) * (1 + (1 - v) * (3 - np.log(4)) / denom2) / denom2
                C23[np.flatnonzero(m2)[m2_small]] = (2 * np.pi / (3 * g_val * np.sqrt(g_val))) / denom3
                C33[np.flatnonzero(m2)[m2_small]] = (np.pi / 4) * (1 - (v * an_val - 2) / denom3)

            # sub-branch 2: g >= 0.1 (查表 2，注意表 2 已在类初始化时翻转为升序！)
            m2_large = ~m2_small
            if np.any(m2_large):
                g_eval = g2[m2_large]
                C11[np.flatnonzero(m2)[m2_large]] = np.interp(g_eval, cls._Cij_table2[:, 0], cls._Cij_table2[:, 1])
                C22[np.flatnonzero(m2)[m2_large]] = np.interp(g_eval, cls._Cij_table2[:, 0], cls._Cij_table2[:, 2])
                C23[np.flatnonzero(m2)[m2_large]] = np.interp(g_eval, cls._Cij_table2[:, 0], cls._Cij_table2[:, 3])
                C33[np.flatnonzero(m2)[m2_large]] = np.interp(g_eval, cls._Cij_table2[:, 0], cls._Cij_table2[:, 4])

        C32 = -C23

        # --- Kalker 线性蠕滑力 ---
        f11 = G * ab * C11
        f22 = G * ab * C22
        f23 = G * ab**1.5 * C23
        f33 = G * ab**2 * C33

        Fx = -f11 * _ksix
        Fy = -f22 * _ksiy - f23 * _ksisp
        Mz = f23 * _ksiy - f33 * _ksisp

        # --- Shen-Hedrick-Elkins 非线性修正 ---
        F0 = np.sqrt(Fx**2 + Fy**2)
        fN = f * _N
        
        # Shen氏公式：只有 F0 <= 3*mu*N 时才使用三次抛物线，否则完全滑动取 mu*N
        F_she = np.where(F0 <= 3 * fN, fN * ((F0 / (fN+1e-12)) - (1/3)*(F0 / (fN+1e-12))**2 + (1/27)*(F0 / (fN+1e-12))**3), fN)
        
        eps = F_she / (F0 + 1e-12)
        
        # 将算好的力写回数组
        FX[mask] = eps * Fx
        FY[mask] = eps * Fy
        MZ[mask] = eps * Mz

        # 如果输入是标量，则返回标量；否则返回数组
        if is_scalar_input:
            return float(FX[0]), float(FY[0]), float(MZ[0])
        return FX, FY, MZ