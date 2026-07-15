r'''
FilePath: \VTCM_PYTHON\physics_modules\suspension.py
Description: 车辆系统一系及二系悬挂力元核心计算模块 (极致性能向量化版)
'''
import numpy as np
# 假设这些配置类和 StepState 在您的项目中已经定义好
from configs.parameters import VehicleParams, Antiyawer_parameters, ExtraForceElements_parameters
from configs.topology import StepState  # 引入我们在拓扑中定义的状态数据类

class SuspensionSystem:
    def __init__(self, veh_params: VehicleParams, antiyawer_params: Antiyawer_parameters, extra_params: ExtraForceElements_parameters = None):
        """初始化悬挂系统力元计算模块"""
        self.p = veh_params
        self.ap = antiyawer_params
        self.ep = extra_params

        # 基础力元开关配置（默认全开）
        self.switches = {
            'antiyawer': 1.0,  # 抗蛇行减振器
            'pvd': 1.0,        # 一系垂向减振器
            'node': 1.0,       # 转臂结点
            'svd': 1.0,        # 二系垂向减振器
            'sld': 1.0,        # 二系横向减振器
            'antiroll': 1.0    # 抗侧滚扭杆
        }

        # 预计算静态承载力
        g = 9.81
        if self.p.category == '货车':
            self.static_force_sec = ((self.p.Mc + 2 * self.p.MB) * g) / 4.0
            self.static_force_pre = ((self.p.Mc + 2 * self.p.MB + 4 * self.p.Mt) * g) / 8.0
            for key in self.switches:
                self.switches[key] = 0.0
                
        elif self.p.category == '客车':
            self.static_force_sec = (self.p.Mc * g) / 4.0
            self.static_force_pre = ((self.p.Mc + 2 * self.p.Mt) * g) / 8.0
            
        elif self.p.category == '机车':
            self.static_force_sec = (self.p.Mc * g) / 4.0
            self.static_force_pre = ((self.p.Mc + 2 * self.p.Mt) * g) / 8.0
            self.switches['node'] = 0.0  # 机车通常不考虑转臂结点力元

    def _compute_freight_forces(self, state: StepState, R_curve, d_invR_dt):
        """货车专用力元计算框架 (待补充干摩擦斜楔公式)"""
        print(" -> [警告] 正在调用重载货车测试框架，摩擦斜楔等非线性公式暂未实现！")
        raise NotImplementedError("货车具体公式后续补充")

    def _compute_loco_forces(self, state: StepState, R_curve, d_invR_dt):
        """机车专用力元计算框架 (待补充单拉杆/牵引装置公式)"""
        print(" -> [警告] 正在调用机车测试框架，核心力元公式暂未实现！")
        raise NotImplementedError("机车具体公式后续补充")
    
    def _compute_passenger_forces(self, state: StepState, R_curve, d_invR_dt, include_extra=False, axlebox_state=None):
        """
        核心客车悬挂力元计算
        """
        p, ap, ep = self.p, self.ap, self.ep
        forces = {}
        
        # 映射关系：4个轮对对应的前后转向架索引
        bogie_idx = np.array([0, 0, 1, 1])
        # [修复] 严格对应 MATLAB 的 (-1)^i (i=1,2,3,4)，实现正确的反对称性
        sign_pow = np.array([-1, 1, -1, 1])
        sign_pow_sec = np.array([-1, 1])
        axlebox_state = {} if axlebox_state is None else axlebox_state

        def axlebox_vector(name):
            value = np.asarray(axlebox_state.get(name, np.zeros(4)), dtype=float)
            if value.shape != (4,):
                raise ValueError(f"axlebox_state[{name!r}] must have shape (4,), got {value.shape}")
            return value

        axle_spin_L = axlebox_vector('spin_L')
        axle_spin_R = axlebox_vector('spin_R')
        axle_spin_vel_L = axlebox_vector('spin_vel_L')
        axle_spin_vel_R = axlebox_vector('spin_vel_R')

        # ================= 0. 极速状态解析 (零拷贝视图) =================
        # 车体状态 (索引 0~4)
        Yc, Zc, Rollc, Spinc, Yawc = state.XCar[0:5]
        dYc, dZc, dRollc, dSpinc, dYawc = state.VCar[0:5]
        
        # 构架状态 (巧妙利用步长切片提取前、后转向架状态)
        Yt = state.XCar[5:15:5];   dYt = state.VCar[5:15:5]
        Zt = state.XCar[6:15:5];   dZt = state.VCar[6:15:5]
        Rollt = state.XCar[7:15:5]; dRollt = state.VCar[7:15:5]
        Spint = state.XCar[8:15:5]; dSpint = state.VCar[8:15:5]
        Yawt = state.XCar[9:15:5];  dYawt = state.VCar[9:15:5]

        # ================= 1. 一系悬挂计算 =================
        Fxf_L, Fxf_R, Fyf_L, Fyf_R, Fzf_L, Fzf_R = [np.zeros(4) for _ in range(6)]

        for i in range(4):
            b_id = bogie_idx[i]
            s_p = sign_pow[i]

            # 曲线几何稳态与动态补偿项
            curve_disp_fx = s_p * p.dw * (p.Lt / R_curve[b_id])
            curve_vel_fx = s_p * p.dw * (p.Lt * d_invR_dt[b_id])
            curve_disp_fy = (p.Lt**2) / (2 * R_curve[b_id])
            curve_vel_fy = (p.Lt**2 * d_invR_dt[b_id] / 2)
            
            # 一系纵向力
            Fxf_L[i] = p.Kpx * (p.dw * Yawt[b_id] + p.Htw * Spint[b_id] - p.dw * state.X_YawW[i] - curve_disp_fx) + \
                       p.Cpx * (p.dw * dYawt[b_id] + p.Htw * dSpint[b_id] - p.dw * state.V_YawW[i] - curve_vel_fx)
            Fxf_R[i] = p.Kpx * (-p.dw * Yawt[b_id] + p.Htw * Spint[b_id] + p.dw * state.X_YawW[i] + curve_disp_fx) + \
                       p.Cpx * (-p.dw * dYawt[b_id] + p.Htw * dSpint[b_id] + p.dw * state.V_YawW[i] + curve_vel_fx)
            
            # 一系横向力
            Fyf_L[i] = p.Kpy * (state.X_YW[i] - Yt[b_id] + p.Htw * Rollt[b_id] + s_p * p.Lt * Yawt[b_id] + curve_disp_fy) + \
                       p.Cpy * (state.V_YW[i] - dYt[b_id] + p.Htw * dRollt[b_id] + s_p * p.Lt * dYawt[b_id] + curve_vel_fy)
            Fyf_R[i] = Fyf_L[i]

            # 一系垂向力 [修复了原本符号错乱的致命Bug]
            Fzf_L[i] = p.Kpz * (Zt[b_id] - state.X_ZW[i] - s_p * p.Lt * Spint[b_id] + p.dw * state.X_RollW[i] - p.dw * Rollt[b_id]) + \
                       p.Cpz * (dZt[b_id] - state.V_ZW[i] - s_p * p.Lt * dSpint[b_id] + p.dw * state.V_RollW[i] - p.dw * dRollt[b_id]) + self.static_force_pre
            Fzf_R[i] = p.Kpz * (Zt[b_id] - state.X_ZW[i] - s_p * p.Lt * Spint[b_id] - p.dw * state.X_RollW[i] + p.dw * Rollt[b_id]) + \
                       p.Cpz * (dZt[b_id] - state.V_ZW[i] - s_p * p.Lt * dSpint[b_id] - p.dw * state.V_RollW[i] + p.dw * dRollt[b_id]) + self.static_force_pre
            
        forces.update({'Fxf_L': Fxf_L, 'Fxf_R': Fxf_R, 'Fyf_L': Fyf_L, 'Fyf_R': Fyf_R, 'Fzf_L': Fzf_L, 'Fzf_R': Fzf_R})

        # ================= 2. 二系悬挂计算 =================
        Fxt_L, Fxt_R, Fyt_L, Fyt_R, Fzt_L, Fzt_R, Mr, Fxs_L, Fxs_R = [np.zeros(2) for _ in range(9)]
        
        for i in range(2):
            s_p = sign_pow_sec[i]

            curve_disp_fx_sec = s_p * p.ds * (p.Lc / R_curve[2])
            curve_vel_fx_sec = s_p * p.ds * (p.Lc * d_invR_dt[2])
            curve_disp_fy_sec = (p.Lc**2) / (2 * R_curve[2])
            curve_vel_fy_sec = (p.Lc**2 * d_invR_dt[2] / 2)
            
            # 二系纵向力
            Fxt_L[i] = p.Ksx * (p.HcB * Spinc + p.HBt * Spint[i] + p.ds * Yawc - p.ds * Yawt[i] - curve_disp_fx_sec) + \
                       p.Csx * (p.HcB * dSpinc + p.HBt * dSpint[i] + p.ds * dYawc - p.ds * dYawt[i] - curve_vel_fx_sec)
            Fxt_R[i] = p.Ksx * (p.HcB * Spinc + p.HBt * Spint[i] - p.ds * Yawc + p.ds * Yawt[i] + curve_disp_fx_sec) + \
                       p.Csx * (p.HcB * dSpinc + p.HBt * dSpint[i] - p.ds * dYawc + p.ds * dYawt[i] + curve_vel_fx_sec)
            
            # 二系横向力 [修复: s_p * p.Lc * Yawc 符号纠正为 '+']
            Fyt_L[i] = p.Ksy * (Yt[i] - Yc + p.HBt * Rollt[i] + p.HcB * Rollc + s_p * p.Lc * Yawc + curve_disp_fy_sec) + \
                       p.Csy * (dYt[i] - dYc + p.HBt * dRollt[i] + p.HcB * dRollc + s_p * p.Lc * dYawc + curve_vel_fy_sec)
            Fyt_R[i] = Fyt_L[i]
            
            # 二系垂向力
            Fzt_L[i] = p.Ksz * (Zc - Zt[i] + p.ds * Rollt[i] - p.ds * Rollc + s_p * p.Lc * Spinc) + \
                       p.Csz * (dZc - dZt[i] + p.ds * dRollt[i] - p.ds * dRollc + s_p * p.Lc * dSpinc) + self.static_force_sec
            Fzt_R[i] = p.Ksz * (Zc - Zt[i] - p.ds * Rollt[i] + p.ds * Rollc + s_p * p.Lc * Spinc) + \
                       p.Csz * (dZc - dZt[i] - p.ds * dRollt[i] + p.ds * dRollc + s_p * p.Lc * dSpinc) + self.static_force_sec

            # 抗侧滚扭杆
            Mr[i] = p.Krx * (Rollc - Rollt[i]) * self.switches['antiroll']

            # 抗蛇行减振器 (带有液压减振器卸荷阀物理特性的 np.interp)
            if ap:
                vxct_L = ap.dsc * dYawc - ap.dsc * dYawt[i] + p.HcB * dSpinc + p.HBt * dSpint[i]
                vxct_R = -ap.dsc * dYawc + ap.dsc * dYawt[i] + p.HcB * dSpinc + p.HBt * dSpint[i]
                xxct_L = ap.dsc * Yawc - ap.dsc * Yawt[i] + p.HcB * Spinc + p.HBt * Spint[i]
                xxct_R = -ap.dsc * Yawc + ap.dsc * Yawt[i] + p.HcB * Spinc + p.HBt * Spint[i]
                
                # 提示: np.interp 默认超出边界时会输出端点值，这在物理上极其完美地模拟了液压减振器的“卸荷阀恒力特性”
                Fxs_L[i] = self.switches['antiyawer'] * (np.interp(vxct_L, ap.yaw_damper_v, ap.yaw_damper_f) + ap.kantiyawer * xxct_L)
                Fxs_R[i] = self.switches['antiyawer'] * (np.interp(vxct_R, ap.yaw_damper_v, ap.yaw_damper_f) + ap.kantiyawer * xxct_R)

        forces.update({'Fxt_L': Fxt_L, 'Fxt_R': Fxt_R, 'Fyt_L': Fyt_L, 'Fyt_R': Fyt_R, 'Fzt_L': Fzt_L, 'Fzt_R': Fzt_R, 'Mr': Mr, 'Fxs_L': Fxs_L, 'Fxs_R': Fxs_R})

        # ================= 3. 额外力元 (一系 PVD & 转臂 Node & 二系 SVD/SLD) =================
        Fpvdz_L, Fpvdz_R = np.zeros(4), np.zeros(4)
        Fnodex_L, Fnodey_L, Fnodez_L = np.zeros(4), np.zeros(4), np.zeros(4)
        Fnodex_R, Fnodey_R, Fnodez_R = np.zeros(4), np.zeros(4), np.zeros(4)
        Fsvdz_L, Fsvdz_R = np.zeros(2), np.zeros(2)
        Fsldy_L, Fsldy_R = np.zeros(2), np.zeros(2)

        if ep and include_extra:
            # 一系额外力元
            for i in range(4):
                b_id = bogie_idx[i]
                s_p = sign_pow[i]

                # 当前35-DOF模型默认轴箱转角为零；可通过预留接口显式传入
                spinL_i = axle_spin_L[i]
                spinR_i = axle_spin_R[i]
                dspinL_i = axle_spin_vel_L[i]
                dspinR_i = axle_spin_vel_R[i]

                # 一系垂向减振器 PVD
                Fpvdz_L[i] = ep.Kz_pvd * (Zt[b_id] - state.X_ZW[i] + s_p * ep.Lpvdx_Bogie * Spint[b_id] + ep.dpvd * state.X_RollW[i] - ep.dpvd * Rollt[b_id] - s_p * ep.Lpvdx_axlebox * spinL_i) + \
                             ep.Cz_pvd * (dZt[b_id] - state.V_ZW[i] + s_p * ep.Lpvdx_Bogie * dSpint[b_id] + ep.dpvd * state.V_RollW[i] - ep.dpvd * dRollt[b_id] - s_p * ep.Lpvdx_axlebox * dspinL_i)
                Fpvdz_R[i] = ep.Kz_pvd * (Zt[b_id] - state.X_ZW[i] + s_p * ep.Lpvdx_Bogie * Spint[b_id] - ep.dpvd * state.X_RollW[i] + ep.dpvd * Rollt[b_id] - s_p * ep.Lpvdx_axlebox * spinR_i) + \
                             ep.Cz_pvd * (dZt[b_id] - state.V_ZW[i] + s_p * ep.Lpvdx_Bogie * dSpint[b_id] - ep.dpvd * state.V_RollW[i] + ep.dpvd * dRollt[b_id] - s_p * ep.Lpvdx_axlebox * dspinR_i)
                Fpvdz_L[i] *= self.switches['pvd']; Fpvdz_R[i] *= self.switches['pvd']

                # 转臂结点 Node
                Fnodex_L[i] = ep.Knodex * (ep.dnode * Yawt[b_id] + ep.Ht_node * Spint[b_id] - ep.dnode * state.X_YawW[i] + s_p * ep.Haxlebox_node * spinL_i)
                Fnodex_R[i] = ep.Knodex * (-ep.dnode * Yawt[b_id] + ep.Ht_node * Spint[b_id] + ep.dnode * state.X_YawW[i] + s_p * ep.Haxlebox_node * spinR_i)
                Fnodey_L[i] = ep.Knodey * (state.X_YW[i] - Yt[b_id] + ep.Ht_node * Rollt[b_id] + s_p * ep.Lnodex_Bogie * Yawt[b_id] + s_p * ep.Lnode_axlebox * state.X_YawW[i])
                Fnodey_R[i] = Fnodey_L[i]
                Fnodez_L[i] = ep.Knodez * (Zt[b_id] - state.X_ZW[i] + s_p * ep.Lnodex_Bogie * Spint[b_id] + ep.dnode * state.X_RollW[i] - ep.dnode * Rollt[b_id] + s_p * ep.Lnode_axlebox * spinL_i)
                Fnodez_R[i] = ep.Knodez * (Zt[b_id] - state.X_ZW[i] + s_p * ep.Lnodex_Bogie * Spint[b_id] - ep.dnode * state.X_RollW[i] + ep.dnode * Rollt[b_id] + s_p * ep.Lnode_axlebox * spinR_i)
                
                Fnodex_L[i] *= self.switches['node']; Fnodex_R[i] *= self.switches['node']
                Fnodey_L[i] *= self.switches['node']; Fnodey_R[i] *= self.switches['node']
                Fnodez_L[i] *= self.switches['node']; Fnodez_R[i] *= self.switches['node']

            # 二系额外力元
            for i in range(2):
                s_p = sign_pow_sec[i]

                # 二系垂向减振器 SVD
                Fsvdz_L[i] = ep.Kz_svd * (Zc - Zt[i] + ep.dsvd * Rollt[i] - ep.dsvd * Rollc + s_p * ep.Lsvd_Car * Spinc) + \
                             ep.Cz_svd * (dZc - dZt[i] + ep.dsvd * dRollt[i] - ep.dsvd * dRollc + s_p * ep.Lsvd_Car * dSpinc)
                Fsvdz_R[i] = ep.Kz_svd * (Zc - Zt[i] - ep.dsvd * Rollt[i] + ep.dsvd * Rollc + s_p * ep.Lsvd_Car * Spinc) + \
                             ep.Cz_svd * (dZc - dZt[i] - ep.dsvd * dRollt[i] + ep.dsvd * dRollc + s_p * ep.Lsvd_Car * dSpinc)
                Fsvdz_L[i] *= self.switches['svd']; Fsvdz_R[i] *= self.switches['svd']

                # 二系横向减振器 SLD
                rx_sld_L = Yt[i] - Yc + ep.Ht_sld * Rollt[i] + ep.Hc_sld * Rollc + s_p * ep.Lsld_Car_L[i] * Yawc + ep.Lsld_Bogie * Yawt[i]
                rx_sld_R = Yt[i] - Yc + ep.Ht_sld * Rollt[i] + ep.Hc_sld * Rollc + s_p * ep.Lsld_Car_R[i] * Yawc - ep.Lsld_Bogie * Yawt[i]
                rv_sld_L = dYt[i] - dYc + ep.Ht_sld * dRollt[i] + ep.Hc_sld * dRollc + s_p * ep.Lsld_Car_L[i] * dYawc + ep.Lsld_Bogie * dYawt[i]
                rv_sld_R = dYt[i] - dYc + ep.Ht_sld * dRollt[i] + ep.Hc_sld * dRollc + s_p * ep.Lsld_Car_R[i] * dYawc - ep.Lsld_Bogie * dYawt[i]
                
                Fsldy_L[i] = self.switches['sld'] * (ep.Ky_sld * rx_sld_L + np.interp(rv_sld_L, ep.lat_damper_v, ep.lat_damper_f))
                Fsldy_R[i] = self.switches['sld'] * (ep.Ky_sld * rx_sld_R + np.interp(rv_sld_R, ep.lat_damper_v, ep.lat_damper_f))

        forces.update({
            'Fpvdz_L': Fpvdz_L, 'Fpvdz_R': Fpvdz_R, 
            'Fnodex_L': Fnodex_L, 'Fnodey_L': Fnodey_L, 'Fnodez_L': Fnodez_L,
            'Fnodex_R': Fnodex_R, 'Fnodey_R': Fnodey_R, 'Fnodez_R': Fnodez_R,
            'Fsvdz_L': Fsvdz_L, 'Fsvdz_R': Fsvdz_R, 
            'Fsldy_L': Fsldy_L, 'Fsldy_R': Fsldy_R
        })

        return forces
            
    def compute_forces(self, state: StepState, R_curve: np.ndarray = np.array([np.inf, np.inf, np.inf]), d_invR_dt: np.ndarray = np.zeros(3), include_extra: bool = False, axlebox_state=None):
        """
        统一的悬挂力计算入口
        :param state: 从 SystemTopology 提取的 StepState 对象
        :param R_curve: [R_bogie_front, R_bogie_rear, R_carbody] 曲线半径
        """
        if self.p.category == '货车':
            return self._compute_freight_forces(state, R_curve, d_invR_dt)
        elif self.p.category == '机车':
            return self._compute_loco_forces(state, R_curve, d_invR_dt)
        elif self.p.category == '客车':
            return self._compute_passenger_forces(state, R_curve, d_invR_dt, include_extra=include_extra, axlebox_state=axlebox_state)
