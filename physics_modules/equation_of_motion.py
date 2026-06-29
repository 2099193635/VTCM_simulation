import numpy as np

class GeneralForceAssembler:
    """系统广义力组装器 (全系统耦合版: 车辆 + 钢轨模态 + 轨下结构)"""
    
    def __init__(self, veh_params, integration_params, rail_params, subrail_params, mode_params, anitiyawer_params=None):
        self.vp = veh_params
        self.ip = integration_params
        self.rp = rail_params
        self.sp = subrail_params
        self.mp = mode_params
        self.ap = anitiyawer_params
        
        # ====================================================================
        # 🌟 预计算：钢轨模态抗弯/抗扭刚度向量 (极大提升运行时极速)
        # 对应 MATLAB: EIY = ((E*Iy/mr)*(pi/Ljs*n)^4)
        # ====================================================================
        nv_modes = np.arange(1, self.mp.NV + 1)
        nl_modes = np.arange(1, self.mp.NL + 1)
        nt_modes = np.arange(1, self.mp.NT + 1)

        pi_Ljs = np.pi / self.ip.Ljs
        
        self.EIY = (self.rp.E * self.rp.Iy / self.rp.mr) * (pi_Ljs * nv_modes)**4
        self.EIZ = (self.rp.E * self.rp.Iz / self.rp.mr) * (pi_Ljs * nl_modes)**4
        self.GIO = (self.rp.Gk / (self.rp.pr * self.rp.I0)) * (pi_Ljs * nt_modes)**2

    def assemble_GF_SYSTEM(self, state, susp_forces, wr_forces, fastener_forces=None, subrail_forces=None, rail_modal_sys=None, rail_states=None):
        """
        将悬挂力、轮轨接触力、轨道力投影并组装为系统全局广义力向量
        """
        vp = self.vp
        g = self.ip.g
        omega = self.ip.omega
        ap = self.ap

        # ====================================================================
        # 1. 提取悬挂力 (Suspension Forces)
        # ====================================================================
        Fxf_L, Fxf_R = susp_forces['Fxf_L'], susp_forces['Fxf_R']
        Fyf_L, Fyf_R = susp_forces['Fyf_L'], susp_forces['Fyf_R']
        Fzf_L, Fzf_R = susp_forces['Fzf_L'], susp_forces['Fzf_R']
        
        Fxt_L, Fxt_R = susp_forces['Fxt_L'], susp_forces['Fxt_R']
        Fyt_L, Fyt_R = susp_forces['Fyt_L'], susp_forces['Fyt_R']
        Fzt_L, Fzt_R = susp_forces['Fzt_L'], susp_forces['Fzt_R']
        
        Mr = susp_forces['Mr']
        Fxs_L, Fxs_R = susp_forces['Fxs_L'], susp_forces['Fxs_R']

        # ====================================================================
        # 2. 提取轮轨接触力与几何信息
        # ====================================================================
        TFx_L, TFy_L, TFz_L = wr_forces['FNx_L'], wr_forces['FNy_L'], wr_forces['FNz_L']
        TFx_R, TFy_R, TFz_R = wr_forces['FNx_R'], wr_forces['FNy_R'], wr_forces['FNz_R']
        TFx_L2, TFy_L2, TFz_L2 = wr_forces['FNx_L2'], wr_forces['FNy_L2'], wr_forces['FNz_L2']
        TFx_R2, TFy_R2, TFz_R2 = wr_forces['FNx_R2'], wr_forces['FNy_R2'], wr_forces['FNz_R2']
        
        MLy, MLz = wr_forces['MLy'], wr_forces['MLz']
        MRy, MRz = wr_forces['MRy'], wr_forces['MRz']
        rL, rR = wr_forces['rL'], wr_forces['rR']
        rL2, rR2 = wr_forces['rL2'], wr_forces['rR2']
        a0, a02 = wr_forces['a0'], wr_forces['a02']
        
        # 预留给钢轨系统的触点扭转力臂
        hrL, eL = wr_forces['hrL'], wr_forces['eL']
        hrR, eR = wr_forces['hrR'], wr_forces['eR']
        hrL2, eL2 = wr_forces['hrL2'], wr_forces['eL2']
        hrR2, eR2 = wr_forces['hrR2'], wr_forces['eR2']

        # ====================================================================
        # 3. 极速向量化：车辆系统广义力
        # ====================================================================
        VWRoll, VWSpin, VWYaw, XWYaw = state.V_RollW, state.V_SpinW, state.V_YawW, state.X_YawW

        GF_Wheelset_Y = -Fyf_L - Fyf_R + TFy_L + TFy_R + TFy_L2 + TFy_R2
        GF_Wheelset_Z = -TFz_L - TFz_R + Fzf_L + Fzf_R + vp.Mw * g - TFz_L2 - TFz_R2
        
        GF_Wheelset_Roll = vp.Jwy * (VWSpin - omega) * VWYaw \
                           + a0 * (TFz_L - TFz_R) - rL * TFy_L - rR * TFy_R + vp.dw * (Fzf_R - Fzf_L) \
                           + a02 * (TFz_L2 - TFz_R2) - rL2 * TFy_L2 - rR2 * TFy_R2
                           
        GF_Wheelset_Yaw = vp.Jwy * VWRoll * (VWSpin - omega) \
                          + a0 * (TFx_L - TFx_R) + a0 * XWYaw * (TFy_L - TFy_R) + MLz + MRz + vp.dw * (Fxf_L - Fxf_R) \
                          + a02 * (TFx_L2 - TFx_R2) + a02 * XWYaw * (TFy_L2 - TFy_R2)
                          
        GF_Wheelset_Spin = rR * TFx_R + rL * TFx_L + rR * XWYaw * TFy_R + rL * XWYaw * TFy_L + MLy + MRy \
                           + rR2 * TFx_R2 + rL2 * TFx_L2 + rR2 * XWYaw * TFy_R2 + rL2 * XWYaw * TFy_L2

        idx1, idx2 = np.array([0, 2]), np.array([1, 3])
        GF_Bogie_Y = Fyf_L[idx1] + Fyf_L[idx2] - Fyt_L + Fyf_R[idx1] + Fyf_R[idx2] - Fyt_R
        GF_Bogie_Z = Fzt_L - Fzf_L[idx1] - Fzf_L[idx2] + Fzt_R - Fzf_R[idx1] - Fzf_R[idx2] + vp.Mt * g
        GF_Bogie_Roll = -(Fyf_L[idx1] + Fyf_R[idx1] + Fyf_L[idx2] + Fyf_R[idx2]) * vp.Htw \
                        + (Fzf_L[idx1] + Fzf_L[idx2] - Fzf_R[idx1] - Fzf_R[idx2]) * vp.dw \
                        + (Fzt_R - Fzt_L) * vp.ds - (Fyt_L + Fyt_R) * vp.HBt + Mr
        GF_Bogie_Yaw = (Fyf_L[idx1] + Fyf_R[idx1] - Fyf_L[idx2] - Fyf_R[idx2]) * vp.Lt \
                       + (Fxf_R[idx1] + Fxf_R[idx2] - Fxf_L[idx1] - Fxf_L[idx2]) * vp.dw \
                       + (Fxt_L - Fxt_R) * vp.ds + (Fxs_L - Fxs_R) * ap.dsc
        GF_Bogie_Spin = -(Fzf_L[idx1] + Fzf_R[idx1] - Fzf_L[idx2] - Fzf_R[idx2]) * vp.Lt \
                        - (Fxf_L[idx1] + Fxf_R[idx1] + Fxf_L[idx2] + Fxf_R[idx2]) * vp.Htw \
                        - (Fxt_L + Fxt_R) * vp.HBt - (Fxs_L + Fxs_R) * vp.HBt

        GF_Car_Y = np.sum(Fyt_L) + np.sum(Fyt_R)
        GF_Car_Z = -np.sum(Fzt_L) - np.sum(Fzt_R) + vp.Mc * g
        GF_Car_Roll = -np.sum(Fyt_L + Fyt_R) * vp.HcB + (Fzt_L[0] + Fzt_L[1] - Fzt_R[0] - Fzt_R[1]) * vp.ds - np.sum(Mr)
        GF_Car_Spin = (Fzt_L[0] + Fzt_R[0] - Fzt_L[1] - Fzt_R[1]) * vp.Lc - np.sum(Fxt_L + Fxt_R) * vp.HcB - np.sum(Fxs_L + Fxs_R) * vp.HcB
        GF_Car_Yaw = (Fyt_L[0] + Fyt_R[0] - Fyt_L[1] - Fyt_R[1]) * vp.Lc + np.sum(Fxt_R) * vp.ds - np.sum(Fxt_L) * vp.ds + np.sum(Fxs_R) * ap.dsc - np.sum(Fxs_L) * ap.dsc

        GF_VEHICLE = np.concatenate([
            [GF_Car_Y, GF_Car_Z, GF_Car_Roll, GF_Car_Spin, GF_Car_Yaw],
            GF_Bogie_Y[0:1], GF_Bogie_Z[0:1], GF_Bogie_Roll[0:1], GF_Bogie_Spin[0:1], GF_Bogie_Yaw[0:1],
            GF_Bogie_Y[1:2], GF_Bogie_Z[1:2], GF_Bogie_Roll[1:2], GF_Bogie_Spin[1:2], GF_Bogie_Yaw[1:2],
            GF_Wheelset_Y[0:1], GF_Wheelset_Z[0:1], GF_Wheelset_Roll[0:1], GF_Wheelset_Spin[0:1], GF_Wheelset_Yaw[0:1],
            GF_Wheelset_Y[1:2], GF_Wheelset_Z[1:2], GF_Wheelset_Roll[1:2], GF_Wheelset_Spin[1:2], GF_Wheelset_Yaw[1:2],
            GF_Wheelset_Y[2:3], GF_Wheelset_Z[2:3], GF_Wheelset_Roll[2:3], GF_Wheelset_Spin[2:3], GF_Wheelset_Yaw[2:3],
            GF_Wheelset_Y[3:4], GF_Wheelset_Z[3:4], GF_Wheelset_Roll[3:4], GF_Wheelset_Spin[3:4], GF_Wheelset_Yaw[3:4]
        ])

        # 如果没有传入下部结构受力，则短路返回纯车辆受力
        if fastener_forces is None or subrail_forces is None:
            return GF_VEHICLE

        # ====================================================================
        # 4. 轨道钢轨广义力 (连续梁模态力)
        # ====================================================================
        FV1_L, FV2_L, FV_L, FL_L = fastener_forces['FV1_L'], fastener_forces['FV2_L'], fastener_forces['FV_L'], fastener_forces['FL_L']
        FV1_R, FV2_R, FV_R, FL_R = fastener_forces['FV1_R'], fastener_forces['FV2_R'], fastener_forces['FV_R'], fastener_forces['FL_R']

        Q_L, P_L = -TFy_L, TFz_L
        Q_R, P_R = -TFy_R, TFz_R
        Q_L2, P_L2 = -TFy_L2, TFz_L2
        Q_R2, P_R2 = -TFy_R2, TFz_R2

        # 扣件扭转力矩
        Ms_L = self.rp.bb * (FV2_L - FV1_L) - self.rp.aa * FL_L
        Ms_R = self.rp.bb * (FV2_R - FV1_R) - self.rp.aa * FL_R

        # 轮轨偏载扭转力矩
        Mw_L = hrL * Q_L - eL * P_L + hrL2 * Q_L2 - eL2 * P_L2
        Mw_R = hrR * Q_R - eR * P_R + hrR2 * Q_R2 - eR2 * P_R2

        # 形函数矩阵 (转置相乘 @ 实现了模态投影积分)
        Krz_W, Kry_W, Kro_W = rail_states['Krz_W'], rail_states['Kry_W'], rail_states['Kro_W']
        Krz_F, Kry_F, Kro_F = rail_modal_sys.Krz_F, rail_modal_sys.Kry_F, rail_modal_sys.Kro_F

        # 左侧钢轨模态方程 (矩阵维度映射: NV = NV - NV*Nsub @ Nsub + NV*4 @ 4)
        GF_RAIL_LV = -self.EIY * state.XRail_Z_L - Krz_F.T @ FV_L + Krz_W.T @ P_L
        GF_RAIL_LL = -self.EIZ * state.XRail_Y_L - Kry_F.T @ FL_L + Kry_W.T @ Q_L
        GF_RAIL_LT = -self.GIO * state.XRail_T_L - Kro_F.T @ Ms_L + Kro_W.T @ Mw_L

        # 右侧钢轨模态方程
        GF_RAIL_RV = -self.EIY * state.XRail_Z_R - Krz_F.T @ FV_R + Krz_W.T @ P_R
        GF_RAIL_RL = -self.EIZ * state.XRail_Y_R - Kry_F.T @ FL_R + Kry_W.T @ Q_R
        GF_RAIL_RT = -self.GIO * state.XRail_T_R - Kro_F.T @ Ms_R + Kro_W.T @ Mw_R

        # ====================================================================
        # 5. 轨枕广义力 (Sleeper)
        # ====================================================================
        FLsV, FLsL = subrail_forces['FLsV'], subrail_forces['FLsL']
        FRsV, FRsL = subrail_forces['FRsV'], subrail_forces['FRsL']

        MLr = -self.rp.bb * (FV1_L - FV2_L)
        MRr = -self.rp.bb * (FV1_R - FV2_R)
        
        GF_Sleeper_Z = FV_L + FV_R - FLsV - FRsV
        GF_Sleeper_Y = FL_L + FL_R - FLsL - FRsL
        GF_Sleeper_Roll = MLr + MRr + self.sp.d * (FV_R - FRsV) - self.sp.d * (FV_L - FLsV)

        # ====================================================================
        # 6. 道床广义力 (Ballast)
        # ====================================================================
        FLb1, FLb2, FLbR, FLbf = subrail_forces['FLb1'], subrail_forces['FLb2'], subrail_forces['FLbR'], subrail_forces['FLbf']
        FRb1, FRb2, FRbL, FRbf = subrail_forces['FRb1'], subrail_forces['FRb2'], subrail_forces['FRbL'], subrail_forces['FRbf']

        GF_Ballast_L = FLsV - FLbf - FLb1 - FLb2 - FLbR
        GF_Ballast_R = FRsV - FRbf - FRb1 - FRb2 - FRbL

        # ====================================================================
        # 7. 全系统大总装 (数千个自由度拼接为 1D 数组)
        # ====================================================================
        GF_SYSTEM = np.concatenate([
            GF_VEHICLE,
            GF_RAIL_LV, GF_RAIL_LL, GF_RAIL_LT,
            GF_RAIL_RV, GF_RAIL_RL, GF_RAIL_RT,
            GF_Sleeper_Z, GF_Sleeper_Y, GF_Sleeper_Roll,
            GF_Ballast_L, GF_Ballast_R
        ])
        
        return GF_SYSTEM