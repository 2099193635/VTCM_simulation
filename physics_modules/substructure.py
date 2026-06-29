import numpy as np


class SubstructureDynamics:
    def __init__(self, fastener_params, rail_params, subrail_params):
        self.fp = fastener_params
        self.rp = rail_params
        self.sp = subrail_params

    def compute_fastener_forces(self, rs, ss, defects=None):
        """Compute rail-fastener-sleeper spring/damper forces."""
        bb, aa, d = self.rp.bb, self.rp.aa, self.sp.d
        Kkjv, Ckjv, Kkjh, Ckjh = self.fp.Kkjv, self.fp.Ckjv, self.fp.Kkjh, self.fp.Ckjh

        Zs, Ys, Rolls = ss['XSleeper_Z'], ss['XSleeper_Y'], ss['XSleeper_Roll']
        VZs, VYs, VRolls = ss['VSleeper_Z'], ss['VSleeper_Y'], ss['VSleeper_Roll']

        if defects is None:
            defects = {}
        one = np.ones_like(Zs, dtype=float)
        Kkjv_L = Kkjv * np.asarray(defects.get('fastener_kv_factor_L', one), dtype=float)
        Ckjv_L = Ckjv * np.asarray(defects.get('fastener_cv_factor_L', one), dtype=float)
        Kkjh_L = Kkjh * np.asarray(defects.get('fastener_kh_factor_L', one), dtype=float)
        Ckjh_L = Ckjh * np.asarray(defects.get('fastener_ch_factor_L', one), dtype=float)
        Kkjv_R = Kkjv * np.asarray(defects.get('fastener_kv_factor_R', one), dtype=float)
        Ckjv_R = Ckjv * np.asarray(defects.get('fastener_cv_factor_R', one), dtype=float)
        Kkjh_R = Kkjh * np.asarray(defects.get('fastener_kh_factor_R', one), dtype=float)
        Ckjh_R = Ckjh * np.asarray(defects.get('fastener_ch_factor_R', one), dtype=float)

        FV1_L = 0.5 * Kkjv_L * (rs['RailF_Zdis_L'] - bb * rs['RailF_Tdis_L'] - Zs + (d + bb) * Rolls) + \
                0.5 * Ckjv_L * (rs['RailF_Zvel_L'] - bb * rs['RailF_Tvel_L'] - VZs + (d + bb) * VRolls)

        FV2_L = 0.5 * Kkjv_L * (rs['RailF_Zdis_L'] + bb * rs['RailF_Tdis_L'] - Zs + (d - bb) * Rolls) + \
                0.5 * Ckjv_L * (rs['RailF_Zvel_L'] + bb * rs['RailF_Tvel_L'] - VZs + (d - bb) * VRolls)

        FV_L = FV1_L + FV2_L

        FL_L = Kkjh_L * (rs['RailF_Ldis_L'] - Ys - aa * rs['RailF_Tdis_L']) + \
               Ckjh_L * (rs['RailF_Lvel_L'] - VYs - aa * rs['RailF_Tvel_L'])

        FV1_R = 0.5 * Kkjv_R * (rs['RailF_Zdis_R'] - bb * rs['RailF_Tdis_R'] - Zs - (d - bb) * Rolls) + \
                0.5 * Ckjv_R * (rs['RailF_Zvel_R'] - bb * rs['RailF_Tvel_R'] - VZs - (d - bb) * VRolls)

        FV2_R = 0.5 * Kkjv_R * (rs['RailF_Zdis_R'] + bb * rs['RailF_Tdis_R'] - Zs - (d + bb) * Rolls) + \
                0.5 * Ckjv_R * (rs['RailF_Zvel_R'] + bb * rs['RailF_Tvel_R'] - VZs - (d + bb) * VRolls)

        FV_R = FV1_R + FV2_R

        FL_R = Kkjh_R * (rs['RailF_Ldis_R'] - Ys - aa * rs['RailF_Tdis_R']) + \
               Ckjh_R * (rs['RailF_Lvel_R'] - VYs - aa * rs['RailF_Tvel_R'])

        return {'FV1_L': FV1_L, 'FV2_L': FV2_L, 'FV_L': FV_L, 'FL_L': FL_L,
                'FV1_R': FV1_R, 'FV2_R': FV2_R, 'FV_R': FV_R, 'FL_R': FL_R}

    def compute_subrail_forces(self, ss, bs, defects=None):
        """Compute sleeper-ballast-subgrade forces with optional void gaps."""
        d, Kbv, Cbv, Kbh, Cbh = self.sp.d, self.sp.Kbv, self.sp.Cbv, self.sp.Kbh, self.sp.Cbh
        Kw, Cw, Kfv, Cfv = self.sp.Kw, self.sp.Cw, self.sp.Kfv, self.sp.Cfv

        Zs, Ys, Rolls = ss['XSleeper_Z'], ss['XSleeper_Y'], ss['XSleeper_Roll']
        VZs, VYs, VRolls = ss['VSleeper_Z'], ss['VSleeper_Y'], ss['VSleeper_Roll']

        XSub_L, VSub_L = bs['XSubgrade_L'], bs['VSubgrade_L']
        XSub_R, VSub_R = bs['XSubgrade_R'], bs['VSubgrade_R']

        rel_L = Zs - XSub_L - d * Rolls
        vrel_L = VZs - VSub_L - d * VRolls
        rel_R = Zs - XSub_R + d * Rolls
        vrel_R = VZs - VSub_R + d * VRolls

        if defects is None:
            defects = {}
        one = np.ones_like(Zs, dtype=float)
        Kbv_L = Kbv * np.asarray(defects.get('ballast_kv_factor_L', one), dtype=float)
        Cbv_L = Cbv * np.asarray(defects.get('ballast_cv_factor_L', one), dtype=float)
        Kbv_R = Kbv * np.asarray(defects.get('ballast_kv_factor_R', one), dtype=float)
        Cbv_R = Cbv * np.asarray(defects.get('ballast_cv_factor_R', one), dtype=float)

        FLsV = Kbv_L * rel_L + Cbv_L * vrel_L
        FLsL = Kbh * Ys + Cbh * VYs

        FRsV = Kbv_R * rel_R + Cbv_R * vrel_R
        FRsL = Kbh * Ys + Cbh * VYs

        void_L = np.asarray(defects.get('void_active_L', np.zeros_like(Zs, dtype=bool)), dtype=bool)
        void_R = np.asarray(defects.get('void_active_R', np.zeros_like(Zs, dtype=bool)), dtype=bool)
        gap_L = np.asarray(defects.get('void_gap_L', np.zeros_like(Zs, dtype=float)), dtype=float)
        gap_R = np.asarray(defects.get('void_gap_R', np.zeros_like(Zs, dtype=float)), dtype=float)
        contact_L = np.ones_like(Zs, dtype=bool)
        contact_R = np.ones_like(Zs, dtype=bool)

        if np.any(void_L):
            compression_L = rel_L - gap_L
            force_L = Kbv_L * compression_L + Cbv_L * vrel_L
            contact_L = (~void_L) | (compression_L >= 0.0)
            FLsV = np.where(void_L, np.where(contact_L, np.maximum(force_L, 0.0), 0.0), FLsV)
        if np.any(void_R):
            compression_R = rel_R - gap_R
            force_R = Kbv_R * compression_R + Cbv_R * vrel_R
            contact_R = (~void_R) | (compression_R >= 0.0)
            FRsV = np.where(void_R, np.where(contact_R, np.maximum(force_R, 0.0), 0.0), FRsV)

        XSubL_1 = np.append(XSub_L[1:], 0)
        XSubL_2 = np.append(0, XSub_L[:-1])
        VSubL_1 = np.append(VSub_L[1:], 0)
        VSubL_2 = np.append(0, VSub_L[:-1])

        XSubR_1 = np.append(XSub_R[1:], 0)
        XSubR_2 = np.append(0, XSub_R[:-1])
        VSubR_1 = np.append(VSub_R[1:], 0)
        VSubR_2 = np.append(0, VSub_R[:-1])

        FLb1 = Kw * (XSub_L - XSubL_1) + Cw * (VSub_L - VSubL_1)
        FLb2 = Kw * (XSub_L - XSubL_2) + Cw * (VSub_L - VSubL_2)
        FLbR = Kw * (XSub_L - XSub_R) + Cw * (VSub_L - VSub_R)
        FLbf = Kfv * XSub_L + Cfv * VSub_L

        FRb1 = Kw * (XSub_R - XSubR_1) + Cw * (VSub_R - VSubR_1)
        FRb2 = Kw * (XSub_R - XSubR_2) + Cw * (VSub_R - VSubR_2)
        FRbL = -FLbR
        FRbf = Kfv * XSub_R + Cfv * VSub_R

        return {'FLsV': FLsV, 'FLsL': FLsL, 'FRsV': FRsV, 'FRsL': FRsL,
                'FLb1': FLb1, 'FLb2': FLb2, 'FLbR': FLbR, 'FLbf': FLbf,
                'FRb1': FRb1, 'FRb2': FRb2, 'FRbL': FRbL, 'FRbf': FRbf,
                'Void_contact_L': contact_L, 'Void_contact_R': contact_R}
