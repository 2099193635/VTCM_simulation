'''
Author: Niscienc 60505912+2099193635@users.noreply.github.com
Date: 2026-03-14 16:55:55
LastEditors: Niscienc 60505912+2099193635@users.noreply.github.com
LastEditTime: 2026-03-14 16:56:05
FilePath: \VTCM_PYTHON\physics_modules\rail_modal.py
Description: 

Copyright (c) 2026 by ${git_name_email}, All Rights Reserved. 
'''
import numpy as np

class RailModalDynamics:
    def __init__(self, rail_params, integration_params, mode_params):
        """初始化钢轨模态系统，预计算静态扣件节点形函数"""
        self.rp = rail_params
        self.ip = integration_params
        self.mp = mode_params
        
        # 扣件的纵向坐标 (数组长度为 Nsub)
        self.X_F = self.ip.Cord_fastener 
        
        # 生成模态阶数序列: 1, 2, ..., N
        nv_modes = np.arange(1, self.mp.NV + 1)
        nl_modes = np.arange(1, self.mp.NL + 1)
        nt_modes = np.arange(1, self.mp.NT + 1)
        
        # 🌟 自动补齐原作者遗漏的扣件处形函数 (Krz_F, Kry_F, Kro_F)
        # 形状: (N_fastener, NV)
        self.Krz_F = np.sqrt(2 / (self.rp.mr * self.ip.Ljs)) * np.sin(np.outer(self.X_F, nv_modes) * np.pi / self.ip.Ljs)
        self.Kry_F = np.sqrt(2 / (self.rp.mr * self.ip.Ljs)) * np.sin(np.outer(self.X_F, nl_modes) * np.pi / self.ip.Ljs)
        self.Kro_F = np.sqrt(2 / (self.rp.pr * self.rp.I0 * self.ip.Ljs)) * np.sin(np.outer(self.X_F, nt_modes) * np.pi / self.ip.Ljs)

    def get_wheelset_shape_functions(self, Xw):
        """
        计算当前步长下，4个动态轮对位置处的形函数矩阵
        Xw: [x1, x2, x3, x4] 绝对坐标
        返回维度为 (4, NV) 的动态形函数
        """
        nv_modes = np.arange(1, self.mp.NV + 1)
        nl_modes = np.arange(1, self.mp.NL + 1)
        nt_modes = np.arange(1, self.mp.NT + 1)
        
        Krz_W = np.sqrt(2 / (self.rp.mr * self.ip.Ljs)) * np.sin(np.outer(Xw, nv_modes) * np.pi / self.ip.Ljs)
        Kry_W = np.sqrt(2 / (self.rp.mr * self.ip.Ljs)) * np.sin(np.outer(Xw, nl_modes) * np.pi / self.ip.Ljs)
        Kro_W = np.sqrt(2 / (self.rp.pr * self.rp.I0 * self.ip.Ljs)) * np.sin(np.outer(Xw, nt_modes) * np.pi / self.ip.Ljs)
        
        return Krz_W, Kry_W, Kro_W

    def extract_physical_states(self, q_state, Xw):
        """
        输入钢轨模态坐标q，输出轮对接触点和扣件处的真实物理位移/速度
        q_state 包含从总状态向量 X, V 中切片出来的 XRail_Z_L 等数组
        """
        Krz_W, Kry_W, Kro_W = self.get_wheelset_shape_functions(Xw)
        
        # 极速矩阵乘法提取物理状态 (@ 运算符相当于原 MATLAB 中的 *)
        states = {}
        
        # 1. 轮对位置处钢轨状态 (长度为 4)
        states['RailW_Zdis_L'] = Krz_W @ q_state['XRail_Z_L']
        states['RailW_Ldis_L'] = Kry_W @ q_state['XRail_Y_L']
        states['RailW_Tdis_L'] = Kro_W @ q_state['XRail_T_L']
        states['RailW_Zvel_L'] = Krz_W @ q_state['VRail_Z_L']
        states['RailW_Lvel_L'] = Kry_W @ q_state['VRail_Y_L']
        states['RailW_Tvel_L'] = Kro_W @ q_state['VRail_T_L']

        states['RailW_Zdis_R'] = Krz_W @ q_state['XRail_Z_R']
        states['RailW_Ldis_R'] = Kry_W @ q_state['XRail_Y_R']
        states['RailW_Tdis_R'] = Kro_W @ q_state['XRail_T_R']
        states['RailW_Zvel_R'] = Krz_W @ q_state['VRail_Z_R']
        states['RailW_Lvel_R'] = Kry_W @ q_state['VRail_Y_R']
        states['RailW_Tvel_R'] = Kro_W @ q_state['VRail_T_R']

        # 2. 扣件位置处钢轨状态 (长度为 Nsub)
        states['RailF_Zdis_L'] = self.Krz_F @ q_state['XRail_Z_L']
        states['RailF_Ldis_L'] = self.Kry_F @ q_state['XRail_Y_L']
        states['RailF_Tdis_L'] = self.Kro_F @ q_state['XRail_T_L']
        states['RailF_Zvel_L'] = self.Krz_F @ q_state['VRail_Z_L']
        states['RailF_Lvel_L'] = self.Kry_F @ q_state['VRail_Y_L']
        states['RailF_Tvel_L'] = self.Kro_F @ q_state['VRail_T_L']

        states['RailF_Zdis_R'] = self.Krz_F @ q_state['XRail_Z_R']
        states['RailF_Ldis_R'] = self.Kry_F @ q_state['XRail_Y_R']
        states['RailF_Tdis_R'] = self.Kro_F @ q_state['XRail_T_R']
        states['RailF_Zvel_R'] = self.Krz_F @ q_state['VRail_Z_R']
        states['RailF_Lvel_R'] = self.Kry_F @ q_state['VRail_Y_R']
        states['RailF_Tvel_R'] = self.Kro_F @ q_state['VRail_T_R']
        
        # 将生成的动态形函数也返回，组装模态刚度力时需要用到
        states['Krz_W'] = Krz_W
        states['Kry_W'] = Kry_W
        states['Kro_W'] = Kro_W
        
        return states