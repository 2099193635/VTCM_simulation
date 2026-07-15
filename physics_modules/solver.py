import numpy as np
import time
from tqdm import tqdm


class SystemDynamics:
    """系统加速度求解器"""
    
    def __init__(self, veh_params, veh_int, para_subrail, control_mode, para_extra_force):
        """
        初始化时预计算车辆系统质量向量 (恒定不变，只需算一次)
        复刻 MATLAB Subroutine_Acceleration_Output.m
        """
        p = veh_params
        sub = para_subrail
        mode = control_mode
        ep = para_extra_force
        inti = veh_int

        # =======================车辆系统=======================
        # 车体质量向量 (5)
        m_car = [p.Mc, p.Mc, p.Jcx, p.Jcy, p.Jcz]
        # 构架质量向量 (2 x 5 = 10)
        m_bogie = [p.Mt, p.Mt, p.Jtx, p.Jty, p.Jtz] * 2
        # 轮对质量向量 (4 x 5 = 20)
        m_wheelset = [p.Mw, p.Mw, p.Jwx, p.Jwy, p.Jwz] * 4
        # # 轴箱质量向量 (左4 + 右4 = 8)
        # m_axlebox = [ep.Jaxlebox] * 8
        # 组装纯车辆系统的 35 自由度质量向量
        m_vehicle = m_car + m_bogie + m_wheelset
        # =======================钢轨系统=======================
        # 单侧轨道的总模态数量
        rail_dofs = mode.NV + mode.NL + mode.NT
        m_left_rail = np.ones(rail_dofs)
        m_right_rail = np.ones(rail_dofs)
        # =======================轨下结构=======================
        # 轨道节点数量
        num_nodes = inti.Nsub + 1
        # 轨枕 (3 段拼接：沉浮Z, 横移Y, 侧滚Roll)
        m_sleeper = np.concatenate([
            np.full(num_nodes, sub.Ms),
            np.full(num_nodes, sub.Ms),
            np.full(num_nodes, sub.Js),
        ])
        # 道床块 (2 段拼接：左右侧道床)
        m_ballast = np.full(2 * num_nodes, sub.Mb)
        # 组装整体系统自由度
        self.Mass_FULL = np.concatenate([
            m_vehicle,
            m_left_rail,
            m_right_rail,
            m_sleeper,
            m_ballast
        ])
        self.Mass_VEHICLE = np.array(m_vehicle)
        

    def compute_acceleration(self, GF_SYSTEM: np.ndarray) -> np.ndarray:
        
        # 核心：F = M * A  =>  A = F / M (Element-wise division)
        A_SYSTEM = GF_SYSTEM / self.Mass_FULL
        
        return A_SYSTEM
    
class DynamicSolver:
    """
    车辆-轨道刚柔耦合动力学核心求解器
    采用 翟方法 (Zhai Method / 新型显式积分法) 进行时域步进求解
    """
    @staticmethod
    def _as_bool(value):
        """Robust bool parser for CLI/config values ('On'/'Off', 'true'/'false', etc.)."""
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            v = value.strip().lower()
            if v in ('on', 'true', '1', 'yes', 'y'):
                return True
            if v in ('off', 'false', '0', 'no', 'n', ''):
                return False
        return bool(value)

    def __init__(self, topology, integration_params, switch_lock_veh_non_z, switch_lock_axlebox,
                 switch_lock_substructure, sim_switches=None):
        """
        初始化动态求解器
        :param topology: 系统拓扑对象
        :param integration_params: 积分参数对象
        :param switch_lock_veh_non_z: 车体非垂向自由度锁定开关
        :param switch_lock_axlebox: 轴箱自由度锁定开关
        :param switch_lock_substructure: 轨下结构自由度锁定开关
        :param sim_switches: 仿真功能开关对象 (ExtraforceElementSwitch)
        """
        self.topo = topology
        self.params = integration_params
        self.sim_switches = sim_switches        # 仿真功能开关（含曲线力开关）

        # 自由度锁定
        self.switch_lock_veh_non_z = self._as_bool(switch_lock_veh_non_z)
        self.switch_lock_axlebox = self._as_bool(switch_lock_axlebox)
        self.switch_lock_substructure = self._as_bool(switch_lock_substructure)

        # 锁定掩码
        self._build_freedom_locker()
    
    def _build_freedom_locker(self):
        num_dof = self.topo.Fnum_Total
        lock_mask = np.zeros(num_dof, dtype=bool) # 初始全为 False

        # --- LOCK0: 锁定车体部分自由度 (仅保留 Z 向沉浮运动) ---
        if self.switch_lock_veh_non_z:
            z_dofs = [1, 6, 11, 16, 21, 26, 31]
            all_veh_dofs = list(range(35))
            lock_dofs_veh = [d for d in all_veh_dofs if d not in z_dofs]
            lock_mask[lock_dofs_veh] = True
        # --- LOCK1: 锁定车体轴箱 ---
        if self.switch_lock_axlebox:
            raise ValueError(
                "switch_lock_axlebox=On is invalid: the current 35-DOF vehicle "
                "topology contains no axlebox DOFs. Indices 35:43 belong to rail modes."
            )
        # --- LOCK2: 锁定轨下结构自由度 ---
        if self.switch_lock_substructure:
            start = self.topo.idx_Sleeper[0]
            end = self.topo.idx_Subgrade_R[1]
            lock_mask[start:end] = True
        
        self.lock_mask = lock_mask
        
    def _prepare_moving_window_migration(self, engines, track_geometry):
        """Build cached operators for grid-aligned moving-window state migration."""
        if not (track_geometry and track_geometry.get('use_local_window_dynamics', False)):
            return None
        sw = track_geometry.get('structure_window', {})
        if not sw or not sw.get('state_migration', False):
            return None
        rail_modal = engines.get('rail_modal') if engines else None
        if rail_modal is None:
            return None

        n_nodes = self.params.Nsub + 1
        cache = {
            'shift_nodes': np.asarray(sw.get('window_shift_nodes', np.zeros(self.topo.Nt)), dtype=np.int64),
            'n_nodes': n_nodes,
            'rail_ops': {
                'z': (rail_modal.Krz_F, np.linalg.pinv(rail_modal.Krz_F)),
                'y': (rail_modal.Kry_F, np.linalg.pinv(rail_modal.Kry_F)),
                't': (rail_modal.Kro_F, np.linalg.pinv(rail_modal.Kro_F)),
            },
        }
        print(" -> [移动窗口迁移] 已启用：窗口按扣件节点推进，轨下节点平移，钢轨模态按节点位移/速度重投影。")
        return cache

    @staticmethod
    def _shift_node_values(values, shift_nodes):
        """Shift node values toward the rear of the local window; new front nodes are zero."""
        n = values.size
        out = np.zeros_like(values)
        if shift_nodes <= 0:
            out[:] = values
        elif shift_nodes < n:
            out[:n - shift_nodes] = values[shift_nodes:]
        return out

    def _migrate_rail_modal_vector(self, vec, cache, shift_nodes):
        """Project old modal coordinates onto the shifted local-window modal basis."""
        nv, nl, nt = self.topo.NV, self.topo.NL, self.topo.NT
        z_rel = slice(0, nv)
        y_rel = slice(nv, nv + nl)
        t_rel = slice(nv + nl, nv + nl + nt)

        def migrate_one(rail_slice):
            rail_vec = vec[rail_slice]
            for comp, rel_slice in (('z', z_rel), ('y', y_rel), ('t', t_rel)):
                K, K_pinv = cache['rail_ops'][comp]
                q_old = rail_vec[rel_slice]
                nodal_old = K @ q_old
                nodal_new = self._shift_node_values(nodal_old, shift_nodes)
                rail_vec[rel_slice] = K_pinv @ nodal_new

        migrate_one(slice(*self.topo.idx_Rail_L))
        migrate_one(slice(*self.topo.idx_Rail_R))

    def _migrate_substructure_vector(self, vec, cache, shift_nodes):
        """Shift sleeper and subgrade nodal states in the local moving window."""
        n = cache['n_nodes']
        sleeper = vec[self.topo.idx_Sleeper[0]:self.topo.idx_Sleeper[1]]
        for j in range(3):
            part = sleeper[j * n:(j + 1) * n]
            part[:] = self._shift_node_values(part, shift_nodes)

        sub_l = vec[self.topo.idx_Subgrade_L[0]:self.topo.idx_Subgrade_L[1]]
        sub_r = vec[self.topo.idx_Subgrade_R[0]:self.topo.idx_Subgrade_R[1]]
        sub_l[:] = self._shift_node_values(sub_l, shift_nodes)
        sub_r[:] = self._shift_node_values(sub_r, shift_nodes)

    def _migrate_moving_window_rows(self, X, V, A, rows, cache, shift_nodes):
        """Migrate the recent integration history rows needed by the explicit scheme."""
        rows = sorted({int(r) for r in rows if 0 <= int(r) < X.shape[0]})
        for row in rows:
            for arr in (X, V, A):
                self._migrate_rail_modal_vector(arr[row, :], cache, shift_nodes)
                self._migrate_substructure_vector(arr[row, :], cache, shift_nodes)

    @staticmethod
    def _suspension_curve_inputs(track_geometry, step_i):
        """Return [front bogie, rear bogie, carbody] radius and d(1/R)/dt."""
        if track_geometry is None:
            return np.full(3, np.inf), np.zeros(3)
        curvature = np.asarray(track_geometry['K'][step_i], dtype=float)
        curvature_rate = np.asarray(track_geometry['dK'][step_i], dtype=float)
        if curvature.size < 7 or curvature_rate.size < 7:
            raise ValueError("track_geometry K/dK must contain 7 vehicle locations")
        selected = curvature[[4, 5, 6]]
        radii = np.full(3, np.inf, dtype=float)
        active = np.abs(selected) > 1e-12
        radii[active] = 1.0 / selected[active]
        return radii, curvature_rate[[4, 5, 6]]
    def solve(self, track_excitation, geom_info, engines, track_geometry=None, sim_switches=None, save_dof_mode='full', save_stride=1, save_spy_level='full'):
        """
        执行动力学主循环
        :param track_excitation: 激扰矩阵元组 (bz_L, by_L, dbz_L, ...)
        :param geom_info: 轮轨接触几何前处理数据 (ContactGeometryInfo)
        :param engines: 打包好的物理引擎字典 (suspension, contact, assembler, dynamics)
        :param track_geometry: 预计算的线型参数矩阵字典 {'K','H','G','dK','dH','S'}，每项 (Nt,4)
        :param sim_switches: 仿真功能开关对象（优先使用构造时传入的值）
        """
        # 优先使用外部传入值，否则使用构造时注册的值
        if sim_switches is None:
            sim_switches = self.sim_switches
        Nt = self.topo.Nt
        dt = self.params.Tstep
        save_stride = max(1, int(save_stride))
        save_steps = np.arange(0, Nt, save_stride, dtype=np.int64)
        Nt_out = int(save_steps.size)
        alpha = self.params.alpha
        beta = self.params.beta
        vc = self.params.Vc
        omg = self.params.omega  # 名义滚动角速度 (rad/s) = Vc/R
        two_point_contact = bool(sim_switches is not None and sim_switches.is_active('Switch_2PointContact'))
        extra_force_enabled = bool(sim_switches is not None and sim_switches.is_active('Switch_ExtraForceElement'))
        
        # 解包物理引擎
        suspension_sys = engines['suspension']
        wr_interaction = engines['contact']
        gf_assembler = engines['assembler']
        sys_dynamics = engines['dynamics']
        
        # 1. 申请内存。vehicle 模式只保留完整系统最近 3 步，避免长时长保存非车体自由度。
        save_mode = str(save_dof_mode).lower().strip()
        vehicle_history_only = save_mode == 'vehicle'
        car_start, car_end = self.topo.idx_Car
        car_dofs = car_end - car_start
        if vehicle_history_only:
            X = np.zeros((3, self.topo.Fnum_Total))
            V = np.zeros((3, self.topo.Fnum_Total))
            A = np.zeros((3, self.topo.Fnum_Total))
            X_out = np.zeros((Nt_out, car_dofs), dtype=np.float32)
            V_out = np.zeros((Nt_out, car_dofs), dtype=np.float32)
            A_out = np.zeros((Nt_out, car_dofs), dtype=np.float32)
            spy_dict = self.topo.allocate_spy_memory(switch_2point_contact='On' if two_point_contact else 'Off', Nt_out=Nt_out, spy_level=save_spy_level)
            print(" -> [内存模式] save_dof_mode=vehicle：完整系统仅保留3步环形缓存，输出仅记录车辆35自由度。")
        else:
            X, V, A, PadComp_L1, PadComp_L2, PadComp_R1, PadComp_R2, spy_dict = self.topo.allocate_memory(
                switch_2point_contact='On' if two_point_contact else 'Off', Nt_out=Nt_out, spy_level=save_spy_level)
            X_out = V_out = A_out = None
        print(f" -> [保存采样] save_stride={save_stride}, 输出步数: {Nt_out}/{Nt}, spy_level={save_spy_level}")

        # ==========================================
        # 静态落车初始条件：轮对已以名义转速滚动
        # ==========================================
        # 轮对 Spin 自由度索引 (0-based, 车辆35个DOF内): 18, 23, 28, 33
        # 对应 V 矩阵全局索引 = topo.idx_Car[0] + [18, 23, 28, 33]
        _car_start = car_start
        _spin_global = _car_start + np.array([18, 23, 28, 33])
        # 未被锁定的 Spin DOF 才赋初值
        _spin_unlocked = _spin_global[~self.lock_mask[_spin_global]]
        V[0, _spin_unlocked] = omg
        print(f" -> [初始条件] 轮对名义自旋速度已设为 omg = {omg:.4f} rad/s (Vc={vc:.2f} m/s, R={self.params.omega and vc/omg:.4f} m)")

        # 解包激扰矩阵 (4个轮对，长度为 Nt)
        bz_L, by_L, dbz_L, dby_L, bz_R, by_R, dbz_R, dby_R, defect_a, defect_L = track_excitation

        print(f" -> [求解器启动] 开始执行时域积分 (翟方法)，总步数: {Nt}，步长: {dt}s")
        migration_cache = self._prepare_moving_window_migration(engines, track_geometry)
        structure_defects = None
        if track_geometry is not None:
            structure_defects = track_geometry.get('structure_defects')
        if structure_defects is not None and structure_defects.is_active():
            summary = structure_defects.summary()
            print(f" -> [结构缺陷] 已启用: {summary.get('record_count', 0)} 条扣件/空吊缺陷记录")
        current_window_shift = 0
        start_time = time.time()

        # ==========================================
        # 核心积分大循环 (对应 MATLAB for ii=1:Nt)
        # ==========================================
        for i in tqdm(range(Nt), desc="动力学积分进度", unit="步", ncols=100):
            row = (i % 3) if vehicle_history_only else i

            if migration_cache is not None:
                target_shift = int(migration_cache['shift_nodes'][min(i, migration_cache['shift_nodes'].size - 1)])
                shift_delta = target_shift - current_window_shift
                if shift_delta > 0:
                    if i == 0:
                        history_rows = []
                    elif i == 1:
                        history_rows = [(i - 1) % 3 if vehicle_history_only else i - 1]
                    else:
                        history_rows = [
                            (i - 1) % 3 if vehicle_history_only else i - 1,
                            (i - 2) % 3 if vehicle_history_only else i - 2,
                        ]
                    self._migrate_moving_window_rows(X, V, A, history_rows, migration_cache, shift_delta)
                    current_window_shift = target_shift
                elif shift_delta < 0:
                    raise RuntimeError("移动窗口节点编号不应倒退，请检查 window_shift_nodes。")

            # ==========================================
            # STEP 1: 积分预测器 (Predictor - 翟方法)
            # ==========================================
            if i == 0:
                # 初始静止状态 (在此可加入落车静承载初始位移 X0)
                pass
            elif i == 1:
                prev = (i - 1) % 3 if vehicle_history_only else i - 1
                # 第二步：无 i-2 的历史加速度数据
                X[row, :] = X[prev, :] + V[prev, :] * dt + (0.5 + alpha) * A[prev, :] * (dt**2)
                V[row, :] = V[prev, :] + (1 + beta) * A[prev, :] * dt
            else:
                prev = (i - 1) % 3 if vehicle_history_only else i - 1
                prev2 = (i - 2) % 3 if vehicle_history_only else i - 2
                # 第三步及以后：完整预测公式
                X[row, :] = X[prev, :] + V[prev, :] * dt + (0.5 + alpha) * A[prev, :] * (dt**2) - alpha * A[prev2, :] * (dt**2)
                V[row, :] = V[prev, :] + (1 + beta) * A[prev, :] * dt - beta * A[prev2, :] * dt

            X[row, self.lock_mask] = 0.0
            V[row, self.lock_mask] = 0.0

            # ==========================================
            # STEP 2: 状态切片提取 (映射给各子结构)
            # ==========================================
            # 将当前步的一维长向量 X[row] 转化为带有各种属性的对象 state
            state = self.topo.extract_state(X[row, :], V[row, :], vc)
            
            # 提取当前步、4个轮对的轨道不平顺 (Irregularity)
            IrreZ_L, VIrreZ_L = bz_L[:, i], dbz_L[:, i]
            IrreZ_R, VIrreZ_R = bz_R[:, i], dbz_R[:, i]
            IrreL_L, VIrreL_L = by_L[:, i], dby_L[:, i]
            IrreL_R, VIrreL_R = by_R[:, i], dby_R[:, i]

            # ==========================================
            # STEP 3: 悬挂力计算 (Force_Pre_Sec)
            # ==========================================
            if (track_geometry is not None
                    and sim_switches is not None
                    and sim_switches.is_active('Switch_CurveTrack')):
                curve_radii, curvature_rate = self._suspension_curve_inputs(track_geometry, i)
            else:
                curve_radii, curvature_rate = np.full(3, np.inf), np.zeros(3)
            susp_forces = suspension_sys.compute_forces(
                state,
                R_curve=curve_radii,
                d_invR_dt=curvature_rate,
                include_extra=extra_force_enabled,
                axlebox_state=None,
            )

            # ==========================================
            # STEP 4: 提取钢轨物理状态
            # ==========================================
            t_current = i * dt
            X0t = self.params.X0 + self.params.Vc * t_current
            Lc = self.params.Lc
            Lt = self.params.Lt
            Xw = np.array([
                X0t + 2 * (Lc + Lt),  # 1位轮对 (前构架前轮)
                X0t + 2 * Lc,         # 2位轮对 (前构架后轮)
                X0t + 2 * Lt,         # 3位轮对 (后构架前轮)
                X0t                   # 4位轮对 (后构架后轮)
            ])
            if (track_geometry is not None
                    and track_geometry.get('use_local_window_dynamics', False)
                    and 'structure_window' in track_geometry):
                Xw = track_geometry['structure_window']['local_s_m'][i, :4]
            rail_states = engines['rail_modal'].extract_physical_states(state.__dict__, Xw=Xw)


            # ==========================================
            # STEP 5: 轮轨接触力计算 (Contact_Wheel_Rail_TwoPoints)
            # ==========================================
            # 初始化当前步 4 个轮对的力学容器字典 (数组格式，为了送给 Assembler)
            wr_forces_keys = [
                'FNx_L', 'FNy_L', 'FNz_L', 'FNx_R', 'FNy_R', 'FNz_R', 
                'FNx_L2', 'FNy_L2', 'FNz_L2', 'FNx_R2', 'FNy_R2', 'FNz_R2',
                'MLy', 'MLz', 'MRy', 'MRz', 'rL', 'rR', 'rL2', 'rR2', 'a0', 'a02',
                'CreepForce_L', 'CreepForce_R',
                'hrL', 'eL', 'hrR', 'eR', 'hrL2', 'eL2', 'hrR2', 'eR2'  # <--- 加上这行
            ]
            wr_forces = {k: np.zeros(4) for k in wr_forces_keys}

            # 分别对 4 个轮对执行空间非线性寻优与接触力学计算
            for nw in range(4):
                # 调用我们在 wheel_rail_contact.py 中写好的终极方法
                f_nw = wr_interaction.calculate_two_point_contact(
                    nw=nw, vc=vc, omg=omg,
                    Zw=state.X_ZW[nw], Yw=state.X_YW[nw], phiw=state.X_RollW[nw], psiw=state.X_YawW[nw],
                    LRKX_Y=rail_states['RailW_Ldis_L'][nw], LRKX_Z=rail_states['RailW_Zdis_L'][nw], thetaL=rail_states['RailW_Tdis_L'][nw], 
                    RRKX_Y=rail_states['RailW_Ldis_R'][nw], RRKX_Z=rail_states['RailW_Zdis_R'][nw], thetaR=rail_states['RailW_Tdis_R'][nw],
                    VXwo=vc, VYwo=state.V_YW[nw], VZwo=state.V_ZW[nw],
                    Vwphi=state.V_RollW[nw], Vwbeta=state.V_SpinW[nw], Vwpsi=state.V_YawW[nw],
                    VrkxY_L=rail_states['RailW_Lvel_L'][nw], VrkxZ_L=rail_states['RailW_Zvel_L'][nw], VrkxO_L=rail_states['RailW_Tvel_L'][nw], 
                    VrkxY_R=rail_states['RailW_Lvel_R'][nw], VrkxZ_R=rail_states['RailW_Zvel_R'][nw], VrkxO_R=rail_states['RailW_Tvel_R'][nw],
                    Irrez_L=IrreZ_L[nw], Irrey_L=IrreL_L[nw], VIrrez_L=VIrreZ_L[nw], VIrrey_L=VIrreL_L[nw],
                    Irrez_R=IrreZ_R[nw], Irrey_R=IrreL_R[nw], VIrrez_R=VIrreZ_R[nw], VIrrey_R=VIrreL_R[nw]
                )
                if not two_point_contact:
                    for key in ('FNx_L2', 'FNy_L2', 'FNz_L2', 'FNx_R2', 'FNy_R2', 'FNz_R2',
                                'rL2', 'rR2', 'hrL2', 'eL2', 'hrR2', 'eR2'):
                        f_nw[key] = 0.0
                    f_nw['a02'] = 0.0
                
                # 将第 nw 个轮对的受力组装到数组中
                for k in wr_forces_keys:
                    if k in f_nw:
                        wr_forces[k][nw] = f_nw[k]

            # ==========================================
            # STEP 6: 组装系统广义力向量 (GF_SYSTEM)
            # ==========================================
            

            step_defects = structure_defects.get_step_defects(i) if structure_defects is not None else None
            fastener_forces = engines['substructure'].compute_fastener_forces(rail_states, state.__dict__, defects=step_defects)
            subrail_forces = engines['substructure'].compute_subrail_forces(state.__dict__, state.__dict__, defects=step_defects)
            
            # 送入全系统大总装！
            GF_SYSTEM = gf_assembler.assemble_GF_SYSTEM(
                state, susp_forces, wr_forces, 
                fastener_forces=fastener_forces, subrail_forces=subrail_forces, 
                rail_modal_sys=engines['rail_modal'], rail_states=rail_states
            )
            
            # ==========================================
            # STEP 6.5: 曲线等效力叠加 (Equivalent Curve Forces)
            # ==========================================
            # 完整实现 35-DOF 曲线等效力，与 MATLAB Force_EquivalentCurveForce.m 完全对应。
            # track_geometry 列顺序 (Nt,7): [Xw1, Xw2, Xw3, Xw4, Xt1, Xt2, Xc]
            if (track_geometry is not None
                    and sim_switches is not None
                    and sim_switches.is_active('Switch_CurveTrack')
                    and 'veh_params' in engines):

                ECF_35 = self._compute_curve_force(
                    step_i=i,
                    tg=track_geometry,
                    wr_forces=wr_forces,
                    state=state,
                    vc=vc,
                    veh=engines['veh_params'],
                    g=self.params.g
                )
                CF = np.zeros(len(GF_SYSTEM))
                CF[:35] = ECF_35
                GF_SYSTEM = GF_SYSTEM + CF

            # ==========================================
            # STEP 7: 求解当前步加速度 A (A = M^-1 * GF)
            # ==========================================
            # idx_start, idx_end = self.topo.idx_Car
            A[row, :] = sys_dynamics.compute_acceleration(GF_SYSTEM)
            # 自由度锁定
            A[row, self.lock_mask] = 0.0
            # ==========================================
            # STEP 8: 监视数据与结果记录 (SPY)
            # ==========================================
            if i % save_stride == 0:
                out_i = i // save_stride
                spy_dict['Output_step_index'][out_i] = i

                # 1. 记录一系悬挂力 (将左4个和右4个拼接成长度为8的数组)
                spy_dict['Yixi_Force_x'][out_i, :] = np.concatenate([susp_forces['Fxf_L'], susp_forces['Fxf_R']])
                spy_dict['Yixi_Force_y'][out_i, :] = np.concatenate([susp_forces['Fyf_L'], susp_forces['Fyf_R']])
                spy_dict['Yixi_Force_z'][out_i, :] = np.concatenate([susp_forces['Fzf_L'], susp_forces['Fzf_R']])

                # 2. 记录二系悬挂力 (将左2个和右2个拼接成长度为4的数组)
                spy_dict['Erxi_Force_x'][out_i, :] = np.concatenate([susp_forces['Fxt_L'], susp_forces['Fxt_R']])
                spy_dict['Erxi_Force_y'][out_i, :] = np.concatenate([susp_forces['Fyt_L'], susp_forces['Fyt_R']])
                spy_dict['Erxi_Force_z'][out_i, :] = np.concatenate([susp_forces['Fzt_L'], susp_forces['Fzt_R']])

                # 3. 记录踏面接触区(点1)轮轨力 (拼接左4个和右4个，长度为8)
                spy_dict['TotalVerticalForce'][out_i, :] = np.concatenate([wr_forces['FNz_L'], wr_forces['FNz_R']])
                spy_dict['TotalLateralForce'][out_i, :] = np.concatenate([wr_forces['FNy_L'], wr_forces['FNy_R']])

                # 4. 记录轮缘接触区(点2)轮轨力 (动态判断是否开启了两点接触监视)
                if 'TotalVerticalForce_Point2' in spy_dict:
                    spy_dict['TotalVerticalForce_Point2'][out_i, :] = np.concatenate([wr_forces['FNz_L2'], wr_forces['FNz_R2']])
                    spy_dict['TotalLateralForce_Point2'][out_i, :] = np.concatenate([wr_forces['FNy_L2'], wr_forces['FNy_R2']])

                # 5. 记录各轮对接触点处钢轨垂向位移（来自耦合系统，含准静态挠度+动态振动）
                spy_dict['RailW_Zdis_L'][out_i, :] = rail_states['RailW_Zdis_L']  # [4,]
                spy_dict['RailW_Zdis_R'][out_i, :] = rail_states['RailW_Zdis_R']  # [4,]

                if step_defects is not None:
                    fast_L = np.asarray(step_defects.get('fastener_active_L', []), dtype=bool)
                    fast_R = np.asarray(step_defects.get('fastener_active_R', []), dtype=bool)
                    void_L = np.asarray(step_defects.get('void_active_L', []), dtype=bool)
                    void_R = np.asarray(step_defects.get('void_active_R', []), dtype=bool)
                    ballast_L = np.asarray(step_defects.get('ballast_active_L', []), dtype=bool)
                    ballast_R = np.asarray(step_defects.get('ballast_active_R', []), dtype=bool)
                    gap_L = np.asarray(step_defects.get('void_gap_L', []), dtype=float)
                    gap_R = np.asarray(step_defects.get('void_gap_R', []), dtype=float)
                    ballast_eta_k_L = np.asarray(step_defects.get('ballast_kv_factor_L', []), dtype=float)
                    ballast_eta_k_R = np.asarray(step_defects.get('ballast_kv_factor_R', []), dtype=float)
                    ballast_eta_c_L = np.asarray(step_defects.get('ballast_cv_factor_L', []), dtype=float)
                    ballast_eta_c_R = np.asarray(step_defects.get('ballast_cv_factor_R', []), dtype=float)
                    contact_L = np.asarray(subrail_forces.get('Void_contact_L', []), dtype=bool)
                    contact_R = np.asarray(subrail_forces.get('Void_contact_R', []), dtype=bool)

                    spy_dict['Structure_defect_fastener_nodes'][out_i, :] = [
                        float(np.count_nonzero(fast_L)),
                        float(np.count_nonzero(fast_R)),
                    ]
                    spy_dict['Structure_defect_void_nodes'][out_i, :] = [
                        float(np.count_nonzero(void_L)),
                        float(np.count_nonzero(void_R)),
                    ]
                    spy_dict['Structure_defect_void_contact_nodes'][out_i, :] = [
                        float(np.count_nonzero(void_L & contact_L)) if contact_L.size == void_L.size else 0.0,
                        float(np.count_nonzero(void_R & contact_R)) if contact_R.size == void_R.size else 0.0,
                    ]
                    spy_dict['Structure_defect_max_gap_m'][out_i, :] = [
                        float(np.max(gap_L[void_L])) if np.any(void_L) else 0.0,
                        float(np.max(gap_R[void_R])) if np.any(void_R) else 0.0,
                    ]
                    spy_dict['Structure_defect_fastener_FV_sum'][out_i, :] = [
                        float(np.sum(fastener_forces['FV_L'][fast_L])) if np.any(fast_L) else 0.0,
                        float(np.sum(fastener_forces['FV_R'][fast_R])) if np.any(fast_R) else 0.0,
                    ]
                    spy_dict['Structure_defect_ballast_FV_sum'][out_i, :] = [
                        float(np.sum(subrail_forces['FLsV'][void_L])) if np.any(void_L) else 0.0,
                        float(np.sum(subrail_forces['FRsV'][void_R])) if np.any(void_R) else 0.0,
                    ]
                    spy_dict['Structure_defect_ballast_nodes'][out_i, :] = [
                        float(np.count_nonzero(ballast_L)),
                        float(np.count_nonzero(ballast_R)),
                    ]
                    spy_dict['Structure_defect_ballast_eta_k_max'][out_i, :] = [
                        float(np.max(ballast_eta_k_L[ballast_L])) if ballast_eta_k_L.size == ballast_L.size and np.any(ballast_L) else 0.0,
                        float(np.max(ballast_eta_k_R[ballast_R])) if ballast_eta_k_R.size == ballast_R.size and np.any(ballast_R) else 0.0,
                    ]
                    spy_dict['Structure_defect_ballast_eta_c_max'][out_i, :] = [
                        float(np.max(ballast_eta_c_L[ballast_L])) if ballast_eta_c_L.size == ballast_L.size and np.any(ballast_L) else 0.0,
                        float(np.max(ballast_eta_c_R[ballast_R])) if ballast_eta_c_R.size == ballast_R.size and np.any(ballast_R) else 0.0,
                    ]
                    spy_dict['Structure_defect_ballast_condition_FV_sum'][out_i, :] = [
                        float(np.sum(subrail_forces['FLsV'][ballast_L])) if np.any(ballast_L) else 0.0,
                        float(np.sum(subrail_forces['FRsV'][ballast_R])) if np.any(ballast_R) else 0.0,
                    ]

                if vehicle_history_only:
                    X_out[out_i, :] = X[row, car_start:car_end]
                    V_out[out_i, :] = V[row, car_start:car_end]
                    A_out[out_i, :] = A[row, car_start:car_end]
            
        print(f" -> [求解完毕] 总耗时: {time.time()-start_time:.2f} s")
        if vehicle_history_only:
            return X_out, V_out, A_out, spy_dict
        return X[save_steps, :], V[save_steps, :], A[save_steps, :], spy_dict

    # ------------------------------------------------------------------
    def _compute_curve_force(self, step_i: int, tg: dict, wr_forces: dict,
                              state, vc: float, veh, g: float) -> np.ndarray:
        """
        计算完整 35-DOF 曲线等效力向量。
        完全对应 MATLAB Force_EquivalentCurveForce.m。

        track_geometry (tg) 列顺序 (Nt, 7):
          col 0: Xw1 (1位轮对)  col 1: Xw2 (2位轮对)
          col 2: Xw3 (3位轮对)  col 3: Xw4 (4位轮对 / X0t)
          col 4: Xt1 (1号构架)  col 5: Xt2 (2号构架)
          col 6: Xc  (车体中心)

        返回长度为 35 的 ndarray，对应 GF_VEHICLE DOF 顺序：
          [0-4]   车体  Y, Z, Roll, Spin, Yaw
          [5-9]   构架1 Y, Z, Roll, Spin, Yaw
          [10-14] 构架2 Y, Z, Roll, Spin, Yaw
          [15-34] 轮对1-4 各5个 DOF (Y, Z, Roll, Spin, Yaw)
        """
        # ── 车辆参数 ──────────────────────────────────────────────────
        Mc, Mt, Mw    = veh.Mc, veh.Mt, veh.Mw
        Jcx, Jcz      = veh.Jcx, veh.Jcz
        Jtx, Jtz      = veh.Jtx, veh.Jtz
        Jwx, Jwy, Jwz = veh.Jwx, veh.Jwy, veh.Jwz
        Kpy, Cpy      = veh.Kpy, veh.Cpy
        Kpx, Cpx      = veh.Kpx, veh.Cpx
        Ksx, Ksy      = veh.Ksx, veh.Ksy
        Csx, Csy      = veh.Csx, veh.Csy
        Lc, Lt        = veh.Lc,  veh.Lt
        HcB, HBt, Htw = veh.HcB, veh.HBt, veh.Htw
        dw, ds        = veh.dw, veh.ds
        R_wheel       = veh.R                      # 标称车轮半径
        omg           = vc / R_wheel               # 标称滚动角速度

        # ── 线型参数（7个位置）───────────────────────────────────────
        K_all   = tg['K'][step_i, :]    # 曲率 (7,)
        H_all   = tg['H'][step_i, :]    # 超高角绝对值 (7,)
        dK_all  = tg['dK'][step_i, :]   # d(K)/dt (7,)
        dH_all  = tg['dH'][step_i, :]   # d(H)/dt (7,)
        ddH_all = tg['ddH'][step_i, :]  # d²(H)/dt² (7,)

        # 带符号的超高角及其导数（右转曲线 K<0 时 H 取负，与 MATLAB Theta_mile 一致）
        _eps = 1e-10
        sk      = np.sign(K_all)
        sk[np.abs(K_all) < _eps] = 0.0
        H_s     = H_all   * sk    # Theta_mile(x)    (7,)
        dH_s    = dH_all  * sk    # dTheta_mile(x)   (7,)
        ddH_s   = ddH_all * sk    # ddTheta_mile(x)  (7,)

        # 安全曲率半径（避免除以零）
        R_all = np.where(np.abs(K_all) > _eps, 1.0 / K_all, 1e10)

        # ── 各部件拆包（MATLAB 命名约定）────────────────────────────
        # 轮对 (cols 0-3)
        tsew  = H_s[:4];    dsew  = dH_s[:4];    ddsew = ddH_s[:4]
        dKw   = dK_all[:4]; Rw    = R_all[:4]
        # 构架 (cols 4-5)
        tset1, tset2   = H_s[4],    H_s[5]
        dset1, dset2   = dH_s[4],   dH_s[5]
        ddset1, ddset2 = ddH_s[4],  ddH_s[5]
        dKt1,  dKt2    = dK_all[4], dK_all[5]
        Rt1,   Rt2     = R_all[4],  R_all[5]
        # 车体 (col 6)
        tsec  = H_s[6];   dsec  = dH_s[6];   ddsec = ddH_s[6]
        dKc   = dK_all[6]; Rc   = R_all[6]

        # ── 接触半距 a0 (来自当前步轮轨接触几何) ─────────────────────
        a0_raw = wr_forces.get('a0', 0.0)
        a0_arr = np.asarray(a0_raw, dtype=float).ravel()
        if a0_arr.size == 0:
            a0_arr = np.zeros(4)
        elif a0_arr.size < 4:
            a0_arr = np.full(4, float(a0_arr[0]))
        a0c  = float(np.mean(a0_arr))
        a0t1 = float((a0_arr[0] + a0_arr[1]) / 2.0)
        a0t2 = float((a0_arr[2] + a0_arr[3]) / 2.0)
        a0w  = a0_arr[:4]                               # 各轮对接触半距

        # ── 轮对旋转角速度 Vwb (Spin DOF, Python 0-based 索引 18,23,28,33) ──
        Vwb = state.V_SpinW   # (4,) 来自 topology.py StepState.extract_state()

        # ══════════════════ 计算 35 项等效力 ══════════════════════════
        # ─── 车体 (P1-P5) ───────────────────────────────────────────
        P1  = Mc * a0c * ddsec  +  Mc * vc**2 / Rc * tsec
        P2  = 0.0
        P3  = (Mc*g*tsec - Mc*(vc**2/Rc)
               - Mc*(R_wheel + Htw + HBt + HcB) * ddsec
               + 2*Ksy*(Lc**2/Rc) + 2*Csy*Lc**2*dKc)
        P4  = (-2*HcB*Ksy*(Lc**2/Rc) - 2*HcB*Csy*Lc**2*dKc
               - Jcx * ddsec)
        P5  = -Jcz * vc * dKc

        # ─── 构架1 (P6-P10) ─────────────────────────────────────────
        P6  = Mt * a0t1 * ddset1  +  Mt * (vc**2 / Rt1) * tset1
        P7  = 0.0
        P8  = (2*Kpy*(Lt**2/Rt1) + 2*Cpy*Lt**2*dKt1
               - Ksy*(Lc**2/Rc) - Csy*Lc**2*dKc
               - Mt*(vc**2/Rt1) - Mt*(R_wheel + Htw)*ddset1
               + Mt*g*tset1)
        P9  = (-2*Htw*Kpy*(Lt**2/Rt1) - 2*Htw*Cpy*Lt**2*dKt1
               - HBt*Ksy*(Lc**2/Rc) - HBt*Csy*Lc**2*dKc
               - Jtx * ddset1)
        P10 = (-Jtz*vc*dKt1
               - 2*Ksx*ds**2*(Lc/Rc) - 2*Csx*ds**2*Lc*dKc)

        # ─── 构架2 (P11-P15) ────────────────────────────────────────
        P11 = Mt * a0t2 * ddset2  +  Mt * (vc**2 / Rt2) * tset2
        P12 = 0.0
        P13 = (2*Kpy*(Lt**2/Rt2) + 2*Cpy*Lt**2*dKt2
               - Ksy*(Lc**2/Rc) - Csy*Lc**2*dKc
               - Mt*(vc**2/Rt2) - Mt*(R_wheel + Htw)*ddset2
               + Mt*g*tset2)
        P14 = (-2*Htw*Kpy*(Lt**2/Rt2) - 2*Htw*Cpy*Lt**2*dKt2
               - HBt*Ksy*(Lc**2/Rc) - HBt*Csy*Lc**2*dKc
               - Jtx * ddset2)
        P15 = (-Jtz*vc*dKt2
               + 2*Ksx*ds**2*(Lc/Rc) + 2*Csx*ds**2*Lc*dKc)

        # ─── 轮对1 (P16-P20) ────────────────────────────────────────
        P16 = Mw * a0w[0] * ddsew[0]  +  Mw * (vc**2 / Rw[0]) * tsew[0]
        P17 = 0.0
        P18 = (-Mw*(vc**2/Rw[0]) - Mw*R_wheel*ddsew[0]
               - Kpy*(Lt**2/Rt1) - Cpy*Lt**2*dKt1
               + Mw*g*tsew[0])
        P19 = Jwy * (Vwb[0] - omg) * (vc / Rw[0])  -  Jwx * ddsew[0]
        P20 = (Jwy * dsew[0] * (Vwb[0] - omg)
               - Jwz*vc*dKw[0]
               - 2*Kpx*dw**2*Lt/Rt1 - 2*Cpx*dw**2*Lt*dKt1)

        # ─── 轮对2 (P21-P25) ────────────────────────────────────────
        P21 = Mw * a0w[1] * ddsew[1]  +  Mw * (vc**2 / Rw[1]) * tsew[1]
        P22 = 0.0
        P23 = (-Mw*(vc**2/Rw[1]) - Mw*R_wheel*ddsew[1]
               - Kpy*(Lt**2/Rt1) - Cpy*Lt**2*dKt1
               + Mw*g*tsew[1])
        P24 = Jwy * (Vwb[1] - omg) * (vc / Rw[1])  -  Jwx * ddsew[1]
        P25 = (Jwy * dsew[1] * (Vwb[1] - omg)
               - Jwz*vc*dKw[1]
               + 2*Kpx*dw**2*Lt/Rt1 + 2*Cpx*dw**2*Lt*dKt1)

        # ─── 轮对3 (P26-P30) ────────────────────────────────────────
        P26 = Mw * a0w[2] * ddsew[2]  +  Mw * (vc**2 / Rw[2]) * tsew[2]
        P27 = 0.0
        P28 = (-Mw*(vc**2/Rw[2]) - Mw*R_wheel*ddsew[2]
               - Kpy*(Lt**2/Rt2) - Cpy*Lt**2*dKt2
               + Mw*g*tsew[2])
        P29 = Jwy * (Vwb[2] - omg) * (vc / Rw[2])  -  Jwx * ddsew[2]
        P30 = (Jwy * dsew[2] * (Vwb[2] - omg)
               - Jwz*vc*dKw[2]
               - 2*Kpx*dw**2*Lt/Rt2 - 2*Cpx*dw**2*Lt*dKt2)

        # ─── 轮对4 (P31-P35) ────────────────────────────────────────
        P31 = Mw * a0w[3] * ddsew[3]  +  Mw * (vc**2 / Rw[3]) * tsew[3]
        P32 = 0.0
        P33 = (-Mw*(vc**2/Rw[3]) - Mw*R_wheel*ddsew[3]
               - Kpy*(Lt**2/Rt2) - Cpy*Lt**2*dKt2
               + Mw*g*tsew[3])
        P34 = Jwy * (Vwb[3] - omg) * (vc / Rw[3])  -  Jwx * ddsew[3]
        P35 = (Jwy * dsew[3] * (Vwb[3] - omg)
               - Jwz*vc*dKw[3]
               + 2*Kpx*dw**2*Lt/Rt2 + 2*Cpx*dw**2*Lt*dKt2)

        return np.array([
            P1,  P2,  P3,  P4,  P5,
            P6,  P7,  P8,  P9,  P10,
            P11, P12, P13, P14, P15,
            P16, P17, P18, P19, P20,
            P21, P22, P23, P24, P25,
            P26, P27, P28, P29, P30,
            P31, P32, P33, P34, P35,
        ], dtype=float)
