'''
Author: Niscienc 60505912+2099193635@users.noreply.github.com
Date: 2026-06-22 16:31:39
LastEditors: Niscienc 60505912+2099193635@users.noreply.github.com
LastEditTime: 2026-06-29 13:34:15
FilePath: \vtcm-simulation\generate_main.py
Description: 

Copyright (c) 2026 by ${git_name_email}, All Rights Reserved. 
'''
'''
Author: Niscienc 60505912+2099193635@users.noreply.github.com
Date: 2026-03-07 16:08:44
LastEditors: Niscienc 60505912+2099193635@users.noreply.github.com
LastEditTime: 2026-06-29 11:38:00
FilePath: C:/VTCM_PYTHON/vtcm-simulation/generate_main.py
Description: 
Copyright (c) 2026 by ${git_name_email}, All Rights Reserved. 
'''
from configs.parameters import *
from defect_injector.irregularity import Irregularity
from defect_injector.settlement_profiles import load_settlement_config
from defect_injector.geometry_defects import load_geometry_defect_config
import numpy as np
import argparse
from configs.topology import SystemTopology
from physics_modules.contact_geometry import WheelRailContactProcessor
from physics_modules.suspension import SuspensionSystem
from physics_modules.wheel_rail_contact import WheelRailInteraction
from physics_modules.equation_of_motion import GeneralForceAssembler
from physics_modules.solver import DynamicSolver, SystemDynamics
from utils.post_processing import ResultPlotter
import datetime 
import os
import json
import shutil
import time
from physics_modules.rail_modal import RailModalDynamics
from physics_modules.substructure import SubstructureDynamics
from physics_modules.structure_defects import StructureDefectManager



def _str_to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ('on', 'true', '1', 'yes', 'y')


def _path_size_bytes(path):
    if not path or not os.path.exists(path):
        return 0
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            fp = os.path.join(root, name)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def _estimate_modal_counts_by_frequency(rail, integration, cutoff_hz, max_modes):
    """
    Estimate rail modal truncation counts from the current finite rail length.

    The rail modal equations in equation_of_motion.py use mass-normalized
    modal stiffnesses, so natural frequency is sqrt(k_modal) / (2*pi).
    """
    cutoff_hz = float(cutoff_hz)
    max_modes = int(max_modes)
    if cutoff_hz <= 0.0:
        return None

    n = np.arange(1, max_modes + 1, dtype=float)
    pi_over_l = np.pi / float(integration.Ljs)

    f_v = np.sqrt((rail.E * rail.Iy / rail.mr) * (pi_over_l * n) ** 4) / (2.0 * np.pi)
    f_l = np.sqrt((rail.E * rail.Iz / rail.mr) * (pi_over_l * n) ** 4) / (2.0 * np.pi)
    f_t = np.sqrt((rail.Gk / (rail.pr * rail.I0)) * (pi_over_l * n) ** 2) / (2.0 * np.pi)

    def _count(freq):
        hits = np.flatnonzero(freq <= cutoff_hz)
        return max(1, int(hits[-1] + 1)) if hits.size else 1

    return _count(f_v), _count(f_l), _count(f_t), f_v, f_l, f_t


def _build_structure_window_plan(args, integration, all_s_abs):
    """
    Build a time-varying structure truncation/window plan.

    Current solver still uses a fixed finite rail model. This plan records the
    intended local window coordinates and active fastener range so the generated
    result is explicit about model truncation, and later bridge/substructure
    solvers can consume the same fields directly.
    """
    mode = str(args.structure_truncation_mode).strip().lower()
    model_length_m = float(integration.Ljs)
    requested_length_m = float(args.structure_window_length)
    if requested_length_m <= 0.0:
        requested_length_m = model_length_m

    lead_m = float(args.structure_window_lead)
    trail_m = float(args.structure_window_trail)
    if lead_m <= 0.0 and trail_m <= 0.0:
        trail_m = min(float(integration.X0), 0.5 * requested_length_m)
        lead_m = max(0.0, requested_length_m - trail_m)
    elif lead_m <= 0.0:
        lead_m = max(0.0, requested_length_m - trail_m)
    elif trail_m <= 0.0:
        trail_m = max(0.0, requested_length_m - lead_m)
    else:
        requested_length_m = lead_m + trail_m

    s_ref = all_s_abs[:, 3]
    if mode == 'moving_window':
        # Move the window on the fastener grid. This gives the solver integer
        # node shifts, so discrete substructure states can be migrated directly.
        start0 = float(s_ref[0] - trail_m)
        shift_nodes = np.floor((s_ref - trail_m - start0) / integration.Lkj).astype(int)
        shift_nodes = np.maximum(shift_nodes, 0)
        window_start_abs = start0 + shift_nodes * integration.Lkj
        window_end_abs = window_start_abs + requested_length_m
        local_s = all_s_abs - window_start_abs[:, None]
    elif mode == 'fixed_window':
        start_arg = float(args.structure_window_start_m)
        start_m = start_arg if start_arg > 0.0 else float(s_ref[0] - trail_m)
        window_start_abs = np.full_like(s_ref, start_m)
        window_end_abs = window_start_abs + requested_length_m
        local_s = all_s_abs - window_start_abs[:, None]
        shift_nodes = np.zeros_like(s_ref, dtype=int)
    else:
        window_start_abs = np.full_like(s_ref, float(integration.S0_mileage))
        window_end_abs = window_start_abs + model_length_m
        local_s = all_s_abs - window_start_abs[:, None]
        shift_nodes = np.zeros_like(s_ref, dtype=int)

    node_start = shift_nodes.copy()
    node_end = node_start + int(round(requested_length_m / integration.Lkj))

    in_window = (local_s >= 0.0) & (local_s <= requested_length_m)
    boundary_buffer = max(0.0, float(args.structure_boundary_buffer))
    if boundary_buffer > 0.0:
        dist_to_edge = np.minimum(local_s, requested_length_m - local_s)
        boundary_weight = np.clip(dist_to_edge / boundary_buffer, 0.0, 1.0)
    else:
        boundary_weight = np.ones_like(local_s)

    return {
        'mode': mode,
        'window_length_m': requested_length_m,
        'window_lead_m': lead_m,
        'window_trail_m': trail_m,
        'window_start_abs_m': window_start_abs,
        'window_end_abs_m': window_end_abs,
        'local_s_m': local_s,                                                               
        'wheel_in_window': in_window[:, :4],
        'boundary_weight': boundary_weight,
        'active_node_start': node_start,
        'active_node_end': node_end,
        'window_shift_nodes': shift_nodes,
        'model_length_m': model_length_m,
        'boundary_buffer_m': boundary_buffer,
        'state_migration': _str_to_bool(getattr(args, 'structure_state_migration', 'On')) and mode == 'moving_window',
    }


def parse_arguments():
    """统一定义并解析所有的外部输入参数"""
    parser = argparse.ArgumentParser(description="车辆-轨道耦合动力学仿真参数配置")
    
    # 1. 宏观运行参量
    parser.add_argument('--vx_set', type=float, default=215.0, help='车辆运行速度 (km/h)')
    parser.add_argument('--tz', type=float, default=5.0, help='仿真总时长 (s)')
    parser.add_argument('--tstep', type=float, default=2e-4, help='积分步长 (s)')
    parser.add_argument('--start_mileage', type=float, default=271.822147720011, help='仿真起始绝对里程 (km)')
    parser.add_argument('--curve_file_dir', type=str, default='preprocessing/台账/处理后/curve_parameters.csv', help='曲线参数文件路径')
    parser.add_argument('--gradient_file_dir', type=str, default='preprocessing/台账/处理后/gradient_parameters.csv', help='坡度参数文件路径')
    parser.add_argument('--cache_file_dir', type=str, default='configs/track_cache.npz', help='轨道缓存文件路径')
    parser.add_argument('--force_rebuild', type=str, default='Off', choices=['On', 'Off'], help='是否强制重建力元（默认On）')

    # 2. 物理拓扑配置
    parser.add_argument('--vehicle_type', type=str, default='高速客车', help='车辆类型')
    parser.add_argument('--rail_type', type=str, default='CHN60', help='钢轨类型')
    parser.add_argument('--fastener_type', type=str, default='Standard_KV', help='扣件类型')
    parser.add_argument('--param_profile_dir', type=str, default='configs/test', help='参数配置目录（标准参数: configs/standard；试验参数: configs/试验名）')
    
    # 3. 轨道激扰控制
    parser.add_argument('--irr_type', type=str, default='外部导入', help='激扰类型 (随机不平顺/谐波不平顺/无不平顺/外部导入)')
    parser.add_argument('--irr_lead_time', type=float, default=2.0, help='不平顺前置无激励时长(s)，随机/外部导入建议设为2.0')
    parser.add_argument('--psd_type', type=str, default='高铁谱', help='功率谱类型 (高铁谱/干线谱/美国谱/德国低干扰谱)')
    parser.add_argument('--defect_switch', type=str, default='off', choices=['on', 'off'], help='是否开启局部病害')
    parser.add_argument('--input_path', type=str, default='', help='不平顺输入路径')
    parser.add_argument('--output_path', type=str, default='', help='不平顺输出路径')
        # 外部不平顺配置
    parser.add_argument('--external_mileage_mode', type=str, default='absolute', choices=['absolute', 'relative'], help='外部不平顺里程模式')
    parser.add_argument('--external_distance_unit', type=str, default='km', choices=['m', 'km'], help='外部不平顺里程单位')
    parser.add_argument('--Type2', type=str, default='空间谱', choices=['空间谱', '时间谱', '时间序列'], help='外部不平顺文件类型（空间谱/时间谱/时间序列）')
    _ext_base   = 'preprocessing/静检数据/呼局/20210416/处理后/静检上行20210416-271-278.merged.aligned.external'
    _ext_prefix = '静检上行20210416-271-278.merged.aligned'
    parser.add_argument('--external_files', type=str, nargs='*',
                        default=[
                            f'VL={_ext_base}/{_ext_prefix}_VL.txt',
                            f'VR={_ext_base}/{_ext_prefix}_VR.txt',
                            f'LL={_ext_base}/{_ext_prefix}_LL.txt',
                            f'LR={_ext_base}/{_ext_prefix}_LR.txt',
                        ],
                        help='外部不平顺文件路径，格式: KEY=VALUE，支持 VL/VR/LL/LR 四个通道')
    parser.add_argument('--settlement_switch', type=str, default='Off', choices=['On', 'Off'], help='是否叠加过渡段沉降')
    parser.add_argument('--geometry_defect_switch', type=str, default='Off', choices=['On', 'Off'], help='enable geometry defects only for spectrum-generated irregularity')
    parser.add_argument('--geometry_defect_config', type=str, default='', help='geometry defect config path; YAML, JSON, and CSV are supported')
    parser.add_argument('--settlement_config', type=str, default='', help='过渡段沉降 YAML 配置文件路径')
    parser.add_argument('--random_seed', type=int, default=-1, help='随机不平顺种子；小于0表示不固定')

    # 4. 下部结构
    parser.add_argument('--N_sub', type=int, default=2000, help='轨道下部结构离散单元数量')
    parser.add_argument('--X0', type=float, default=20.0, help='仿真初始状态位移 (m)')
    parser.add_argument('--auto_extend_rail', type=str, default='On', choices=['On', 'Off'], help='当Tz对应走行距离超过当前柔性轨道时，自动增大N_sub而不是截断Tz')
    parser.add_argument('--structure_truncation_mode', type=str, default='moving_window', choices=['global', 'fixed_window', 'moving_window'], help='长大结构截断模式：global=现有整段模型，fixed_window=固定局部窗口，moving_window=随车移动窗口元数据')
    parser.add_argument('--structure_window_length', type=float, default=240.0, help='局部结构窗口长度(m)，0表示使用当前柔性轨道总长')
    parser.add_argument('--structure_window_lead', type=float, default=180.0, help='移动窗口中车辆前方保留长度(m)，0时自动按窗口长度-X0估计')
    parser.add_argument('--structure_window_trail', type=float, default=60.0, help='移动窗口中车辆后方保留长度(m)，0时自动按X0估计')
    parser.add_argument('--structure_window_start_m', type=float, default=0.0, help='fixed_window模式下窗口绝对起点(m)，0表示起点里程-X0')
    parser.add_argument('--structure_boundary_buffer', type=float, default=20.0, help='局部窗口两端缓冲/吸收区长度(m)，用于输出边界权重')
    parser.add_argument('--structure_state_migration', type=str, default='On', choices=['On', 'Off'], help='moving_window下按扣件节点推进并迁移轨道/轨下结构状态')
    parser.add_argument('--modal_truncation', type=str, default='frequency', choices=['fixed_count', 'frequency'], help='钢轨模态截断方式：fixed_count=固定阶数，frequency=按频率上限估算阶数')
    parser.add_argument('--modal_cutoff_hz', type=float, default=250.0, help='frequency模态截断的最高关注频率(Hz)')
    parser.add_argument('--modal_max_modes', type=int, default=1200, help='frequency模态截断搜索上限，防止长大结构阶数无限增长')

    # 5. 积分参数
    parser.add_argument('--alpha', type=float, default=0.5, help='Newmark-beta 方法的 alpha 参数')
    parser.add_argument('--beta', type=float, default=0.25, help='Newmark-beta 方法的 beta 参数')
    parser.add_argument('--g', type=float, default=9.81, help='重力加速度 (m/s^2)')

    # 6. 力元控制开关
    parser.add_argument('--switch_curve_track', type=str, default='Off', choices=['On', 'Off'], help='是否开启线型引起的附加力')
    parser.add_argument('--switch_2point_contact', type=str, default='Off', choices=['On', 'Off'], help='是否开启两点接触模型')
    parser.add_argument('--switch_extra_force_element', type=str, default='Off', choices=['On', 'Off'], help='是否开启额外力元')
    parser.add_argument('--switch_pad_zone', type=str, default='Off', choices=['On', 'Off'], help='是否开启扣件区分')
    parser.add_argument('--switch_pad_partition', type=str, default='Off', choices=['On', 'Off'], help='是否开启扣件分区')
    parser.add_argument('--switch_railcant_unsymmetric', type=str, default='Off', choices=['On', 'Off'], help='是否开启轨道超高非对称')
    
    # 7. 锁定控制开关
    parser.add_argument('--switch_lock_veh_non_z', type=str, default='Off', choices=['On', 'Off'], help='是否锁定车辆非垂向自由度')
    parser.add_argument('--switch_lock_axlebox', type=str, default='Off', choices=['On', 'Off'], help='是否锁定轴箱自由度')
    parser.add_argument('--switch_lock_substructure', type=str, default='Off', choices=['On', 'Off'], help='是否锁定轨道下部结构自由度')
    
    # 8. 输出与可视化控制
    parser.add_argument('--save_data', type=str, default='On', choices=['On', 'Off'], help='是否将结果保存到本地')
    parser.add_argument('--save_dof_mode', type=str, default='vehicle', choices=['full', 'vehicle'], help='结果保存自由度模式：full=完整系统，vehicle=仅车体自由度')
    parser.add_argument('--save_stride', type=int, default=10, help='输出保存步长：积分每 save_stride 步记录一次，长线路建议 5~20')
    parser.add_argument('--save_spy_level', type=str, default='core', choices=['core', 'full'], help='监测量保存级别：core=核心响应，full=包含原始蠕滑等大矩阵')
    parser.add_argument('--save_compressed', type=str, default='Off', choices=['On', 'Off'], help='是否使用 np.savez_compressed 压缩保存；长线路可设 Off 加快落盘')
    parser.add_argument('--output_tier', type=str, default='normal', choices=['normal', 'debug'], help='结果输出档位：normal 过滤调试大字段，debug 保留完整监测字段')
    parser.add_argument('--save_format', type=str, default='npz_raw', choices=['npz_raw', 'npz_compressed', 'zarr'], help='结果保存格式')
    parser.add_argument('--project_name', type=str, default='long_distance', help='结果项目名，保存路径为 results/项目名/运行名')
    parser.add_argument('--run_note', type=str, default='10km_moving_window', help='试验描述（会写入结果目录名与参数归档）')
    parser.add_argument('--plot_figs', type=str, default='On', choices=['On', 'Off'], help='仿真完成后是否自动弹出图表')
    # (如果在 Jupyter/IDE 中直接运行，用 parse_known_args 防止参数识别报错)
    parser.add_argument('--structure_defect_switch', type=str, default='Off', choices=['On', 'Off'], help='enable local fastener failure and sleeper void defects')
    parser.add_argument('--structure_defect_config', type=str, default='', help='structure defect config path; YAML, JSON, and CSV are supported')
    args, _ = parser.parse_known_args()
    return args
    

def main(args):
    main_start_time = time.perf_counter()
    output_tier = str(getattr(args, 'output_tier', 'normal')).strip().lower()
    if output_tier not in ('normal', 'debug'):
        output_tier = 'normal'
    save_format = str(getattr(args, 'save_format', 'npz_raw')).strip().lower()
    if save_format not in ('npz_raw', 'npz_compressed', 'zarr'):
        save_format = 'npz_raw'
    requested_save_dof_mode = str(getattr(args, 'save_dof_mode', 'vehicle')).strip().lower()
    actual_save_dof_mode = requested_save_dof_mode
    if output_tier == 'normal' and requested_save_dof_mode == 'full':
        actual_save_dof_mode = 'vehicle'
        print(" -> [输出策略] normal 档已将 save_dof_mode=full 自动降为 vehicle，避免写出完整系统大数组")
    #=========================Part1 导入运算参数=========================#
    # 1. 车辆系统参数
    veh_emu = VehicleParams(vehicle_type=args.vehicle_type, yaml_dir=args.param_profile_dir)
    # 2. 钢轨参数
    rail = RailParams(rail_type=args.rail_type, yaml_dir=args.param_profile_dir)
    # 3. 扣件参数
    faster_kv = Fastener_KV(fastener_type=args.fastener_type, yaml_dir=args.param_profile_dir)
    fdkv_params = FastenerFDKVParams(temperature=20, fdkv_switch='ON', yaml_dir=args.param_profile_dir)
    # 4. 轨下结构参数
    subrail_standard = Subrail_Params(subrail_type='Standard_Subrail', yaml_dir=args.param_profile_dir)
    print(f" -> [参数配置] 已加载参数目录: {args.param_profile_dir}")
    track_alignment = RealTrackAlignment(
        curve_file_dir= args.curve_file_dir,
        gradient_file_dir=args.gradient_file_dir,
        cache_file_dir=args.cache_file_dir,
        force_rebuild = str(args.force_rebuild).strip().lower() == 'on'
    )
    start_mileage_m = args.start_mileage * 1000.0
    test_s_straight = start_mileage_m
    # 5. 线型参数
    k1, h1, g1, Rcurve, Thetacurve, Lcurve, curvecase, L1, L2, Lz1, Lz2, hcg, S, ZH_abs = track_alignment.get_geometry_at(test_s_straight)

    #=========================Part2 生成循环积分参数、组件=========================#
    # 1. 设置钢轨模态求解数量
    mode_params = ModesParameters()
    # 2. 设置数值积分参数
    S0_start = start_mileage_m
    _irr_type_norm = str(args.irr_type).strip()
    _need_irre_lead = _irr_type_norm in ('随机不平顺', '外部导入')
    _tz_effective = float(args.tz) + (float(args.irr_lead_time) if _need_irre_lead else 0.0)
    if _need_irre_lead and float(args.irr_lead_time) > 0:
        print(f" -> [不平顺前置缓冲] 已启用 {float(args.irr_lead_time):.2f}s 无不平顺工况，总仿真时长: {_tz_effective:.2f}s")

    _window_mode = str(args.structure_truncation_mode).strip().lower()
    _use_true_moving_window = _window_mode == 'moving_window'

    # moving_window 下真正截断动力学模型：只建立局部窗口长度的轨道/轨下结构。
    # global/fixed_window 仍沿用整段有限轨道，必要时可 auto_extend。
    _n_sub_runtime = int(args.N_sub)
    _vc_preview = float(args.vx_set) / 3.6
    _required_running_m = _vc_preview * _tz_effective
    _window_length_runtime = float(args.structure_window_length) if float(args.structure_window_length) > 0.0 else _n_sub_runtime * faster_kv.Lkj
    if _use_true_moving_window:
        _n_sub_runtime = max(4, int(np.ceil(_window_length_runtime / faster_kv.Lkj)))
        print(f" -> [真实移动窗口] 动力学轨道长度截断为 {_n_sub_runtime * faster_kv.Lkj:.2f} m "
              f"(N_sub={_n_sub_runtime})，全线路走行距离 {_required_running_m:.2f} m 不再扩展到模型自由度。")
    else:
        _required_ljs = _required_running_m + float(args.X0) + 2.0 * (veh_emu.Lt + veh_emu.Lc) + float(args.structure_boundary_buffer)
        if _str_to_bool(args.auto_extend_rail):
            _required_n_sub = int(np.ceil(_required_ljs / faster_kv.Lkj))
            if _required_n_sub > _n_sub_runtime:
                print(f" -> [长大结构] auto_extend_rail=On: N_sub {_n_sub_runtime} -> {_required_n_sub}, "
                      f"覆盖目标走行距离 {_required_running_m:.2f} m")
                _n_sub_runtime = _required_n_sub

    integration = IntegrationParams(
        Lc=veh_emu.Lc,                      # 直接从车辆实例导入
        Lt=veh_emu.Lt,                      # 直接从车辆实例导入
        R=veh_emu.R,                        # 直接从车辆实例导入
        Lkj=faster_kv.Lkj,                  # 直接从扣件实例导入
        Vx_set=args.vx_set,                 # 设定车速 300 km/h
        Tz=_tz_effective,                   # 随机/外部导入时自动叠加前置无不平顺时长
        Tstep=args.tstep,
        S0_mileage=S0_start,                # 设定地球绝对坐标起点
        Nsub=_n_sub_runtime,                # 设定轨下结构离散单元数量
        X0 = args.X0,                       # 设定仿真初始状态位移
        enforce_track_boundary = not _use_true_moving_window,
        alpha = args.alpha,                 # Newmark-beta 方法的 alpha 参数
        beta = args.beta,                   # Newmark-beta 方法的 beta 参数
        g = args.g                         # 重力加速度
    )

    if _use_true_moving_window:
        print(" -> [真实移动窗口] 求解器将使用 structure_window.local_s_m 作为钢轨局部坐标；"
              "下部结构状态保留在局部窗口坐标内。")

    _modal_freqs = None
    if str(args.modal_truncation).strip().lower() == 'frequency':
        _modal_est = _estimate_modal_counts_by_frequency(
            rail=rail,
            integration=integration,
            cutoff_hz=args.modal_cutoff_hz,
            max_modes=args.modal_max_modes
        )
        if _modal_est is not None:
            _nv, _nl, _nt, _fv, _fl, _ft = _modal_est
            mode_params = ModesParameters(NV=_nv, NL=_nl, NT=_nt)
            _modal_freqs = (_fv[:_nv], _fl[:_nl], _ft[:_nt])
            print(f" -> [模态截断] frequency: cutoff={float(args.modal_cutoff_hz):.1f} Hz, "
                  f"NV/NL/NT={_nv}/{_nl}/{_nt}, Ljs={integration.Ljs:.2f} m")
    else:
        print(f" -> [模态截断] fixed_count: NV/NL/NT={mode_params.NV}/{mode_params.NL}/{mode_params.NT}")

    sim_switches = ExtraforceElementSwitch(
            Switch_CurveTrack=args.switch_curve_track,
            Switch_2PointContact=args.switch_2point_contact,
            Switch_ExtraForceElement = args.switch_extra_force_element,
            Switch_PadZone = args.switch_pad_zone,
            Switch_PadPartition = args.switch_pad_partition,
            Switch_RailCant_Unsymmetric = args.switch_railcant_unsymmetric
        )
    
    # === 预计算各计算步各轮对位置的线型参数 (曲率 / 超高 / 坡度 及其时间变化率) ===
    # 问题根源：原代码仅在 s0_start+1000 处采样一次，导致曲率/超高在整个积分中
    # 始终为常数，车辆通过不同线型时无法感知线型变化。
    # 修复：按每步各轮对的真实绝对里程预计算 (Nt, 4) 参数矩阵，供 solver 每步索引取用。
    _t_vec = np.arange(integration.Nt) * integration.Tstep
    _s4    = integration.S0_mileage + integration.X0 + integration.Vc * _t_vec   # 4位轮对绝对里程
    _Lc, _Lt = integration.Lc, integration.Lt
    # 7个关键位置的绝对里程 (Nt,7):
    #   0=Xw1, 1=Xw2, 2=Xw3, 3=Xw4(=X0t), 4=Xt1, 5=Xt2, 6=Xc
    # 对应 MATLAB Force_EquivalentCurveForce.m 中各构件中心位置计算：
    #   Xw4=X0t, Xw3=X0t+2*Lt, Xw2=X0t+2*Lc, Xw1=X0t+2*(Lt+Lc)
    #   Xt1=Xw1-Lt=X0t+2*Lc+Lt, Xt2=X0t+Lt, Xc=Xw3+Lc-Lt=X0t+Lc+Lt
    _all_s = np.column_stack([
        _s4 + 2*(_Lc + _Lt),    # col 0: Xw1 (1位轮对)
        _s4 + 2*_Lc,            # col 1: Xw2 (2位轮对)
        _s4 + 2*_Lt,            # col 2: Xw3 (3位轮对)
        _s4,                    # col 3: Xw4 (4位轮对 / X0t 基准)
        _s4 + 2*_Lc + _Lt,      # col 4: Xt1 (1号构架中心)
        _s4 + _Lt,              # col 5: Xt2 (2号构架中心)
        _s4 + _Lc + _Lt,        # col 6: Xc  (车体中心)
    ])  # shape: (Nt, 7)

    _sg = track_alignment.s_grid
    _k_mat  = np.column_stack([np.interp(_all_s[:, j], _sg, track_alignment.k_grid) for j in range(7)])
    _h_mat  = np.column_stack([np.interp(_all_s[:, j], _sg, track_alignment.h_grid) for j in range(7)])
    _g_mat  = np.column_stack([np.interp(_all_s[:, j], _sg, track_alignment.g_grid) for j in range(7)])
    # 时间变化率（中心差分；边界用单侧差分）
    _dk_mat  = np.gradient(_k_mat,  integration.Tstep, axis=0)   # d(K)/dt (Nt, 7)
    _dh_mat  = np.gradient(_h_mat,  integration.Tstep, axis=0)   # d(H)/dt = dTheta/dt (Nt, 7)
    _ddh_mat = np.gradient(_dh_mat, integration.Tstep, axis=0)   # d²(H)/dt² (Nt, 7)

    _structure_window = _build_structure_window_plan(args, integration, _all_s)
    track_geometry = {
        'K'  : _k_mat,     # 曲率 1/m                     (Nt, 7)
        'H'  : _h_mat,     # 超高角 rad（无符号绝对值）   (Nt, 7)
        'G'  : _g_mat,     # 坡度（无量纲）               (Nt, 7)
        'dK' : _dk_mat,    # 曲率变化率 1/m/s             (Nt, 7)
        'dH' : _dh_mat,    # 超高角变化率 rad/s           (Nt, 7)
        'ddH': _ddh_mat,   # 超高角二阶变化率 rad/s²      (Nt, 7)
        'S'  : _all_s,     # 7个位置的绝对里程 m          (Nt, 7)
        'use_local_window_dynamics': _use_true_moving_window,
        'structure_window': _structure_window
    }
    _structure_defects = StructureDefectManager.from_config(
        config_path=args.structure_defect_config,
        integration_params=integration,
        structure_window=_structure_window,
        enabled=_str_to_bool(args.structure_defect_switch)
    )
    track_geometry['structure_defects'] = _structure_defects
    _structure_defect_summary = _structure_defects.summary()
    if _structure_defects.is_active():
        print(f" -> [结构缺陷] 已读取 {_structure_defect_summary['record_count']} 条缺陷记录: {args.structure_defect_config}")
    elif _str_to_bool(args.structure_defect_switch):
        print(" -> [结构缺陷] structure_defect_switch=On，但未读取到缺陷记录；本次按无结构缺陷运行。")
    _window_ok_ratio = float(np.mean(_structure_window['wheel_in_window']))
    print(f" -> [结构截断] mode={_structure_window['mode']}, window={_structure_window['window_length_m']:.2f} m, "
          f"wheel-in-window={_window_ok_ratio*100:.1f}%")
    if _window_ok_ratio < 1.0:
        print(" -> [警告] 存在轮对位置落在结构截断窗口外，请增大 structure_window_length/lead/trail 或检查 fixed_window 起点。")
    print(f" -> [线型预计算] 完成! 曲率范围: [{_k_mat.min():.5f}, {_k_mat.max():.5f}] 1/m, "
          f"超高范围: [{_h_mat.min()*1000:.1f}, {_h_mat.max()*1000:.1f}] mm")

    # 兼容别名："时间序列" 统一映射为 irregularity 内部使用的 "时间谱"
    _type2 = '时间谱' if args.Type2 == '时间序列' else args.Type2
    settlement_specs = []
    if _str_to_bool(getattr(args, 'settlement_switch', 'Off')):
        settlement_config = str(getattr(args, 'settlement_config', '')).strip()
        if not settlement_config:
            raise ValueError('settlement_switch=On requires --settlement_config')
        settlement_specs = load_settlement_config(settlement_config)
        print(f" -> [过渡段沉降] 已加载 {len(settlement_specs)} 个沉降曲线: {settlement_config}")
    if _str_to_bool(getattr(args, 'geometry_defect_switch', 'Off')):
        geometry_defect_config = str(getattr(args, 'geometry_defect_config', '')).strip()
        if not geometry_defect_config:
            raise ValueError('geometry_defect_switch=On requires --geometry_defect_config')
        geometry_defect_specs = load_geometry_defect_config(geometry_defect_config)
        if args.irr_type == '外部导入':
            print(' -> [geometry defects] external irregularity import keeps the input profile unchanged; geometry defects are archived only.')
        else:
            print(f" -> [geometry defects] loaded {len(geometry_defect_specs)} geometry defect records: {geometry_defect_config}")
    if int(getattr(args, 'random_seed', -1)) >= 0:
        np.random.seed(int(args.random_seed))
        print(f" -> [随机不平顺] 已固定随机种子: {int(args.random_seed)}")

    track_simulator = Irregularity(
        Lc=integration.Lc, 
        Lt=integration.Lt, 
        Vc=integration.Vc,       # 直接读取中枢算好的 m/s 速度
        Tstep=integration.Tstep, # 直接读取中枢步长
        Tz=integration.Tz,       # 直接读取中枢(可能被防脱轨截断后)的真实时长
        Nt=integration.Nt,       # 直接读取中枢算好的总步数
        type=args.irr_type,
        Tstart=max(0.0, float(args.irr_lead_time)),  # 不平顺起步前无激励时长(s)
        Type2 = _type2,      # 时间谱/空间谱
        powerSpectrum_type=args.psd_type, 
        mile=max(2000, int(np.ceil(integration.Lz / 1000.0)) + 2),  # 按实际走行距离扩展里程池
        external_mileage_mode = args.external_mileage_mode,  # 外部里程模式（绝对里程/相对里程）
        external_distance_unit = args.external_distance_unit,  # 外部里程单位（m/km）
        external_start_mileage = args.start_mileage,   # 外部里程起点（绝对/相对里程均可，单位由 external_distance_unit 控制）
        input_path = args.input_path,       # 不平顺输入路径
        output_path = args.output_path,     # 不平顺输出路径
        settlement_specs = settlement_specs,
        geometry_defect_specs = geometry_defect_specs
    )
    # 将 KEY=VALUE 列表解析为字典，并自动注入 start_mileage
    _external_files_dict = {}
    for _item in (args.external_files or []):
        if '=' in _item:
            _k, _v = _item.split('=', 1)
            _external_files_dict[_k.strip()] = _v.strip()
    # 注意：仅在 absolute 模式下默认注入 start_mileage，避免 relative 文件被错误平移
    if args.external_mileage_mode == 'absolute':
        _external_files_dict.setdefault('start_mileage', args.start_mileage)

    track_excitation = track_simulator.excitation_irregularity(
        defect_switch=args.defect_switch,
        external_files=_external_files_dict)
    bz_L, by_L, dbz_L, dby_L, bz_R, by_R, dbz_R, dby_R, a, L = track_excitation
    print(f"左轨垂向不平顺范围: [{bz_L.min()*1000:.2f}, {bz_L.max()*1000:.2f}] mm")
    print(f"右轨垂向不平顺范围: [{bz_R.min()*1000:.2f}, {bz_R.max()*1000:.2f}] mm")
    #=========================Part 3 实例化物理引擎与系统拓扑=========================#
    # 1. 提取接触几何前处理信息
    processor = WheelRailContactProcessor()
    rail_raw = np.loadtxt('Profile_file/rail_fade.txt')
    wheel_raw = np.loadtxt('Profile_file/wheel_fade.txt') 
    geom_info = processor.process_pre_information(rail_raw, wheel_raw)
    
    # 2. 实例化系统拓扑
    rail_modal_sys = RailModalDynamics(rail_params=rail, integration_params=integration, mode_params=mode_params)
    substructure_sys = SubstructureDynamics(fastener_params=faster_kv, rail_params=rail, subrail_params=subrail_standard)
    topo = SystemTopology(Nt=integration.Nt, Nsub=integration.Nsub, NV=mode_params.NV, NL=mode_params.NL, NT=mode_params.NT)
    ap = Antiyawer_parameters(yaml_dir=args.param_profile_dir)
    ep = ExtraForceElements_parameters(Lc=veh_emu.Lc, yaml_dir=args.param_profile_dir)
    suspension_sys = SuspensionSystem(veh_params=veh_emu, antiyawer_params=ap, extra_params=ep)
    wr_interaction = WheelRailInteraction(geom_info, veh_params=veh_emu)
    gf_assembler = GeneralForceAssembler(
        veh_params=veh_emu, integration_params=integration, rail_params=rail, 
        subrail_params=subrail_standard, mode_params=mode_params, anitiyawer_params=ap
    )
    sys_dynamics = SystemDynamics(veh_params=veh_emu, veh_int=integration, para_subrail=subrail_standard, control_mode= mode_params, para_extra_force=ep)
    physics_engines = {
        'suspension': suspension_sys, 'contact': wr_interaction, 'assembler': gf_assembler,
        'dynamics': sys_dynamics, 'rail_modal': rail_modal_sys, 'substructure': substructure_sys,
        'veh_params': veh_emu   # 车辆参数对象，供曲线等效力计算使用
    }
    
    #=========================Part 4 启动主积分器求解=========================#
    solver = DynamicSolver(topology=topo, 
                            integration_params=integration,
                            switch_lock_veh_non_z = args.switch_lock_veh_non_z,  
                            switch_lock_axlebox = args.switch_lock_axlebox, 
                            switch_lock_substructure = args.switch_lock_substructure)
    X, V, A, spy_data = solver.solve(
        track_excitation=track_excitation,
        geom_info=geom_info,
        engines=physics_engines,
        track_geometry=track_geometry,
        sim_switches=sim_switches,
        save_dof_mode=actual_save_dof_mode,
        save_stride=args.save_stride,
        save_spy_level=args.save_spy_level
    )
    _save_steps = np.asarray(spy_data.get('Output_step_index', np.arange(A.shape[0]) * max(1, int(args.save_stride))), dtype=np.int64)
    _dt_output = integration.Tstep * max(1, int(args.save_stride))

    # =========================Part 4.1 追加后处理元数据=========================#
    # 保存不平顺（采用第4轮对基准轨道激励，长度 Nt+1）
    # 约定：bz/by 为位移(m)，dbz/dby 为变化率(m/s)
    try:
        _irr_idx = np.clip(_save_steps, 0, bz_L.shape[1] - 1)
        spy_data['Irre_bz_L_ref'] = bz_L[3, _irr_idx].astype(np.float32)
        spy_data['Irre_bz_R_ref'] = bz_R[3, _irr_idx].astype(np.float32)
        spy_data['Irre_by_L_ref'] = by_L[3, _irr_idx].astype(np.float32)
        spy_data['Irre_by_R_ref'] = by_R[3, _irr_idx].astype(np.float32)
        spy_data['Irre_dbz_L_ref'] = dbz_L[3, _irr_idx].astype(np.float32)
        spy_data['Irre_dbz_R_ref'] = dbz_R[3, _irr_idx].astype(np.float32)
        spy_data['Irre_dby_L_ref'] = dby_L[3, _irr_idx].astype(np.float32)
        spy_data['Irre_dby_R_ref'] = dby_R[3, _irr_idx].astype(np.float32)

        # 对应输出采样点的空间坐标（相对里程）
        spy_data['Irre_distance_m'] = (_save_steps * integration.Vc * integration.Tstep).astype(np.float64)

        settlement_profile = getattr(track_simulator, 'settlement_profile_full', None)
        settlement_distance = getattr(track_simulator, 'settlement_distance_rel_m', None)
        settlement_metadata = getattr(track_simulator, 'settlement_metadata', [])
        if settlement_profile is not None:
            settlement_profile = np.asarray(settlement_profile, dtype=float)
            spy_data['Settlement_profile_m'] = settlement_profile[_irr_idx].astype(np.float32)
            spy_data['Settlement_profile_mm'] = (settlement_profile[_irr_idx] * 1000.0).astype(np.float32)
        else:
            spy_data['Settlement_profile_m'] = np.zeros_like(_irr_idx, dtype=np.float32)
            spy_data['Settlement_profile_mm'] = np.zeros_like(_irr_idx, dtype=np.float32)
        if settlement_distance is not None:
            settlement_distance = np.asarray(settlement_distance, dtype=float)
            spy_data['Settlement_distance_rel_m'] = settlement_distance[_irr_idx].astype(np.float64)
        else:
            spy_data['Settlement_distance_rel_m'] = spy_data['Irre_distance_m']
        spy_data['Settlement_metadata_json'] = np.array(json.dumps(settlement_metadata, ensure_ascii=False))
        spy_data['Settlement_enabled'] = np.array(bool(settlement_metadata))

        geometry_profile = getattr(track_simulator, 'geometry_defect_profile_full', None)
        geometry_distance = getattr(track_simulator, 'geometry_defect_distance_rel_m', None)
        geometry_metadata = getattr(track_simulator, 'geometry_defect_metadata', [])
        if geometry_profile is not None:
            _geo_len = len(np.asarray(geometry_profile.get('VL', [])))
            _geo_idx = np.clip(_irr_idx, 0, max(_geo_len - 1, 0))
            for _key in ('VL', 'VR', 'LL', 'LR'):
                _arr = np.asarray(geometry_profile.get(_key, np.zeros(_geo_len)), dtype=float)
                if _arr.size:
                    spy_data[f'Geometry_defect_{_key}_m'] = _arr[_geo_idx].astype(np.float32)
                    spy_data[f'Geometry_defect_{_key}_mm'] = (_arr[_geo_idx] * 1000.0).astype(np.float32)
                else:
                    spy_data[f'Geometry_defect_{_key}_m'] = np.zeros_like(_irr_idx, dtype=np.float32)
                    spy_data[f'Geometry_defect_{_key}_mm'] = np.zeros_like(_irr_idx, dtype=np.float32)
        else:
            for _key in ('VL', 'VR', 'LL', 'LR'):
                spy_data[f'Geometry_defect_{_key}_m'] = np.zeros_like(_irr_idx, dtype=np.float32)
                spy_data[f'Geometry_defect_{_key}_mm'] = np.zeros_like(_irr_idx, dtype=np.float32)
        if geometry_distance is not None:
            geometry_distance = np.asarray(geometry_distance, dtype=float)
            _dist_idx = np.clip(_irr_idx, 0, max(geometry_distance.size - 1, 0))
            spy_data['Geometry_defect_distance_rel_m'] = geometry_distance[_dist_idx].astype(np.float64)
        else:
            spy_data['Geometry_defect_distance_rel_m'] = spy_data['Irre_distance_m']
        spy_data['Geometry_defect_metadata_json'] = np.array(json.dumps(geometry_metadata, ensure_ascii=False))
        spy_data['Geometry_defect_enabled'] = np.array(bool(geometry_metadata))
    except Exception as e:
        print(f" -> [警告] 不平顺附加数据写入失败: {e}")

    # 保存平纵断面（直接使用已预计算的 track_geometry，以 4位轮对(基准)为参考轨迹）
    try:
        sim_s_abs = track_geometry['S'][_save_steps, 3]      # 4位轮对绝对里程 (Nout,)
        k_profile = track_geometry['K'][_save_steps, 3]      # 4位轮对曲率
        h_profile = track_geometry['H'][_save_steps, 3]      # 4位轮对超高角
        g_profile = track_geometry['G'][_save_steps, 3]      # 4位轮对坡度

        ds = integration.Vc * _dt_output
        z_profile = np.cumsum(g_profile) * ds

        spy_data['Track_abs_mileage_m']   = sim_s_abs
        spy_data['Track_rel_mileage_m']   = sim_s_abs - sim_s_abs[0]
        spy_data['Track_curvature_1pm']   = k_profile.astype(np.float32)
        spy_data['Track_cant_m']          = h_profile.astype(np.float32)
        spy_data['Track_gradient']        = g_profile.astype(np.float32)
        spy_data['Track_vertical_profile_m'] = z_profile.astype(np.float32)
        try:
            _stiffness_field = _structure_defects.ballast_stiffness_field(sim_s_abs)
            spy_data['Stiffness_eta_k_L_ref'] = _stiffness_field['eta_k_L'].astype(np.float32)
            spy_data['Stiffness_eta_k_R_ref'] = _stiffness_field['eta_k_R'].astype(np.float32)
            spy_data['Stiffness_irregularity_mask_L'] = _stiffness_field['mask_L'].astype(np.int8)
            spy_data['Stiffness_irregularity_mask_R'] = _stiffness_field['mask_R'].astype(np.int8)
        except Exception as e:
            print(f" -> [warning] failed to write stiffness irregularity labels: {e}")
            spy_data['Stiffness_eta_k_L_ref'] = np.ones_like(sim_s_abs, dtype=np.float32)
            spy_data['Stiffness_eta_k_R_ref'] = np.ones_like(sim_s_abs, dtype=np.float32)
            spy_data['Stiffness_irregularity_mask_L'] = np.zeros_like(sim_s_abs, dtype=np.int8)
            spy_data['Stiffness_irregularity_mask_R'] = np.zeros_like(sim_s_abs, dtype=np.int8)
        # 附加保存全部 4 个轮对的曲率/超高矩阵，便于后处理对比
        spy_data['Track_K_all_ws'] = track_geometry['K'][_save_steps, :4].astype(np.float32)   # (Nout, 4) 仅轮对位置
        spy_data['Track_H_all_ws'] = track_geometry['H'][_save_steps, :4].astype(np.float32)   # (Nout, 4) 仅轮对位置

        sw = track_geometry.get('structure_window', {})
        if sw:
            spy_data['Structure_window_start_abs_m'] = sw['window_start_abs_m'][_save_steps]
            spy_data['Structure_window_end_abs_m'] = sw['window_end_abs_m'][_save_steps]
            spy_data['Structure_local_s_m'] = sw['local_s_m'][_save_steps, :].astype(np.float32)
            spy_data['Structure_wheel_in_window'] = sw['wheel_in_window'][_save_steps, :].astype(np.int8)
            spy_data['Structure_boundary_weight'] = sw['boundary_weight'][_save_steps, :].astype(np.float32)
            spy_data['Structure_active_node_start'] = sw['active_node_start'][_save_steps]
            spy_data['Structure_active_node_end'] = sw['active_node_end'][_save_steps]
            spy_data['Structure_window_shift_nodes'] = sw['window_shift_nodes'][_save_steps]
            spy_data['Structure_window_meta'] = np.array([
                sw['window_length_m'],
                sw['window_lead_m'],
                sw['window_trail_m'],
                sw['model_length_m'],
                sw['boundary_buffer_m'],
                float(integration.Nsub),
                float(integration.Lkj),
                1.0 if sw.get('state_migration', False) else 0.0,
            ], dtype=np.float64)

        if _modal_freqs is not None:
            fv, fl, ft = _modal_freqs
            spy_data['Rail_modal_freqs_vertical_hz'] = fv
            spy_data['Rail_modal_freqs_lateral_hz'] = fl
            spy_data['Rail_modal_freqs_torsion_hz'] = ft
        spy_data['Rail_modal_counts'] = np.array([mode_params.NV, mode_params.NL, mode_params.NT], dtype=np.int32)
    except Exception as e:
        print(f" -> [警告] 平纵断面/结构截断附加数据写入失败: {e}")

    lead_time_s = max(0.0, float(args.irr_lead_time)) if _need_irre_lead else 0.0
    valid_start_index = min(int(round(lead_time_s / _dt_output)), max(A.shape[0] - 1, 0))
    spy_data['valid_start_index'] = np.array(valid_start_index, dtype=np.int64)
    spy_data['valid_start_mileage_m'] = np.array(start_mileage_m, dtype=float)
    spy_data['lead_time_s'] = np.array(lead_time_s, dtype=float)
    spy_data['effective_duration_s'] = np.array(float(args.tz), dtype=float)
    spy_data['output_tier'] = np.array(output_tier)
    spy_data['save_format'] = np.array(save_format)
    spy_data['requested_save_dof_mode'] = np.array(requested_save_dof_mode)
    spy_data['actual_save_dof_mode'] = np.array(actual_save_dof_mode)

    print("\n===================================================================")
    print("               仿真计算完成！数据已缓存至内存。               ")
    print("===================================================================")

    #=========================Part 5 结果保存与可视化=========================#
    # 1. 自动保存数据至 results/<project_name>/<run_name>/files/
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _note = str(args.run_note).strip()
    _note = _note if _note else 'standard'
    run_name = f"{args.vehicle_type}-{args.irr_type}-{actual_save_dof_mode}-{output_tier}-{_note}-{timestamp}"
    project_results_root = os.path.join('results', ResultPlotter._sanitize_name(args.project_name))
    save_start_time = time.perf_counter()
    compressed_override = True if _str_to_bool(args.save_compressed) else None
    saved_npz_path = ResultPlotter.save_data(
        run_name=run_name,
        X=X,
        V=V,
        A=A,
        spy_dict=spy_data,
        dt=_dt_output,
        idx_car_start=topo.idx_Car[0],
        idx_car_end=topo.idx_Car[1],
        save_dof_mode=actual_save_dof_mode,
        results_root=project_results_root,
        compressed=compressed_override,
        output_tier=output_tier,
        save_format=save_format,
    )
    save_elapsed_s = time.perf_counter() - save_start_time
    result_size_bytes = _path_size_bytes(saved_npz_path)

    # 2. 在同一目录保存 argparse 参数字典
    files_dir = os.path.dirname(saved_npz_path)
    defect_summary_file = ''
    defect_config_archive = ''
    try:
        defect_summary_file = os.path.join(files_dir, 'structure_defects_summary.json')
        with open(defect_summary_file, 'w', encoding='utf-8') as f:
            json.dump(_structure_defect_summary, f, ensure_ascii=False, indent=2)
        if args.structure_defect_config and os.path.exists(args.structure_defect_config):
            _cfg_ext = os.path.splitext(args.structure_defect_config)[1] or '.txt'
            defect_config_archive = os.path.join(files_dir, f'structure_defect_config{_cfg_ext}')
            shutil.copy2(args.structure_defect_config, defect_config_archive)
        if args.geometry_defect_config and os.path.exists(args.geometry_defect_config):
            _geo_ext = os.path.splitext(args.geometry_defect_config)[1] or '.txt'
            geometry_config_archive = os.path.join(files_dir, f'geometry_defect_config{_geo_ext}')
            shutil.copy2(args.geometry_defect_config, geometry_config_archive)
        print(f" -> [结构缺陷] 缺陷摘要已保存至: {defect_summary_file}")
    except Exception as e:
        print(f" -> [警告] 结构缺陷归档失败: {e}")

    try:
        args_dict = vars(args)
        args_file = os.path.join(files_dir, 'argparse_params.json')
        with open(args_file, 'w', encoding='utf-8') as f:
            json.dump(args_dict, f, ensure_ascii=False, indent=2)
        print(f" -> [参数归档] argparse 参数已保存至: {args_file}")
    except Exception as e:
        print(f" -> [警告] argparse 参数保存失败: {e}")

    # 3. 保存 run_meta.yaml（参数目录、试验描述、覆盖信息等）
    try:
        files_dir = os.path.dirname(saved_npz_path)
        run_meta_file = os.path.join(files_dir, 'run_meta.yaml')

        profile_abs = os.path.abspath(args.param_profile_dir)
        standard_abs = os.path.abspath(os.path.join('configs', 'standard'))
        yaml_names = [
            'vehicle_params.yaml',
            'rail_params.yaml',
            'fastener_kv.yaml',
            'subrail_params.yaml',
            'fastener_fdkv_params.yaml',
            'extra_force_elements.yaml',
            'antiyawer_params.yaml',
        ]

        with open(run_meta_file, 'w', encoding='utf-8') as f:
            f.write(f"run_name: {run_name}\n")
            f.write(f"timestamp: {timestamp}\n")
            f.write(f"vehicle_type: {args.vehicle_type}\n")
            f.write(f"irr_type: {args.irr_type}\n")
            f.write(f"save_dof_mode: {actual_save_dof_mode}\n")
            f.write(f"requested_save_dof_mode: {requested_save_dof_mode}\n")
            f.write(f"output_tier: {output_tier}\n")
            f.write(f"save_format: {save_format}\n")
            f.write(f"save_stride: {int(args.save_stride)}\n")
            f.write(f"save_spy_level: {args.save_spy_level}\n")
            f.write(f"save_compressed: {args.save_compressed}\n")
            f.write(f"save_elapsed_s: {save_elapsed_s:.6f}\n")
            f.write(f"result_size_bytes: {int(result_size_bytes)}\n")
            f.write(f"dt_integrator_s: {float(integration.Tstep):.9f}\n")
            f.write(f"dt_output_s: {float(_dt_output):.9f}\n")
            f.write(f"output_steps: {int(A.shape[0])}\n")
            f.write(f"project_name: {args.project_name}\n")
            f.write(f"project_results_root: {os.path.abspath(project_results_root)}\n")
            f.write(f"run_note: {str(args.run_note).strip() or 'standard'}\n")
            f.write(f"param_profile_dir: {args.param_profile_dir}\n")
            f.write(f"param_profile_dir_abs: {profile_abs}\n")
            f.write(f"standard_profile_dir_abs: {standard_abs}\n")
            f.write(f"tz_input_s: {float(args.tz):.6f}\n")
            f.write(f"irr_lead_time_s: {float(args.irr_lead_time):.6f}\n")
            f.write(f"tz_effective_s: {float(_tz_effective):.6f}\n")
            f.write(f"tz_actual_s: {float(integration.Tz):.6f}\n")
            f.write(f"auto_extend_rail: {args.auto_extend_rail}\n")
            f.write(f"n_sub_input: {int(args.N_sub)}\n")
            f.write(f"n_sub_runtime: {int(integration.Nsub)}\n")
            f.write(f"rail_model_length_m: {float(integration.Ljs):.6f}\n")
            f.write(f"structure_truncation_mode: {args.structure_truncation_mode}\n")
            f.write(f"structure_window_length_m: {float(_structure_window['window_length_m']):.6f}\n")
            f.write(f"structure_window_lead_m: {float(_structure_window['window_lead_m']):.6f}\n")
            f.write(f"structure_window_trail_m: {float(_structure_window['window_trail_m']):.6f}\n")
            f.write(f"structure_boundary_buffer_m: {float(_structure_window['boundary_buffer_m']):.6f}\n")
            f.write(f"structure_state_migration: {args.structure_state_migration}\n")
            f.write(f"structure_state_migration_active: {bool(_structure_window.get('state_migration', False))}\n")
            f.write(f"structure_defect_switch: {args.structure_defect_switch}\n")
            f.write(f"structure_defect_config: {args.structure_defect_config}\n")
            f.write(f"structure_defect_record_count: {int(_structure_defect_summary.get('record_count', 0))}\n")
            f.write(f"structure_defect_summary: {defect_summary_file}\n")
            f.write(f"structure_defect_config_archive: {defect_config_archive}\n")
            f.write(f"geometry_defect_switch: {args.geometry_defect_switch}\n")
            f.write(f"geometry_defect_config: {args.geometry_defect_config}\n")
            f.write(f"geometry_defect_record_count: {len(geometry_defect_specs)}\n")
            f.write(f"geometry_defect_config_archive: {geometry_config_archive}\n")
            f.write(f"modal_truncation: {args.modal_truncation}\n")
            f.write(f"modal_cutoff_hz: {float(args.modal_cutoff_hz):.6f}\n")
            f.write(f"rail_modal_counts: [{mode_params.NV}, {mode_params.NL}, {mode_params.NT}]\n")
            f.write(f"result_npz: {saved_npz_path}\n")
            f.write(f"argparse_params: {args_file if 'args_file' in locals() else ''}\n")
            f.write("yaml_sources:\n")
            for name in yaml_names:
                p_profile = os.path.join(profile_abs, name)
                p_standard = os.path.join(standard_abs, name)
                profile_exists = os.path.exists(p_profile)
                standard_exists = os.path.exists(p_standard)
                source = 'profile' if profile_exists else ('standard' if standard_exists else 'missing')
                used_path = p_profile if profile_exists else (p_standard if standard_exists else '')
                f.write(f"  {name}:\n")
                f.write(f"    source: {source}\n")
                f.write(f"    used_path: {used_path}\n")

        print(f" -> [参数归档] 运行元数据已保存至: {run_meta_file}")
    except Exception as e:
        print(f" -> [警告] run_meta.yaml 保存失败: {e}")

    try:
        perf_file = os.path.join(files_dir, 'perf_summary.json')
        perf_summary = {
            'output_tier': output_tier,
            'save_format': save_format,
            'requested_save_dof_mode': requested_save_dof_mode,
            'actual_save_dof_mode': actual_save_dof_mode,
            'nt': int(integration.Nt),
            'nsub': int(integration.Nsub),
            'fnum_total': int(topo.Fnum_Total),
            'dt_integrator_s': float(integration.Tstep),
            'dt_output_s': float(_dt_output),
            'tz_effective_s': float(_tz_effective),
            'save_elapsed_s': float(save_elapsed_s),
            'total_elapsed_s': float(time.perf_counter() - main_start_time),
            'result_size_bytes': int(result_size_bytes),
            'result_path': saved_npz_path,
        }
        with open(perf_file, 'w', encoding='utf-8') as f:
            json.dump(perf_summary, f, ensure_ascii=False, indent=2)
        print(f" -> [性能统计] 保存耗时 {save_elapsed_s:.2f}s，结果体积 {result_size_bytes / (1024 ** 2):.2f} MiB")
        print(f" -> [性能统计] perf_summary.json 已保存至: {perf_file}")
    except Exception as e:
        print(f" -> [警告] perf_summary.json 保存失败: {e}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
