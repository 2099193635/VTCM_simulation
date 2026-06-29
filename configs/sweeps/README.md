# 参数扫描 YAML 方案

推荐把“参数扫描清单”写成**显式试验列表**，而不是把 3 档参数分散在多个 YAML 里。

优点：
- 每组试验改了什么一眼可见
- 适合 10~20 组工程化对比
- 与 `configs/standard` 的基准参数解耦
- 可自动展开成 `configs/trials/<sweep_name>/<case_id>/` 目录供主程序直接读取

## 推荐结构

- `configs/standard/`：标准参数基准
- `configs/sweeps/<sweep_name>.yaml`：参数扫描清单
- `configs/trials/<sweep_name>/<case_id>/`：由扫描清单自动生成的试验参数目录

## Sweep YAML 设计原则

1. `base_profile_dir` 指向基准参数目录
2. `output_root` 指向生成后的试验目录根目录
3. `cases` 中每一项就是一组试验
4. 每组试验用 `updates` 显式写出修改项
5. 只写改动参数，未写部分自动回退到 `configs/standard`

## 运行流程

1. 编写 sweep YAML
2. 运行生成脚本：
   `python utils/build_param_sweep.py --manifest configs/sweeps/high_speed_passenger_kpz_cpz_scan.yaml`
3. 主程序按某个 case 运行：
   `python generate_main.py --param_profile_dir configs/trials/high_speed_passenger_kpz_cpz_scan/case_01_kpz045_cpz12k --run_note case_01_kpz045_cpz12k --save_dof_mode vehicle`
4. 如需批量运行整个 sweep：
    `python utils/run_param_sweep.py --manifest configs/sweeps/high_speed_passenger_kpz_cpz_scan.yaml --build-first`

## 批量运行脚本说明

- `utils/run_param_sweep.py` 会读取 sweep YAML 中的 `common`，自动拼出 `generate_main.py` 命令
- 默认按 `cases` 顺序逐组运行，并把日志写到 `output_root/sweep_run_时间戳.yaml`
- 建议先用 dry-run 检查命令：
   `python utils/run_param_sweep.py --manifest configs/sweeps/high_speed_passenger_kpz_cpz_scan.yaml --dry-run`
- 只跑部分 case：
   `python utils/run_param_sweep.py --manifest configs/sweeps/high_speed_passenger_kpz_cpz_scan.yaml --cases case_01_kpz045_cpz06k case_02_kpz045_cpz12k`
- 遇错即停：
   `python utils/run_param_sweep.py --manifest configs/sweeps/high_speed_passenger_kpz_cpz_scan.yaml --stop-on-error`
- 需要附加额外主程序参数时，可用：
   `python utils/run_param_sweep.py --manifest ... --extra-args --tz 7 --plot_figs Off`

## 一、二级悬挂联合扫描

当前一、二级悬挂刚度/阻尼联合扫描清单：

`configs/sweeps/high_speed_passenger_primary_secondary_suspension_scan.yaml`

该清单围绕高速客车标准参数扫描：
- 一系垂向：`Kpz_base`, `Cpz_base`
- 二系横向/垂向：`Ksy_base`, `Ksz_base`, `Csy_base`, `Csz_base`

一键生成参数目录、运行各 case、并绘制单工况分析图与多工况叠加对比图：

`python run_sweep_and_compare.py`

仅检查命令，不启动仿真：

`python run_sweep_and_compare.py --dry-run`

只运行/比较部分 case：

`python run_sweep_and_compare.py --cases case_01_baseline case_02_primary_soft`

已有结果时只做对比绘图：

`python run_sweep_and_compare.py --compare-only`

## 第二轮：构架 PSD 校准扫描

第二轮扫描清单：
`configs/sweeps/high_speed_passenger_bogie_psd_round2_scan.yaml`

该清单保持二系悬挂参数不变，只改变：
- 一系横向：`Kpy_base`, `Cpy_base`
- 一系纵向：`Kpx_base`, `Cpx_base`
- 一系垂向：`Kpz_base`, `Cpz_base`
- 抗侧滚刚度：`Krx`

运行全部第二轮工况并生成构架 PSD 拟合排序：

`python run_sweep_and_compare.py --manifest configs/sweeps/high_speed_passenger_bogie_psd_round2_scan.yaml --skip-single-analysis`

只检查命令，不实际运行：

`python run_sweep_and_compare.py --manifest configs/sweeps/high_speed_passenger_bogie_psd_round2_scan.yaml --dry-run`

已有仿真结果时只重新比较：

`python run_sweep_and_compare.py --manifest configs/sweeps/high_speed_passenger_bogie_psd_round2_scan.yaml --compare-only --skip-single-analysis`

## 为什么比“直接改一个试验 YAML”更清楚

因为扫描问题的本质是：
- 一次要管理很多组参数
- 需要回溯“哪一组改了什么”
- 需要保证每组都从同一份 standard 出发

所以最清晰的方式不是“一个试验目录手改很多次”，而是：
- `standard` 固定
- `sweep 清单` 固定记录所有试验
- `trials` 自动生成
