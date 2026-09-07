from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

INTERNAL_COMMON_KEYS = {'note_prefix', 'description', 'main_script'}
GEOMETRY_DEFECT_FILE = 'geometry_defects.yaml'
SETTLEMENT_FILE = 'transition_settlements.yaml'


def _tee_pipe(pipe, terminal_stream, log_file):
    try:
        while True:
            chunk = pipe.read(1)
            if not chunk:
                break
            terminal_stream.write(chunk)
            terminal_stream.flush()
            log_file.write(chunk)
            log_file.flush()
    finally:
        pipe.close()


def _worker_environment(threads_per_worker: int) -> dict[str, str]:
    env = os.environ.copy()
    thread_count = str(max(1, int(threads_per_worker)))
    for name in (
        'OMP_NUM_THREADS',
        'MKL_NUM_THREADS',
        'OPENBLAS_NUM_THREADS',
        'NUMEXPR_NUM_THREADS',
    ):
        env[name] = thread_count
    env.setdefault('PYTHONIOENCODING', 'utf-8')
    return env


def run_with_live_logs(
    command: list[str],
    cwd: Path,
    stdout_log: Path,
    stderr_log: Path,
    env: dict[str, str] | None = None,
) -> int:
    with stdout_log.open('w', encoding='utf-8', errors='replace') as out_file, stderr_log.open('w', encoding='utf-8', errors='replace') as err_file:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
            env=env,
        )
        threads = [
            threading.Thread(target=_tee_pipe, args=(process.stdout, sys.stdout, out_file), daemon=True),
            threading.Thread(target=_tee_pipe, args=(process.stderr, sys.stderr, err_file), daemon=True),
        ]
        for thread in threads:
            thread.start()
        return_code = process.wait()
        for thread in threads:
            thread.join()
        return return_code


def run_to_logs(
    command: list[str],
    cwd: Path,
    stdout_log: Path,
    stderr_log: Path,
    env: dict[str, str] | None = None,
) -> int:
    """Run one case without interleaving parallel worker output in the terminal."""
    with stdout_log.open('w', encoding='utf-8', errors='replace') as out_file, stderr_log.open('w', encoding='utf-8', errors='replace') as err_file:
        completed = subprocess.run(
            command,
            cwd=cwd,
            stdout=out_file,
            stderr=err_file,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env,
        )
    return completed.returncode


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f'未找到文件: {path}')
    with path.open('r', encoding='utf-8') as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f'YAML 顶层必须为字典: {path}')
    return data


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)


def normalize_cases(cases: list[dict[str, Any]], include: list[str] | None, exclude: list[str] | None) -> list[dict[str, Any]]:
    include_set = set(include or [])
    exclude_set = set(exclude or [])
    filtered: list[dict[str, Any]] = []
    for case in cases:
        case_id = case.get('case_id')
        if not case_id:
            raise ValueError('manifest 中每个 case 都必须包含 case_id')
        if include_set and case_id not in include_set:
            continue
        if case_id in exclude_set:
            continue
        filtered.append(case)
    return filtered


def append_cli_arg(command: list[str], key: str, value: Any, source: str = 'common') -> None:
    if value is None:
        return
    if isinstance(value, bool):
        value = 'On' if value else 'Off'
    if isinstance(value, (dict, list, tuple)):
        raise ValueError(f'{source} 参数暂不支持复合类型: {key}')
    command.extend([f'--{key}', str(value)])


def build_case_command(
    python_exe: str,
    workspace_root: Path,
    common: dict[str, Any],
    profile_dir: Path,
    case_id: str,
    manifest_name: str,
    case_args: dict[str, Any] | None,
    extra_args: list[str],
) -> list[str]:
    main_script = str(common.get('main_script', 'generate_main.py'))
    command = [python_exe, str(workspace_root / main_script)]

    for key, value in common.items():
        if key in INTERNAL_COMMON_KEYS:
            continue
        append_cli_arg(command, key, value, source='common')

    for key, value in (case_args or {}).items():
        append_cli_arg(command, key, value, source='case_args')

    note_prefix = str(common.get('note_prefix', 'sweep'))
    run_note = f'{note_prefix}_{case_id}'
    if 'project_name' not in common:
        command.extend(['--project_name', manifest_name])
    command.extend(['--param_profile_dir', str(profile_dir)])
    command.extend(['--run_note', run_note])
    structure_defect_config = profile_dir / 'structure_defects.yaml'
    if structure_defect_config.exists():
        command.extend([
            '--structure_defect_switch', 'On',
            '--structure_defect_config', str(structure_defect_config),
        ])
    geometry_defect_config = profile_dir / GEOMETRY_DEFECT_FILE
    if geometry_defect_config.exists():
        command.extend([
            '--geometry_defect_switch', 'On',
            '--geometry_defect_config', str(geometry_defect_config),
        ])
    settlement_config = profile_dir / SETTLEMENT_FILE
    if settlement_config.exists():
        command.extend([
            '--settlement_switch', 'On',
            '--settlement_config', str(settlement_config),
        ])
    command.extend(extra_args)
    return command


def find_case_result(workspace_root: Path, project_name: str, run_note: str, started_at: float) -> Path:
    project_root = workspace_root / 'results' / project_name
    candidates = []
    if project_root.exists():
        for result_file in project_root.glob(f'*{run_note}-*/files/simulation_result.npz'):
            try:
                if result_file.stat().st_mtime >= started_at - 2.0:
                    candidates.append(result_file)
            except OSError:
                continue
    if not candidates:
        raise FileNotFoundError(f'No result file found for run_note={run_note} under {project_root}')
    return max(candidates, key=lambda path: path.stat().st_mtime)


def has_completed_case_result(workspace_root: Path, project_name: str, run_note: str) -> bool:
    """Return whether a prior run for this exact case has a result artifact."""
    project_root = workspace_root / 'results' / project_name
    if not project_root.exists():
        return False
    return any(project_root.glob(f'*{run_note}-*/files/simulation_result.npz'))


def run_extended_response_overview(python_exe: str, workspace_root: Path, result_file: Path, log_file: Path) -> int:
    analysis_code = (
        "from analyze_results import run_all_analyses; "
        f"run_all_analyses(target_file={str(result_file)!r}, "
        "run_core=True, run_irre=False, run_accel=False, run_psd=False, show=False)"
    )
    command = [python_exe, '-c', analysis_code]
    with log_file.open('w', encoding='utf-8', errors='replace') as output:
        completed = subprocess.run(
            command, cwd=workspace_root, stdout=output, stderr=subprocess.STDOUT, text=True
        )
    figures_dir = result_file.parents[1] / 'figures'
    expected = [
        figures_dir / 'extended_response_overview.png',
        figures_dir / 'extended_response_overview.svg',
        figures_dir / 'component_stiffness_overview.png',
        figures_dir / 'component_stiffness_overview.svg',
    ]
    if completed.returncode == 0 and not all(path.exists() for path in expected):
        return 2
    return completed.returncode


def execute_case(
    *,
    case_index: int,
    case: dict[str, Any],
    python_exe: str,
    workspace_root: Path,
    output_root: Path,
    logs_dir: Path,
    common: dict[str, Any],
    manifest_name: str,
    extra_args: list[str],
    analyze_after_case: bool,
    dry_run: bool,
    stream_logs: bool,
    threads_per_worker: int,
) -> dict[str, Any]:
    case_id = str(case['case_id'])
    profile_dir = output_root / case_id
    if not profile_dir.exists():
        raise FileNotFoundError(f'未找到 case 参数目录: {profile_dir}，可先加 --build-first')

    command = build_case_command(
        python_exe=python_exe,
        workspace_root=workspace_root,
        common=common,
        profile_dir=profile_dir,
        case_id=case_id,
        manifest_name=manifest_name,
        case_args=case.get('case_args', {}) or {},
        extra_args=extra_args,
    )
    case_started_wall = time.time()
    case_started = time.perf_counter()
    status = 'dry-run'
    return_code = None
    stdout_log = None
    stderr_log = None
    analysis_log = None

    if not dry_run:
        stdout_log = logs_dir / f'{case_id}.out.log'
        stderr_log = logs_dir / f'{case_id}.err.log'
        env = _worker_environment(threads_per_worker)
        runner = run_with_live_logs if stream_logs else run_to_logs
        return_code = runner(command, workspace_root, stdout_log, stderr_log, env=env)
        status = 'success' if return_code == 0 else 'failed'
        if status == 'success' and analyze_after_case:
            project_name = str(common.get('project_name', manifest_name))
            run_note = f"{common.get('note_prefix', 'sweep')}_{case_id}"
            result_file = find_case_result(workspace_root, project_name, run_note, case_started_wall)
            analysis_log = logs_dir / f'{case_id}.analysis.log'
            analysis_return_code = run_extended_response_overview(
                python_exe, workspace_root, result_file, analysis_log
            )
            if analysis_return_code != 0:
                return_code = analysis_return_code
                status = 'analysis_failed'

    return {
        'case_index': case_index,
        'case_id': case_id,
        'profile_dir': str(profile_dir),
        'command': command,
        'status': status,
        'return_code': return_code,
        'elapsed_s': round(time.perf_counter() - case_started, 3),
        'stdout_log': str(stdout_log) if stdout_log is not None else None,
        'stderr_log': str(stderr_log) if stderr_log is not None else None,
        'analysis_log': str(analysis_log) if analysis_log is not None else None,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='按 sweep manifest 批量运行 generate_main.py')
    parser.add_argument('--manifest', required=True, help='扫描清单 YAML 路径')
    parser.add_argument('--python-exe', default=sys.executable, help='运行 generate_main.py 的 Python 可执行文件')
    parser.add_argument('--build-first', action='store_true', help='运行前先调用 build_param_sweep.py 生成/更新 trial 参数目录')
    parser.add_argument('--dry-run', action='store_true', help='只打印将执行的命令，不实际运行')
    parser.add_argument('--cases', nargs='*', help='只运行指定 case_id 列表')
    parser.add_argument('--skip-cases', nargs='*', help='跳过指定 case_id 列表')
    parser.add_argument(
        '--skip-completed',
        action='store_true',
        help='扫描 results/<project_name>，自动跳过已有 simulation_result.npz 的 case',
    )
    parser.add_argument('--analyze-after-case', action='store_true', help='Generate extended_response_overview after each successful case')
    parser.add_argument('--stop-on-error', action='store_true', help='遇到首个失败 case 即停止')
    parser.add_argument('--workers', type=int, default=1, help='同时运行的仿真进程数，默认 1')
    parser.add_argument('--threads-per-worker', type=int, default=1, help='每个仿真进程允许使用的数值库线程数，默认 1')
    parser.add_argument('--extra-args', nargs=argparse.REMAINDER, default=[], help='透传给 generate_main.py 的额外参数（放在命令最后）')
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.workers < 1:
        raise ValueError('--workers 必须大于等于 1')
    if args.threads_per_worker < 1:
        raise ValueError('--threads-per-worker 必须大于等于 1')
    manifest_path = Path(args.manifest).resolve()
    workspace_root = manifest_path.parent.parent.parent
    manifest = load_yaml(manifest_path)
    manifest_name = str(manifest.get('manifest_name', manifest_path.stem))

    output_root = workspace_root / manifest.get('output_root', 'configs/trials/generated')
    common = manifest.get('common', {})
    if not isinstance(common, dict):
        raise ValueError('manifest.common 必须为字典')
    cases = manifest.get('cases', [])
    if not isinstance(cases, list) or not cases:
        raise ValueError('manifest 中 cases 必须是非空列表')

    selected_cases = normalize_cases(cases, args.cases, args.skip_cases)
    if args.skip_completed:
        project_name = str(common.get('project_name', manifest_name))
        note_prefix = str(common.get('note_prefix', 'sweep'))
        before_count = len(selected_cases)
        selected_cases = [
            case for case in selected_cases
            if not has_completed_case_result(
                workspace_root,
                project_name,
                f"{note_prefix}_{case['case_id']}",
            )
        ]
        print(
            f'已扫描 results/{project_name}：跳过 {before_count - len(selected_cases)} 个已完成 case，'
            f'剩余 {len(selected_cases)} 个。'
        )
    if not selected_cases:
        if args.skip_completed:
            print('所有选定 case 均已有结果，无需继续生成。')
            return
        raise ValueError('筛选后没有可运行的 case')

    if args.build_first:
        build_command = [
            args.python_exe,
            str(workspace_root / 'utils' / 'build_param_sweep.py'),
            '--manifest',
            str(manifest_path),
        ]
        print('[build] ' + subprocess.list2cmdline(build_command))
        if not args.dry_run:
            subprocess.run(build_command, cwd=workspace_root, check=True)

    run_started_at = datetime.now()
    log_entries: list[dict[str, Any]] = []
    logs_dir = output_root / 'logs'
    if not args.dry_run:
        logs_dir.mkdir(parents=True, exist_ok=True)

    print(
        f'将运行 {len(selected_cases)} 组 case，workers={args.workers}，'
        f'threads_per_worker={args.threads_per_worker}，输出目录根路径: {output_root}'
    )

    def report(entry: dict[str, Any]) -> None:
        completed = len(log_entries)
        print(
            f"[{completed}/{len(selected_cases)}] {entry['case_id']} -> "
            f"{entry['status']}, elapsed={entry['elapsed_s']:.3f}s"
        )
        if entry['stdout_log']:
            print(f"     stdout: {entry['stdout_log']}")
        if entry['stderr_log']:
            print(f"     stderr: {entry['stderr_log']}")

    common_kwargs = {
        'python_exe': args.python_exe,
        'workspace_root': workspace_root,
        'output_root': output_root,
        'logs_dir': logs_dir,
        'common': common,
        'manifest_name': manifest_name,
        'extra_args': args.extra_args,
        'analyze_after_case': args.analyze_after_case,
        'dry_run': args.dry_run,
        'threads_per_worker': args.threads_per_worker,
    }

    if args.workers == 1:
        for index, case in enumerate(selected_cases, start=1):
            print(f"[启动 {index}/{len(selected_cases)}] {case['case_id']}")
            entry = execute_case(
                case_index=index,
                case=case,
                stream_logs=True,
                **common_kwargs,
            )
            log_entries.append(entry)
            report(entry)
            if entry['status'] not in {'success', 'dry-run'} and args.stop_on_error:
                break
    else:
        next_case = iter(enumerate(selected_cases, start=1))
        stop_scheduling = False
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            pending = {}

            def submit_next() -> bool:
                try:
                    index, case = next(next_case)
                except StopIteration:
                    return False
                print(f"[启动 {index}/{len(selected_cases)}] {case['case_id']}")
                future = executor.submit(
                    execute_case,
                    case_index=index,
                    case=case,
                    stream_logs=False,
                    **common_kwargs,
                )
                pending[future] = (index, str(case['case_id']))
                return True

            for _ in range(min(args.workers, len(selected_cases))):
                submit_next()

            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    index, case_id = pending.pop(future)
                    try:
                        entry = future.result()
                    except Exception as exc:
                        entry = {
                            'case_index': index,
                            'case_id': case_id,
                            'profile_dir': None,
                            'command': None,
                            'status': 'controller_failed',
                            'return_code': None,
                            'elapsed_s': 0.0,
                            'stdout_log': None,
                            'stderr_log': None,
                            'analysis_log': None,
                            'error': repr(exc),
                        }
                    log_entries.append(entry)
                    report(entry)
                    if entry['status'] not in {'success', 'dry-run'} and args.stop_on_error:
                        stop_scheduling = True
                while not stop_scheduling and len(pending) < args.workers and submit_next():
                    pass

    log_entries.sort(key=lambda item: int(item.get('case_index', 0)))

    summary = {
        'manifest': str(manifest_path),
        'output_root': str(output_root),
        'project_name': common.get('project_name', manifest_name),
        'run_started_at': run_started_at.isoformat(timespec='seconds'),
        'run_finished_at': datetime.now().isoformat(timespec='seconds'),
        'dry_run': args.dry_run,
        'build_first': args.build_first,
        'analyze_after_case': args.analyze_after_case,
        'workers': args.workers,
        'threads_per_worker': args.threads_per_worker,
        'total_cases': len(selected_cases),
        'success_cases': sum(1 for item in log_entries if item['status'] == 'success'),
        'failed_cases': sum(1 for item in log_entries if item['status'] not in {'success', 'dry-run'}),
        'entries': log_entries,
    }
    log_name = f"sweep_run_{run_started_at.strftime('%Y%m%d_%H%M%S')}.yaml"
    log_path = output_root / log_name
    dump_yaml(log_path, summary)
    print(f'\n批量运行日志已写入: {log_path}')


if __name__ == '__main__':
    main()
