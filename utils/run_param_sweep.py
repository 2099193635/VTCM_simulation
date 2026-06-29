from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

INTERNAL_COMMON_KEYS = {'note_prefix', 'description', 'main_script'}
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


def run_with_live_logs(command: list[str], cwd: Path, stdout_log: Path, stderr_log: Path) -> int:
    with stdout_log.open('w', encoding='utf-8', errors='replace') as out_file, stderr_log.open('w', encoding='utf-8', errors='replace') as err_file:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors='replace',
            bufsize=1,
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
    settlement_config = profile_dir / SETTLEMENT_FILE
    if settlement_config.exists():
        command.extend([
            '--settlement_switch', 'On',
            '--settlement_config', str(settlement_config),
        ])
    command.extend(extra_args)
    return command


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='按 sweep manifest 批量运行 generate_main.py')
    parser.add_argument('--manifest', required=True, help='扫描清单 YAML 路径')
    parser.add_argument('--python-exe', default=sys.executable, help='运行 generate_main.py 的 Python 可执行文件')
    parser.add_argument('--build-first', action='store_true', help='运行前先调用 build_param_sweep.py 生成/更新 trial 参数目录')
    parser.add_argument('--dry-run', action='store_true', help='只打印将执行的命令，不实际运行')
    parser.add_argument('--cases', nargs='*', help='只运行指定 case_id 列表')
    parser.add_argument('--skip-cases', nargs='*', help='跳过指定 case_id 列表')
    parser.add_argument('--stop-on-error', action='store_true', help='遇到首个失败 case 即停止')
    parser.add_argument('--extra-args', nargs=argparse.REMAINDER, default=[], help='透传给 generate_main.py 的额外参数（放在命令最后）')
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
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
    if not selected_cases:
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

    print(f'将运行 {len(selected_cases)} 组 case，输出目录根路径: {output_root}')
    for index, case in enumerate(selected_cases, start=1):
        case_id = str(case['case_id'])
        profile_dir = output_root / case_id
        if not profile_dir.exists():
            raise FileNotFoundError(f'未找到 case 参数目录: {profile_dir}，可先加 --build-first')

        command = build_case_command(
            python_exe=args.python_exe,
            workspace_root=workspace_root,
            common=common,
            profile_dir=profile_dir,
            case_id=case_id,
            manifest_name=manifest_name,
            case_args=case.get('case_args', {}) or {},
            extra_args=args.extra_args,
        )
        command_str = subprocess.list2cmdline(command)
        print(f'[{index}/{len(selected_cases)}] {case_id}')
        print('  ' + command_str)

        case_started = time.perf_counter()
        status = 'dry-run'
        return_code = None
        stdout_log = None
        stderr_log = None
        if not args.dry_run:
            stdout_log = logs_dir / f'{case_id}.out.log'
            stderr_log = logs_dir / f'{case_id}.err.log'
            return_code = run_with_live_logs(command, workspace_root, stdout_log, stderr_log)
            status = 'success' if return_code == 0 else 'failed'
        elapsed = round(time.perf_counter() - case_started, 3)
        print(f'  -> {status}, elapsed={elapsed:.3f}s')
        if stdout_log is not None:
            print(f'     stdout: {stdout_log}')
        if stderr_log is not None:
            print(f'     stderr: {stderr_log}')

        log_entries.append({
            'case_id': case_id,
            'profile_dir': str(profile_dir),
            'command': command,
            'status': status,
            'return_code': return_code,
            'elapsed_s': elapsed,
            'stdout_log': str(stdout_log) if stdout_log is not None else None,
            'stderr_log': str(stderr_log) if stderr_log is not None else None,
        })

        if status == 'failed' and args.stop_on_error:
            break

    summary = {
        'manifest': str(manifest_path),
        'output_root': str(output_root),
        'project_name': common.get('project_name', manifest_name),
        'run_started_at': run_started_at.isoformat(timespec='seconds'),
        'run_finished_at': datetime.now().isoformat(timespec='seconds'),
        'dry_run': args.dry_run,
        'build_first': args.build_first,
        'total_cases': len(selected_cases),
        'success_cases': sum(1 for item in log_entries if item['status'] == 'success'),
        'failed_cases': sum(1 for item in log_entries if item['status'] == 'failed'),
        'entries': log_entries,
    }
    log_name = f"sweep_run_{run_started_at.strftime('%Y%m%d_%H%M%S')}.yaml"
    log_path = output_root / log_name
    dump_yaml(log_path, summary)
    print(f'\n批量运行日志已写入: {log_path}')


if __name__ == '__main__':
    main()
