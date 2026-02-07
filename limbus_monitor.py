"""Steam launch and health monitor for Limbus Company.

Requires: pip install psutil
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import psutil


class ErrorCodes:
    """Error code constants for failure reporting."""

    E_STEAM_NOT_RUNNING = "E_STEAM_NOT_RUNNING"
    E_GAME_START_TIMEOUT = "E_GAME_START_TIMEOUT"
    E_OPEN_PROCESS_HANDLE_FAILED = "E_OPEN_PROCESS_HANDLE_FAILED"
    E_WINDOW_NOT_FOUND = "E_WINDOW_NOT_FOUND"
    E_WINDOW_NOT_RESPONSIVE = "E_WINDOW_NOT_RESPONSIVE"


@dataclass
class Config:
    """Configuration values for launch and monitoring."""

    startup_poll_interval_ms: int = 300
    startup_timeout_s: int = 20
    health_check_interval_s: int = 3
    window_wait_timeout_ms: int = 200
    require_window: bool = True
    require_responsive: bool = True
    soft_fail_threshold: int = 3
    steam_process_name: str = "steam.exe"
    steam_app_id: int = 1973530
    process_name_candidates: List[str] = field(default_factory=lambda: ["LimbusCompany.exe"])
    steam_uri_template: str = "steam://run/{app_id}"


class WinApiAdapter:
    """Thin adapter for Win32 API calls."""

    SYNCHRONIZE = 0x00100000
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    PROCESS_QUERY_INFORMATION = 0x0400

    WAIT_OBJECT_0 = 0x00000000
    WAIT_TIMEOUT = 0x00000102

    WM_NULL = 0x0000
    SMTO_ABORTIFHUNG = 0x0002

    def __init__(self) -> None:
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)

        self.kernel32.OpenProcess.argtypes = [ctypes.c_uint, ctypes.c_bool, ctypes.c_uint]
        self.kernel32.OpenProcess.restype = ctypes.c_void_p

        self.kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        self.kernel32.WaitForSingleObject.restype = ctypes.c_uint

        self.kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
        self.kernel32.GetExitCodeProcess.restype = ctypes.c_bool

        self.kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        self.kernel32.CloseHandle.restype = ctypes.c_bool

        self.user32.EnumWindows.argtypes = [ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p), ctypes.c_void_p]
        self.user32.EnumWindows.restype = ctypes.c_bool

        self.user32.GetWindowThreadProcessId.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
        self.user32.GetWindowThreadProcessId.restype = ctypes.c_uint

        self.user32.IsWindowVisible.argtypes = [ctypes.c_void_p]
        self.user32.IsWindowVisible.restype = ctypes.c_bool

        self.user32.SendMessageTimeoutW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.user32.SendMessageTimeoutW.restype = ctypes.c_void_p

    def open_process(self, pid: int) -> Optional[int]:
        """Open a process handle for monitoring."""
        desired = self.SYNCHRONIZE | self.PROCESS_QUERY_LIMITED_INFORMATION | self.PROCESS_QUERY_INFORMATION
        handle = self.kernel32.OpenProcess(desired, False, pid)
        if not handle:
            return None
        return int(handle)

    def wait_for_process_exit(self, handle: int, timeout_ms: int) -> int:
        """Wait for process exit or timeout."""
        return int(self.kernel32.WaitForSingleObject(ctypes.c_void_p(handle), timeout_ms))

    def get_exit_code(self, handle: int) -> Optional[int]:
        """Get the exit code for a process handle."""
        exit_code = ctypes.c_uint(0)
        success = self.kernel32.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(exit_code))
        if not success:
            return None
        return int(exit_code.value)

    def close_handle(self, handle: int) -> None:
        """Close a process handle."""
        if handle:
            self.kernel32.CloseHandle(ctypes.c_void_p(handle))

    def enum_windows_for_pid(self, pid: int) -> List[int]:
        """Enumerate top-level window handles that belong to the PID."""
        handles: List[int] = []

        @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        def enum_proc(hwnd: ctypes.c_void_p, lparam: ctypes.c_void_p) -> bool:
            window_pid = ctypes.c_uint(0)
            self.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(window_pid))
            if window_pid.value == pid and self.user32.IsWindowVisible(hwnd):
                handles.append(int(hwnd))
            return True

        self.user32.EnumWindows(enum_proc, 0)
        return handles

    def send_message_timeout(self, hwnd: int, timeout_ms: int) -> bool:
        """Check responsiveness by sending a WM_NULL message."""
        result = ctypes.c_void_p(0)
        response = self.user32.SendMessageTimeoutW(
            ctypes.c_void_p(hwnd),
            self.WM_NULL,
            ctypes.c_void_p(0),
            ctypes.c_void_p(0),
            self.SMTO_ABORTIFHUNG,
            timeout_ms,
            ctypes.byref(result),
        )
        return bool(response)


def log_event(code: str, **fields: object) -> None:
    """Write a structured log event to stdout."""
    payload = {"code": code, **fields}
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def is_process_running(process_name: str) -> bool:
    """Check if a process with the given name is running."""
    for proc in psutil.process_iter(["name"]):
        try:
            if proc.info.get("name", "").lower() == process_name.lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


def find_process(candidates: Sequence[str], baseline_time: float) -> Optional[psutil.Process]:
    """Find a candidate process that started after baseline_time."""
    candidate_lower = {name.lower() for name in candidates}
    for proc in psutil.process_iter(["pid", "name", "create_time"]):
        try:
            name = (proc.info.get("name") or "").lower()
            if name not in candidate_lower:
                continue
            create_time = float(proc.info.get("create_time") or 0.0)
            if create_time >= baseline_time - 1.0:
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
            continue
    return None


def launch_game(app_id: int, uri_template: str) -> None:
    """Launch game via Steam protocol."""
    uri = uri_template.format(app_id=app_id)
    subprocess_args = ["cmd", "/c", "start", "", uri]
    import subprocess

    subprocess.Popen(subprocess_args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Launch Limbus Company via Steam and monitor health.")
    parser.add_argument("--app-id", type=int, default=None, help="Steam app id to launch.")
    parser.add_argument("--proc-name", action="append", default=None, help="Process name to match (repeatable).")
    parser.add_argument("--require-window", dest="require_window", action="store_true", default=None)
    parser.add_argument("--no-require-window", dest="require_window", action="store_false", default=None)
    parser.add_argument("--require-responsive", dest="require_responsive", action="store_true", default=None)
    parser.add_argument("--no-require-responsive", dest="require_responsive", action="store_false", default=None)
    parser.add_argument("--startup-timeout", type=int, default=None, help="Startup timeout seconds.")
    parser.add_argument("--startup-poll-interval-ms", type=int, default=None)
    parser.add_argument("--health-check-interval", type=int, default=None, help="Health check interval seconds.")
    parser.add_argument("--window-wait-timeout-ms", type=int, default=None)
    parser.add_argument("--soft-fail-threshold", type=int, default=None)
    parser.add_argument("--steam-uri-template", type=str, default=None)
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    """Build config from defaults and CLI overrides."""
    config = Config()
    if args.app_id is not None:
        config.steam_app_id = args.app_id
    if args.proc_name:
        config.process_name_candidates = list(args.proc_name)
    if args.require_window is not None:
        config.require_window = args.require_window
    if args.require_responsive is not None:
        config.require_responsive = args.require_responsive
    if args.startup_timeout is not None:
        config.startup_timeout_s = args.startup_timeout
    if args.startup_poll_interval_ms is not None:
        config.startup_poll_interval_ms = args.startup_poll_interval_ms
    if args.health_check_interval is not None:
        config.health_check_interval_s = args.health_check_interval
    if args.window_wait_timeout_ms is not None:
        config.window_wait_timeout_ms = args.window_wait_timeout_ms
    if args.soft_fail_threshold is not None:
        config.soft_fail_threshold = args.soft_fail_threshold
    if args.steam_uri_template is not None:
        config.steam_uri_template = args.steam_uri_template
    return config


def monitor_game(config: Config) -> int:
    """Main monitoring routine implementing the state machine."""
    state = "INIT"
    error_code: Optional[str] = None
    api = WinApiAdapter()

    state = "CHECK_STEAM"
    if is_process_running(config.steam_process_name):
        log_event("I_STEAM_DETECTED", process=config.steam_process_name)
    else:
        log_event(ErrorCodes.E_STEAM_NOT_RUNNING, process=config.steam_process_name)
        return 2

    state = "LAUNCH_GAME"
    launch_time = time.time()
    launch_game(config.steam_app_id, config.steam_uri_template)
    log_event("I_LAUNCH_REQUEST_SENT", app_id=config.steam_app_id, uri_template=config.steam_uri_template)

    state = "WAIT_FOR_PROCESS"
    wait_start = time.time()
    log_event("I_WAIT_FOR_PROCESS", phase="start", timeout_s=config.startup_timeout_s)
    target_proc: Optional[psutil.Process] = None
    while time.time() - launch_time <= config.startup_timeout_s:
        target_proc = find_process(config.process_name_candidates, launch_time)
        if target_proc:
            break
        time.sleep(config.startup_poll_interval_ms / 1000.0)
    wait_elapsed = time.time() - wait_start
    log_event("I_WAIT_FOR_PROCESS", phase="end", elapsed_s=round(wait_elapsed, 3))

    if not target_proc:
        log_event(ErrorCodes.E_GAME_START_TIMEOUT, timeout_s=config.startup_timeout_s)
        return 2

    state = "BIND_PROCESS"
    pid = target_proc.pid
    proc_name = target_proc.name()
    log_event("I_PROCESS_DETECTED", pid=pid, process=proc_name, create_time=target_proc.create_time())
    handle = api.open_process(pid)
    if not handle:
        last_error = ctypes.get_last_error()
        log_event(ErrorCodes.E_OPEN_PROCESS_HANDLE_FAILED, pid=pid, win32_error=last_error)
        return 2
    log_event("I_HANDLE_BOUND", pid=pid, handle=handle)

    state = "RUNNING"
    start_run_time = time.time()
    soft_fail_count = 0
    last_failure_code: Optional[str] = None

    try:
        while True:
            wait_result = api.wait_for_process_exit(handle, int(config.health_check_interval_s * 1000))
            if wait_result == api.WAIT_OBJECT_0:
                exit_code = api.get_exit_code(handle)
                runtime_s = round(time.time() - start_run_time, 3)
                abnormal = exit_code not in (0, None)
                log_event(
                    "I_PROCESS_EXITED",
                    pid=pid,
                    exit_code=exit_code,
                    runtime_s=runtime_s,
                    abnormal_exit=abnormal,
                )
                if exit_code == 0:
                    return 0
                return 2

            windows: List[int] = []
            if config.require_window:
                windows = api.enum_windows_for_pid(pid)
                if not windows:
                    soft_fail_count += 1
                    last_failure_code = ErrorCodes.E_WINDOW_NOT_FOUND
                    log_event(ErrorCodes.E_WINDOW_NOT_FOUND, pid=pid, soft_fail_count=soft_fail_count)
                else:
                    if config.require_responsive:
                        responsive = any(api.send_message_timeout(hwnd, config.window_wait_timeout_ms) for hwnd in windows)
                        if not responsive:
                            soft_fail_count += 1
                            last_failure_code = ErrorCodes.E_WINDOW_NOT_RESPONSIVE
                            log_event(
                                ErrorCodes.E_WINDOW_NOT_RESPONSIVE,
                                pid=pid,
                                soft_fail_count=soft_fail_count,
                            )
                        else:
                            soft_fail_count = 0
                            log_event("I_HEALTH_OK", pid=pid, soft_fail_count=soft_fail_count)
                    else:
                        soft_fail_count = 0
                        log_event("I_HEALTH_OK", pid=pid, soft_fail_count=soft_fail_count)
            else:
                soft_fail_count = 0
                log_event("I_HEALTH_OK", pid=pid, soft_fail_count=soft_fail_count)

            if last_failure_code and soft_fail_count >= config.soft_fail_threshold:
                log_event(last_failure_code, pid=pid, soft_fail_count=soft_fail_count, threshold=config.soft_fail_threshold)
                return 2
    finally:
        api.close_handle(handle)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point."""
    try:
        args = parse_args(argv)
        config = build_config(args)
    except SystemExit:
        return 1
    return monitor_game(config)


if __name__ == "__main__":
    sys.exit(main())
