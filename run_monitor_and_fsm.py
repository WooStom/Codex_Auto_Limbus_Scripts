#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在单一控制台中同时运行 monitor + fsm，并合并输出日志。"""

from __future__ import annotations

import json
import locale
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class LiveStatus:
    game_status: str = "未知"
    fsm_state: str = "UNKNOWN"
    last_emit_at: Dict[str, float] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)


def _extract_message(raw: str) -> tuple[str, Dict[str, object]]:
    msg = raw
    if raw.startswith("[") and "] " in raw:
        msg = raw.split("] ", 1)[1]
    if " | " in msg:
        base, _, meta = msg.partition(" | ")
        try:
            return base, json.loads(meta)
        except Exception:
            return base, {}
    return msg, {}


def _should_emit(status: LiveStatus, key: str, interval_s: float) -> bool:
    now = time.time()
    with status.lock:
        last = status.last_emit_at.get(key, 0.0)
        if (now - last) < interval_s:
            return False
        status.last_emit_at[key] = now
    return True


def _update_status(status: LiveStatus, tag: str, message: str, meta: Dict[str, object]) -> None:
    with status.lock:
        if tag == "MONITOR":
            if "检测到游戏启动" in message:
                status.game_status = "运行中"
            elif "检测到游戏关闭" in message or "已全部关闭" in message:
                status.game_status = "已关闭"
            elif "未检测到" in message and "steam" in message.lower():
                status.game_status = "启动失败"
        elif tag == "FSM":
            if "状态切换：" in message:
                m = re.search(r"->\s*([A-Z_]+)", message)
                if m:
                    status.fsm_state = m.group(1)
            to_state = meta.get("to_state")
            if isinstance(to_state, str) and to_state:
                status.fsm_state = to_state


def _format_line(status: LiveStatus, tag: str, raw_line: str) -> Optional[str]:
    message, meta = _extract_message(raw_line)
    _update_status(status, tag, message, meta)

    if "等待游戏窗口出现" in message:
        if not _should_emit(status, "wait_window", 5.0):
            return None

    level = "INFO"
    if "异常" in message or "失败" in message:
        level = "ERROR"
    elif "状态切换" in message:
        level = "STATE"

    with status.lock:
        game = status.game_status
        fsm = status.fsm_state

    return f"[{_ts()}][{tag}][{level}][游戏={game}][FSM={fsm}] {message}"


def _decode_line(raw: bytes) -> str:
    encodings = [
        "utf-8",
        locale.getpreferredencoding(False) or "gb18030",
        "gb18030",
        "cp936",
    ]
    used = set()
    for enc in encodings:
        if enc in used:
            continue
        used.add(enc)
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _pump_output(proc: subprocess.Popen[bytes], tag: str, status: LiveStatus) -> None:
    if proc.stdout is None:
        return
    for raw in iter(proc.stdout.readline, b""):
        text = _decode_line(raw).rstrip("\r\n")
        if text:
            formatted = _format_line(status, tag, text)
            if formatted:
                print(formatted, flush=True)


def _terminate(proc: Optional[subprocess.Popen[str]], name: str) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    try:
        print(f"[{_ts()}][MAIN] 正在停止 {name}...", flush=True)
        proc.terminate()
    except Exception:
        pass


def _snapshot_status(status: LiveStatus) -> tuple[str, str]:
    with status.lock:
        return status.game_status, status.fsm_state


def main() -> int:
    base = Path(__file__).resolve().parent
    runtime_logs = base / "runtime" / "logs"
    runtime_logs.mkdir(parents=True, exist_ok=True)

    monitor_script = base / "game_lifecycle_monitor.py"
    fsm_script = base / "splash_login_fsm.py"

    if not monitor_script.exists():
        print(f"[{_ts()}][MAIN] 缺少脚本: {monitor_script}", flush=True)
        return 2
    if not fsm_script.exists():
        print(f"[{_ts()}][MAIN] 缺少脚本: {fsm_script}", flush=True)
        return 2

    monitor_cmd: List[str] = [
        sys.executable,
        str(monitor_script),
        "--jsonl",
        str(runtime_logs / "limbus_lifecycle.jsonl"),
        "--text-log",
        str(runtime_logs / "limbus_lifecycle.log"),
    ]
    fsm_cmd: List[str] = [
        sys.executable,
        str(fsm_script),
        "--jsonl-log",
        str(runtime_logs / "fsm_splash_login.jsonl"),
        "--text-log",
        str(runtime_logs / "fsm_splash_login.log"),
    ]

    status = LiveStatus()

    print(f"[{_ts()}][MAIN] 启动 monitor...", flush=True)
    monitor_proc = subprocess.Popen(
        monitor_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
        bufsize=0,
    )

    # 让 monitor 先发起游戏启动请求，再启动 fsm。
    time.sleep(1.0)

    print(f"[{_ts()}][MAIN] 启动 fsm...", flush=True)
    fsm_proc = subprocess.Popen(
        fsm_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
        bufsize=0,
    )

    t1 = threading.Thread(target=_pump_output, args=(monitor_proc, "MONITOR", status), daemon=True)
    t2 = threading.Thread(target=_pump_output, args=(fsm_proc, "FSM", status), daemon=True)
    t1.start()
    t2.start()

    stop_flag = False

    def _sig_handler(_signum: int, _frame: object) -> None:
        nonlocal stop_flag
        stop_flag = True

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    exit_code = 0
    last_status_print = 0.0
    last_status_value = ("", "")
    try:
        while True:
            if stop_flag:
                _terminate(monitor_proc, "monitor")
                _terminate(fsm_proc, "fsm")
                break

            mon_rc = monitor_proc.poll()
            fsm_rc = fsm_proc.poll()

            if mon_rc is not None and fsm_rc is not None:
                exit_code = 0 if (mon_rc == 0 and fsm_rc == 0) else 2
                break

            # 任一异常退出时，联动关闭另一个，避免残留进程。
            if mon_rc not in (None, 0):
                print(f"[{_ts()}][MAIN] monitor 异常退出，准备停止 fsm", flush=True)
                _terminate(fsm_proc, "fsm")
            if fsm_rc not in (None, 0):
                print(f"[{_ts()}][MAIN] fsm 异常退出，准备停止 monitor", flush=True)
                _terminate(monitor_proc, "monitor")

            game, fsm = _snapshot_status(status)
            now = time.time()
            # 状态变化立刻打印；无变化则每 5 秒打印一次心跳状态。
            if (game, fsm) != last_status_value or (now - last_status_print) >= 5.0:
                print(f"[{_ts()}][STATUS] 当前状态: 游戏={game}, FSM={fsm}", flush=True)
                last_status_value = (game, fsm)
                last_status_print = now

            time.sleep(0.2)
    finally:
        _terminate(monitor_proc, "monitor")
        _terminate(fsm_proc, "fsm")
        t1.join(timeout=2.0)
        t2.join(timeout=2.0)

    print(f"[{_ts()}][MAIN] monitor + fsm 已结束", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
