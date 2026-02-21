#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""仅监控 Limbus Company 进程生命周期（启动/关闭）并输出中文日志。"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

try:
    import psutil
except Exception:  # pragma: no cover
    print("[错误] 缺少依赖 psutil，请先执行: pip install psutil", file=sys.stderr)
    raise


DEFAULT_PROC_NAME = "LimbusCompany.exe"


def now_local_str() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@dataclass
class Config:
    process_names: List[str]
    steam_process_name: str
    steam_app_id: int
    exit_on_close: bool
    poll_interval_s: float
    jsonl_file: Path
    text_log_file: Path


class DualLogger:
    """同步写入 jsonl 和中文文本日志。"""

    def __init__(self, jsonl_path: Path, text_path: Path) -> None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.parent.mkdir(parents=True, exist_ok=True)
        self._jsonl = jsonl_path
        self._text = text_path

    def log(self, event: str, message_cn: str, **extra: object) -> None:
        payload: Dict[str, object] = {
            "ts": now_iso(),
            "event": event,
            "message_cn": message_cn,
        }
        payload.update(extra)

        line = json.dumps(payload, ensure_ascii=False)
        with self._jsonl.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

        text_line = f"[{now_local_str()}] {message_cn}"
        if extra:
            text_line = f"{text_line} | {json.dumps(extra, ensure_ascii=False)}"
        with self._text.open("a", encoding="utf-8") as f:
            f.write(text_line + "\n")

        print(text_line, flush=True)


class GameLifecycleMonitor:
    def __init__(self, config: Config, logger: DualLogger) -> None:
        self.config = config
        self.logger = logger
        self._running = True

    def stop(self, *_: object) -> None:
        self._running = False

    def _snapshot(self) -> Dict[int, str]:
        names = {n.lower() for n in self.config.process_names}
        found: Dict[int, str] = {}
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                name = (proc.info.get("name") or "").strip()
                if name.lower() in names:
                    found[int(proc.info["pid"])] = name
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, KeyError):
                continue
        return found

    def _is_process_running(self, process_name: str) -> bool:
        target = process_name.lower()
        for proc in psutil.process_iter(["name"]):
            try:
                name = (proc.info.get("name") or "").strip().lower()
                if name == target:
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return False

    def _launch_game_by_steam(self) -> bool:
        uri = f"steam://run/{self.config.steam_app_id}"
        try:
            if os.name == "nt":
                os.startfile(uri)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(
                    ["cmd", "/c", "start", "", uri],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            return True
        except Exception:
            return False

    def run(self) -> int:
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)

        watch_names = ", ".join(self.config.process_names)
        self.logger.log(
            event="MONITOR_STARTED",
            message_cn=f"监控已启动，目标进程：{watch_names}，轮询间隔：{self.config.poll_interval_s:.1f}秒",
            process_names=self.config.process_names,
            poll_interval_s=self.config.poll_interval_s,
            jsonl_file=str(self.config.jsonl_file),
            text_log_file=str(self.config.text_log_file),
        )

        if not self._is_process_running(self.config.steam_process_name):
            self.logger.log(
                event="E_STEAM_NOT_RUNNING",
                message_cn=f"异常：未检测到 {self.config.steam_process_name}，程序结束",
                steam_process=self.config.steam_process_name,
            )
            return 2
        self.logger.log(
            event="I_STEAM_DETECTED",
            message_cn=f"检测到 Steam 进程：{self.config.steam_process_name}",
            steam_process=self.config.steam_process_name,
        )

        if not self._launch_game_by_steam():
            self.logger.log(
                event="E_GAME_LAUNCH_FAILED",
                message_cn=f"异常：已检测到 Steam，但请求启动游戏失败（AppID={self.config.steam_app_id}）",
                app_id=self.config.steam_app_id,
            )
            return 2
        self.logger.log(
            event="I_GAME_LAUNCH_REQUESTED",
            message_cn=f"已发送游戏启动请求（AppID={self.config.steam_app_id}）",
            app_id=self.config.steam_app_id,
        )

        prev: Set[int] = set(self._snapshot().keys())
        had_running_game = bool(prev)
        if prev:
            self.logger.log(
                event="GAME_ALREADY_RUNNING",
                message_cn="脚本启动时检测到游戏已在运行，后续只监控关闭事件",
                pids=sorted(prev),
            )

        while self._running:
            current_map = self._snapshot()
            current = set(current_map.keys())

            started = sorted(current - prev)
            closed = sorted(prev - current)

            for pid in started:
                had_running_game = True
                self.logger.log(
                    event="GAME_STARTED",
                    message_cn=f"检测到游戏启动：PID={pid}，进程名={current_map.get(pid, '未知')}",
                    pid=pid,
                    process=current_map.get(pid, ""),
                )

            for pid in closed:
                self.logger.log(
                    event="GAME_CLOSED",
                    message_cn=f"检测到游戏关闭：PID={pid}",
                    pid=pid,
                )

            if self.config.exit_on_close and had_running_game and not current:
                self.logger.log(
                    event="MONITOR_AUTO_STOP",
                    message_cn="检测到游戏已全部关闭，监控自动结束",
                )
                self._running = False
                break

            prev = current
            time.sleep(self.config.poll_interval_s)

        self.logger.log(event="MONITOR_STOPPED", message_cn="监控已停止")
        return 0


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Limbus Company 生命周期监控（仅启动/关闭）")
    parser.add_argument(
        "--proc-name",
        action="append",
        default=None,
        help=f"要监控的进程名，可重复传入。默认：{DEFAULT_PROC_NAME}",
    )
    parser.add_argument("--steam-name", default="steam.exe", help="Steam 进程名，默认 steam.exe")
    parser.add_argument("--steam-app-id", type=int, default=1973530, help="Steam AppID，默认 1973530")
    parser.add_argument(
        "--keep-window-after-close",
        action="store_true",
        help="游戏关闭后不自动结束监控（默认自动结束）",
    )
    parser.add_argument("--poll", type=float, default=1.0, help="轮询间隔（秒），默认 1.0")
    parser.add_argument(
        "--jsonl",
        default=r"G:\Project\limbuscompany_Scripts3\runtime\logs\limbus_lifecycle.jsonl",
        help="jsonl 日志路径",
    )
    parser.add_argument(
        "--text-log",
        default=r"G:\Project\limbuscompany_Scripts3\runtime\logs\limbus_lifecycle.log",
        help="中文文本日志路径",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    process_names = args.proc_name if args.proc_name else [DEFAULT_PROC_NAME]
    cfg = Config(
        process_names=process_names,
        steam_process_name=args.steam_name,
        steam_app_id=int(args.steam_app_id),
        exit_on_close=not bool(args.keep_window_after_close),
        poll_interval_s=max(0.2, float(args.poll)),
        jsonl_file=Path(args.jsonl),
        text_log_file=Path(args.text_log),
    )
    logger = DualLogger(cfg.jsonl_file, cfg.text_log_file)
    monitor = GameLifecycleMonitor(cfg, logger)
    return monitor.run()


if __name__ == "__main__":
    raise SystemExit(main())
