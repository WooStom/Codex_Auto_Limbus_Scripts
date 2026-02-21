#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SPLASH/LOGIN/CONNECTING/LOBBY/RECHARGE 模板识别 + 状态机 + 实时画框窗口。"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import mss
import numpy as np


STATE_SPLASH = "SPLASH"
STATE_LOGIN = "LOGIN"
STATE_CONNECTING = "CONNECTING"
STATE_RECHARGE = "RECHARGE"
STATE_LOBBY = "LOBBY"
STATE_UNKNOWN = "UNKNOWN"


@dataclass
class Anchor:
    state: str
    path: Path
    image_gray: np.ndarray
    width: int
    height: int


@dataclass
class DetectResult:
    state: str
    score: float
    top_left: Tuple[int, int]
    width: int
    height: int
    anchor_name: str


class TextJsonLogger:
    def __init__(self, text_log: Path, jsonl_log: Path) -> None:
        text_log.parent.mkdir(parents=True, exist_ok=True)
        jsonl_log.parent.mkdir(parents=True, exist_ok=True)
        self.text_log = text_log
        self.jsonl_log = jsonl_log

    def log(self, event: str, message_cn: str, **extra: object) -> None:
        ts_local = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ts_iso = datetime.now().astimezone().isoformat(timespec="seconds")

        text = f"[{ts_local}] {message_cn}"
        if extra:
            text = f"{text} | {json.dumps(extra, ensure_ascii=False)}"
        print(text, flush=True)
        with self.text_log.open("a", encoding="utf-8") as f:
            f.write(text + "\n")

        payload = {"ts": ts_iso, "event": event, "message_cn": message_cn}
        payload.update(extra)
        with self.jsonl_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _read_image_cn(path: Path) -> Optional[np.ndarray]:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def load_anchors(anchor_dir: Path, target_res: str) -> List[Anchor]:
    anchors: List[Anchor] = []
    for state in [STATE_CONNECTING, STATE_RECHARGE, STATE_LOBBY, STATE_SPLASH, STATE_LOGIN]:
        pattern = f"{state}_{target_res}_*.png"
        for p in sorted(anchor_dir.glob(pattern)):
            image_bgr = _read_image_cn(p)
            if image_bgr is None:
                continue
            image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            h, w = image_gray.shape[:2]
            anchors.append(
                Anchor(
                    state=state,
                    path=p,
                    image_gray=image_gray,
                    width=w,
                    height=h,
                )
            )
    return anchors


def find_window_rect(title_keyword: str) -> Optional[Tuple[int, int, int, int]]:
    user32 = ctypes.windll.user32
    keyword = title_keyword.lower()
    found: List[Tuple[int, int, int, int]] = []

    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def enum_cb(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True

        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True

        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value or ""
        if keyword not in title.lower():
            return True

        rect = ctypes.wintypes.RECT()  # type: ignore[attr-defined]
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return True

        width = int(rect.right - rect.left)
        height = int(rect.bottom - rect.top)
        if width <= 0 or height <= 0:
            return True

        found.append((int(rect.left), int(rect.top), width, height))
        return False

    user32.EnumWindows(EnumWindowsProc(enum_cb), 0)
    return found[0] if found else None


def best_detection_by_state(frame_gray: np.ndarray, anchors: List[Anchor]) -> Dict[str, DetectResult]:
    best: Dict[str, DetectResult] = {}
    for a in anchors:
        if a.height > frame_gray.shape[0] or a.width > frame_gray.shape[1]:
            continue
        result = cv2.matchTemplate(frame_gray, a.image_gray, cv2.TM_CCOEFF_NORMED)
        _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(result)
        det = DetectResult(
            state=a.state,
            score=float(max_val),
            top_left=(int(max_loc[0]), int(max_loc[1])),
            width=a.width,
            height=a.height,
            anchor_name=a.path.name,
        )
        prev = best.get(a.state)
        if prev is None or det.score > prev.score:
            best[a.state] = det
    return best


def is_process_running(process_name: str) -> bool:
    target = process_name.lower()
    import psutil

    for proc in psutil.process_iter(["name"]):
        try:
            name = (proc.info.get("name") or "").strip().lower()
            if name == target:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return False


def pass_threshold(det: Optional[DetectResult], default_threshold: float, lobby_threshold: float) -> bool:
    if det is None:
        return False
    if det.state == STATE_LOBBY:
        return det.score >= lobby_threshold
    return det.score >= default_threshold


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SPLASH/LOGIN/CONNECTING/LOBBY/RECHARGE 模板识别状态机（日志 + 实时画框）")
    parser.add_argument(
        "--anchor-dir",
        default=r"G:\Project\limbuscompany_Scripts3\anchorPoint",
        help="锚点图片目录",
    )
    parser.add_argument("--window-title", default="LimbusCompany", help="窗口标题关键字")
    parser.add_argument("--proc-name", default="LimbusCompany.exe", help="游戏进程名，默认 LimbusCompany.exe")
    parser.add_argument("--target-res", default="800_600", help="锚点分辨率标记，默认 800_600")
    parser.add_argument("--threshold", type=float, default=0.80, help="匹配阈值，默认 0.80")
    parser.add_argument("--lobby-threshold", type=float, default=0.75, help="LOBBY 匹配阈值，默认 0.75")
    parser.add_argument(
        "--login-priority-margin",
        type=float,
        default=0.03,
        help="LOGIN 优先边际分差，默认 0.03（防止登录阶段被 SPLASH 干扰）",
    )
    parser.add_argument("--stable-frames", type=int, default=3, help="连续命中帧数，默认 3")
    parser.add_argument("--poll", type=float, default=0.10, help="轮询间隔秒，默认 0.10")
    parser.add_argument(
        "--keep-running-after-game-close",
        action="store_true",
        help="游戏关闭后继续运行（默认会自动退出）",
    )
    parser.add_argument(
        "--text-log",
        default=r"G:\Project\limbuscompany_Scripts3\runtime\logs\fsm_splash_login.log",
        help="中文日志路径",
    )
    parser.add_argument(
        "--jsonl-log",
        default=r"G:\Project\limbuscompany_Scripts3\runtime\logs\fsm_splash_login.jsonl",
        help="jsonl 日志路径",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    if os.name != "nt":
        print("该脚本仅支持 Windows 运行。")
        return 2

    anchor_dir = Path(args.anchor_dir)
    logger = TextJsonLogger(Path(args.text_log), Path(args.jsonl_log))

    anchors = load_anchors(anchor_dir, args.target_res)
    if not anchors:
        logger.log("E_NO_ANCHORS", "异常：未加载到锚点图片", anchor_dir=str(anchor_dir), target_res=args.target_res)
        return 2

    logger.log(
        "I_ANCHORS_LOADED",
        "锚点加载完成",
        count=len(anchors),
        anchor_dir=str(anchor_dir),
        target_res=args.target_res,
    )

    current_state = STATE_UNKNOWN
    pending_state = STATE_UNKNOWN
    pending_count = 0
    seen_game_running = False

    logger.log(
        "I_FSM_STARTED",
        "状态机启动：SPLASH/LOGIN/CONNECTING/LOBBY/RECHARGE",
        threshold=args.threshold,
        lobby_threshold=args.lobby_threshold,
        login_priority_margin=args.login_priority_margin,
        stable_frames=args.stable_frames,
        window_title=args.window_title,
        proc_name=args.proc_name,
    )

    with mss.mss() as sct:
        while True:
            running_now = is_process_running(args.proc_name)
            if running_now:
                seen_game_running = True
            elif seen_game_running and not args.keep_running_after_game_close:
                logger.log(
                    "I_AUTO_STOP_GAME_CLOSED",
                    "检测到游戏已关闭，状态机自动结束",
                    proc_name=args.proc_name,
                )
                break

            rect = find_window_rect(args.window_title)
            if rect is None:
                logger.log(
                    "I_WAIT_WINDOW",
                    "等待游戏窗口出现",
                    window_title=args.window_title,
                    proc_running=running_now,
                )
                time.sleep(1.0)
                continue

            left, top, width, height = rect
            shot = sct.grab({"left": left, "top": top, "width": width, "height": height})
            frame_bgr = np.array(shot)[:, :, :3]
            frame_bgr = cv2.resize(frame_bgr, (800, 600), interpolation=cv2.INTER_AREA)
            frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            best = best_detection_by_state(frame_gray, anchors)
            connecting_det = best.get(STATE_CONNECTING)
            recharge_det = best.get(STATE_RECHARGE)
            lobby_det = best.get(STATE_LOBBY)
            splash_det = best.get(STATE_SPLASH)
            login_det = best.get(STATE_LOGIN)

            scores = [
                connecting_det.score if pass_threshold(connecting_det, args.threshold, args.lobby_threshold) else -1.0,
                recharge_det.score if pass_threshold(recharge_det, args.threshold, args.lobby_threshold) else -1.0,
                lobby_det.score if pass_threshold(lobby_det, args.threshold, args.lobby_threshold) else -1.0,
                splash_det.score if pass_threshold(splash_det, args.threshold, args.lobby_threshold) else -1.0,
                login_det.score if pass_threshold(login_det, args.threshold, args.lobby_threshold) else -1.0,
            ]
            max_score = max(scores)
            # scores 已经按各自阈值过滤，>=0 代表至少有一个状态通过对应阈值。
            has_valid_anchor = max_score >= 0.0

            candidate: Optional[DetectResult] = None

            if not has_valid_anchor:
                pending_state = STATE_UNKNOWN
                pending_count = 0
                if current_state != STATE_UNKNOWN:
                    prev = current_state
                    current_state = STATE_UNKNOWN
                    logger.log(
                        "STATE_TRANSITION",
                        f"状态切换：{prev} -> {current_state}",
                        from_state=prev,
                        to_state=current_state,
                        reason="all_scores_below_threshold",
                        max_score=round(max_score, 4),
                    )
            else:
                # CONNECTING 最高优先级：只要出现就立即抢占当前状态。
                if connecting_det and connecting_det.score >= args.threshold:
                    prev = current_state
                    current_state = STATE_CONNECTING
                    candidate = connecting_det
                    pending_state = STATE_UNKNOWN
                    pending_count = 0
                    if prev != current_state:
                        logger.log(
                            "STATE_TRANSITION",
                            f"状态切换：{prev} -> {current_state}",
                            from_state=prev,
                            to_state=current_state,
                            score=round(connecting_det.score, 4),
                            anchor=connecting_det.anchor_name,
                        )

                # 非 CONNECTING 状态按权限等级选择：
                # RECHARGE > LOBBY > LOGIN > SPLASH
                if candidate is None:
                    for det in [recharge_det, lobby_det, login_det, splash_det]:
                        if pass_threshold(det, args.threshold, args.lobby_threshold):
                            candidate = det
                            break

                if pass_threshold(candidate, args.threshold, args.lobby_threshold):
                    if candidate.state == current_state:
                        pending_state = STATE_UNKNOWN
                        pending_count = 0
                    else:
                        if pending_state == candidate.state:
                            pending_count += 1
                        else:
                            pending_state = candidate.state
                            pending_count = 1
                        if pending_count >= max(1, args.stable_frames):
                            prev = current_state
                            current_state = candidate.state
                            logger.log(
                                "STATE_TRANSITION",
                                f"状态切换：{prev} -> {current_state}",
                                from_state=prev,
                                to_state=current_state,
                                score=round(candidate.score, 4),
                                anchor=candidate.anchor_name,
                            )
                            pending_state = STATE_UNKNOWN
                            pending_count = 0
                else:
                    pending_state = STATE_UNKNOWN
                    pending_count = 0

            # 只显示一个主框：优先当前状态，其次候选状态，避免双框误导。
            display_det: Optional[DetectResult] = None
            if current_state == STATE_CONNECTING and pass_threshold(connecting_det, args.threshold, args.lobby_threshold):
                display_det = connecting_det
            elif current_state == STATE_RECHARGE and pass_threshold(recharge_det, args.threshold, args.lobby_threshold):
                display_det = recharge_det
            elif current_state == STATE_LOBBY and pass_threshold(lobby_det, args.threshold, args.lobby_threshold):
                display_det = lobby_det
            elif current_state == STATE_LOGIN and pass_threshold(login_det, args.threshold, args.lobby_threshold):
                display_det = login_det
            elif current_state == STATE_SPLASH and pass_threshold(splash_det, args.threshold, args.lobby_threshold):
                display_det = splash_det
            elif pass_threshold(candidate, args.threshold, args.lobby_threshold):
                display_det = candidate

            if display_det:
                if display_det.state == STATE_CONNECTING:
                    color = (255, 120, 0)
                elif display_det.state == STATE_RECHARGE:
                    color = (255, 0, 255)
                elif display_det.state == STATE_LOBBY:
                    color = (255, 255, 0)
                elif display_det.state == STATE_LOGIN:
                    color = (0, 200, 0)
                else:
                    color = (0, 255, 255)
                p1 = display_det.top_left
                p2 = (p1[0] + display_det.width, p1[1] + display_det.height)
                cv2.rectangle(frame_bgr, p1, p2, color, 2)
                cv2.putText(
                    frame_bgr,
                    f"{display_det.state} {display_det.score:.3f}",
                    (p1[0], max(15, p1[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

            # 两个分数仍保留，便于调参观察。
            connecting_score_text = f"{connecting_det.score:.3f}" if connecting_det else "N/A"
            recharge_score_text = f"{recharge_det.score:.3f}" if recharge_det else "N/A"
            lobby_score_text = f"{lobby_det.score:.3f}" if lobby_det else "N/A"
            splash_score_text = f"{splash_det.score:.3f}" if splash_det else "N/A"
            login_score_text = f"{login_det.score:.3f}" if login_det else "N/A"
            cv2.putText(
                frame_bgr,
                f"CONNECTING={connecting_score_text}  RECHARGE={recharge_score_text}  LOBBY={lobby_score_text}  SPLASH={splash_score_text}  LOGIN={login_score_text}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame_bgr,
                f"FSM={current_state}  TH={args.threshold:.2f}  STABLE={args.stable_frames}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (30, 30, 255),
                2,
            )
            cv2.putText(
                frame_bgr,
                "Press Q to quit",
                (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Limbus SPLASH-LOGIN FSM", frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                logger.log("I_MANUAL_STOP", "手动停止状态机")
                break

            time.sleep(max(0.0, float(args.poll)))

    cv2.destroyAllWindows()
    logger.log("I_FSM_STOPPED", "状态机已结束")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
