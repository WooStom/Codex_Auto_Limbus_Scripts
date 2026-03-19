#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SPLASH/LOGIN/CONNECTING/LOBBY/RECHARGE 模板识别 + 状态机 + 实时画框窗口。"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import json
import os
import random
import re
import threading
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


@dataclass
class FatigueResult:
    stable_value: Optional[int]
    stable_confidence: float
    candidate_value: Optional[int]
    candidate_confidence: float
    roi: Tuple[int, int, int, int]


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


class FatigueReader:
    """基于 OCR 读取体力左侧当前值。"""

    def __init__(
        self,
        sample_dir: Path,
        target_res: str,
        min_confidence: float = 0.55,
        stable_frames: int = 3,
    ) -> None:
        self.sample_dir = sample_dir
        self.target_res = target_res
        self.min_confidence = float(min_confidence)
        self.stable_frames = max(1, int(stable_frames))
        self.ocr_name = "RapidOCR"
        self.ocr = None
        self._candidate_value: Optional[int] = None
        self._candidate_count = 0
        self._stable_value: Optional[int] = None
        self._stable_confidence = 0.0
        try:
            from rapidocr_onnxruntime import RapidOCR

            self.ocr = RapidOCR()
        except Exception:
            self.ocr = None

    @property
    def template_count(self) -> int:
        # 沿用历史字段名，避免外部日志/脚本改动。
        return 1 if self.ocr is not None else 0

    def _roi_ratio(self) -> Tuple[float, float, float, float]:
        if self.target_res == "1920_1080":
            # ROI.docx: x=0.427 y=0.894 w=0.135 h=0.051
            return 0.427, 0.894, 0.135, 0.051
        # 800x600：仅覆盖左侧当前体力数字，排除 "/" 与右侧上限值。
        # dx=0.262000 dy=0.931000 rw=0.040000 rh=0.031000
        return 0.262000, 0.931000, 0.040000, 0.031000

    @staticmethod
    def _preprocess_variants(src: np.ndarray) -> List[np.ndarray]:
        up = cv2.resize(src, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        hsv = cv2.cvtColor(up, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (30, 20, 20), (110, 255, 255))
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw_bgr = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
        return [up, mask_bgr, bw_bgr]

    @staticmethod
    def _parse_left_number(text: str) -> Optional[int]:
        s = text.replace(" ", "")
        s = s.replace("O", "0").replace("o", "0")
        s = re.sub(r"[^0-9/]", "", s)
        if not s:
            return None
        if "/" in s:
            left = s.split("/", 1)[0]
            if left.isdigit():
                return int(left)
            return None
        m = re.search(r"\d{1,3}", s)
        if not m:
            return None
        return int(m.group(0))

    def _ocr_once(self, roi_bgr: np.ndarray) -> Tuple[Optional[int], float]:
        if self.ocr is None:
            return None, 0.0
        best_value: Optional[int] = None
        best_conf = 0.0

        def _run(images: List[np.ndarray]) -> Tuple[Optional[int], float]:
            local_best_value: Optional[int] = None
            local_best_conf = 0.0
            for img in images:
                try:
                    result, _elapse = self.ocr(img)
                except Exception:
                    continue
                if not result:
                    continue
                parts = sorted(result, key=lambda item: float(item[0][0][0]))
                text = "".join(str(item[1]) for item in parts)
                confs = [float(item[2]) for item in parts]
                value = self._parse_left_number(text)
                if value is None:
                    continue
                conf = float(np.mean(confs)) if confs else 0.0
                if value < 0 or value > 999:
                    continue
                if conf > local_best_conf:
                    local_best_conf = conf
                    local_best_value = value
            return local_best_value, local_best_conf

        variants = self._preprocess_variants(roi_bgr)
        # 优先左半区，减少把右侧“上限值”串进来。
        left_variants = [img[:, : max(8, int(img.shape[1] * 0.52))] for img in variants]
        best_value, best_conf = _run(left_variants)
        if best_value is None:
            best_value, best_conf = _run(variants)
        if best_conf < self.min_confidence:
            return None, best_conf
        return best_value, best_conf

    def read(self, frame_bgr: np.ndarray) -> FatigueResult:
        h, w = frame_bgr.shape[:2]
        rx, ry, rw, rh = self._roi_ratio()
        x = int(w * rx)
        y = int(h * ry)
        ww = max(8, int(w * rw))
        hh = max(8, int(h * rh))
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        ww = min(ww, w - x)
        hh = min(hh, h - y)
        # 识别使用放宽区域：显示框用于可视化，检测框向上/右扩展避免数字被截断。
        pad_left = 6
        pad_right = 2
        pad_up = 8
        pad_down = 4
        dx0 = max(0, x - pad_left)
        dy0 = max(0, y - pad_up)
        dx1 = min(w, x + ww + pad_right)
        dy1 = min(h, y + hh + pad_down)
        roi_detect = frame_bgr[dy0:dy1, dx0:dx1]

        candidate_value: Optional[int] = None
        candidate_conf = 0.0
        if roi_detect.size > 0:
            candidate_value, candidate_conf = self._ocr_once(roi_detect)

        if candidate_value is not None:
            if candidate_value == self._candidate_value:
                self._candidate_count += 1
            else:
                self._candidate_value = candidate_value
                self._candidate_count = 1
            if self._candidate_count >= self.stable_frames:
                self._stable_value = candidate_value
                self._stable_confidence = candidate_conf
        else:
            self._candidate_value = None
            self._candidate_count = 0

        return FatigueResult(
            stable_value=self._stable_value,
            stable_confidence=float(self._stable_confidence),
            candidate_value=candidate_value,
            candidate_confidence=float(candidate_conf),
            roi=(x, y, ww, hh),
        )


class AsyncFatigueWorker:
    """后台 OCR 线程：主线程只提交最新帧，避免渲染循环阻塞。"""

    def __init__(self, reader: FatigueReader) -> None:
        self.reader = reader
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._stop = False
        self._pending_frame: Optional[np.ndarray] = None
        self._latest_result = FatigueResult(
            stable_value=None,
            stable_confidence=0.0,
            candidate_value=None,
            candidate_confidence=0.0,
            roi=(0, 0, 0, 0),
        )
        self._thread = threading.Thread(target=self._run, name="fatigue-ocr-worker", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop = True
        self._event.set()
        self._thread.join(timeout=timeout)

    def submit(self, frame_bgr: np.ndarray) -> None:
        # 只保留最新帧，避免 OCR 排队造成延迟持续累积。
        with self._lock:
            self._pending_frame = frame_bgr.copy()
        self._event.set()

    def latest(self) -> FatigueResult:
        with self._lock:
            return self._latest_result

    def _run(self) -> None:
        while not self._stop:
            self._event.wait(timeout=0.2)
            self._event.clear()
            if self._stop:
                break
            frame: Optional[np.ndarray]
            with self._lock:
                frame = self._pending_frame
                self._pending_frame = None
            if frame is None:
                continue
            try:
                result = self.reader.read(frame)
            except Exception:
                continue
            with self._lock:
                self._latest_result = result


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


def terminate_processes_by_name(process_name: str) -> int:
    target = process_name.lower()
    import psutil

    stopped = 0
    for proc in psutil.process_iter(["name"]):
        try:
            name = (proc.info.get("name") or "").strip().lower()
            if name != target:
                continue
            proc.terminate()
            stopped += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return stopped


def pass_threshold(det: Optional[DetectResult], default_threshold: float, lobby_threshold: float) -> bool:
    if det is None:
        return False
    if det.state == STATE_LOBBY:
        return det.score >= lobby_threshold
    return det.score >= default_threshold


def ratio_roi_to_rect(frame_w: int, frame_h: int, roi_ratio: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
    rx, ry, rw, rh = roi_ratio
    x = int(frame_w * rx)
    y = int(frame_h * ry)
    w = max(2, int(frame_w * rw))
    h = max(2, int(frame_h * rh))
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = min(w, frame_w - x)
    h = min(h, frame_h - y)
    return x, y, w, h


def left_click_at_screen(x: int, y: int) -> None:
    user32 = ctypes.windll.user32
    # 先平滑移动鼠标，避免看起来像“瞬移”。
    pt = ctypes.wintypes.POINT()  # type: ignore[attr-defined]
    if user32.GetCursorPos(ctypes.byref(pt)):
        sx, sy = int(pt.x), int(pt.y)
    else:
        sx, sy = int(x), int(y)

    tx, ty = int(x), int(y)
    dx = tx - sx
    dy = ty - sy
    dist = max(abs(dx), abs(dy))
    steps = max(8, min(36, dist // 12 if dist > 0 else 8))
    # 轻微弧线偏移，让轨迹看起来更自然。
    arc = random.uniform(-6.0, 6.0)

    for i in range(1, steps + 1):
        t = i / float(steps)
        # ease-in-out
        e = 3 * t * t - 2 * t * t * t
        cx = sx + dx * e
        cy = sy + dy * e
        # 在中段增加一点法线方向偏移，形成弧线
        bend = (1.0 - abs(2.0 * t - 1.0)) * arc
        if dist > 0:
            nx = -dy / float(max(1, dist))
            ny = dx / float(max(1, dist))
            cx += nx * bend
            cy += ny * bend
        user32.SetCursorPos(int(round(cx)), int(round(cy)))
        time.sleep(0.003 + random.uniform(0.0, 0.003))

    user32.SetCursorPos(tx, ty)
    time.sleep(0.01 + random.uniform(0.0, 0.015))
    user32.mouse_event(0x0002, 0, 0, 0, 0)  # LEFTDOWN
    time.sleep(0.012 + random.uniform(0.0, 0.02))
    user32.mouse_event(0x0004, 0, 0, 0, 0)  # LEFTUP


def press_enter_key() -> None:
    """Send a single Enter key press using Win32 keyboard events."""
    user32 = ctypes.windll.user32
    vk_return = 0x0D
    keyeventf_keyup = 0x0002
    user32.keybd_event(vk_return, 0, 0, 0)
    time.sleep(0.03)
    user32.keybd_event(vk_return, 0, keyeventf_keyup, 0)


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
    parser.add_argument("--poll", type=float, default=0.03, help="轮询间隔秒，默认 0.03（更高帧率）")
    parser.add_argument(
        "--fatigue-sample-dir",
        default=r"G:\Project\limbuscompany_Scripts3\Fatigue_value\sample",
        help="体力数字样本目录（按文件名末尾数字作为标签）",
    )
    parser.add_argument("--fatigue-min-confidence", type=float, default=0.35, help="体力数字最小置信度，默认 0.35")
    parser.add_argument("--fatigue-stable-frames", type=int, default=3, help="体力值稳定帧数，默认 3")
    parser.add_argument("--fatigue-interval", type=float, default=0.45, help="体力 OCR 执行间隔秒，默认 0.45（降低卡顿）")
    parser.add_argument(
        "--login-click-roi",
        default="0.323,0.367,0.394,0.343",
        help="LOGIN 可点击区域归一化 ROI: x,y,w,h（基于 800x600）",
    )
    parser.add_argument(
        "--lobby-click-roi",
        default="0.238,0.905,0.130,0.055",
        help="LOBBY 可点击区域归一化 ROI: x,y,w,h（基于 800x600）",
    )
    parser.add_argument(
        "--recharge-click1-roi",
        default="0.512,0.708333,0.154,0.045",
        help="RECHARGE_1（黄色）可点击区域归一化 ROI: x,y,w,h（基于 800x600）",
    )
    parser.add_argument(
        "--recharge-click2-roi",
        default="0.614500,0.468333,0.028,0.034",
        help="RECHARGE_2（绿色）可点击区域归一化 ROI: x,y,w,h（基于 800x600）",
    )
    parser.add_argument("--login-auto-clicks", type=int, default=5, help="LOGIN 自动点击次数，默认 5")
    parser.add_argument("--login-click-interval-min", type=float, default=1.0, help="LOGIN 点击最小间隔秒，默认 1.0")
    parser.add_argument("--login-click-interval-max", type=float, default=1.5, help="LOGIN 点击最大间隔秒，默认 1.5")
    parser.add_argument("--lobby-fatigue-click-threshold", type=int, default=20, help="LOBBY 自动点击体力阈值，默认 >=20")
    parser.add_argument("--lobby-click-interval-min", type=float, default=2.0, help="LOBBY 点击最小间隔秒，默认 2.0")
    parser.add_argument("--lobby-click-interval-max", type=float, default=2.5, help="LOBBY 点击最大间隔秒，默认 2.5")
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
    click_interval_min = float(args.login_click_interval_min)
    click_interval_max = float(args.login_click_interval_max)
    if click_interval_min > click_interval_max:
        click_interval_min, click_interval_max = click_interval_max, click_interval_min
    click_interval_min = max(0.05, click_interval_min)
    click_interval_max = max(click_interval_min, click_interval_max)
    lobby_click_interval_min = float(args.lobby_click_interval_min)
    lobby_click_interval_max = float(args.lobby_click_interval_max)
    if lobby_click_interval_min > lobby_click_interval_max:
        lobby_click_interval_min, lobby_click_interval_max = lobby_click_interval_max, lobby_click_interval_min
    lobby_click_interval_min = max(0.05, lobby_click_interval_min)
    lobby_click_interval_max = max(lobby_click_interval_min, lobby_click_interval_max)

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
    fatigue_reader = FatigueReader(
        sample_dir=Path(args.fatigue_sample_dir),
        target_res=args.target_res,
        min_confidence=args.fatigue_min_confidence,
        stable_frames=args.fatigue_stable_frames,
    )
    if fatigue_reader.template_count == 0:
        logger.log(
            "E_FATIGUE_OCR_UNAVAILABLE",
            "异常：体力 OCR 后端不可用，请检查 rapidocr-onnxruntime 安装",
            backend=fatigue_reader.ocr_name,
        )
    fatigue_worker = AsyncFatigueWorker(fatigue_reader)
    fatigue_worker.start()
    last_logged_fatigue: Optional[int] = None
    last_fatigue_read_log_at = 0.0
    last_fatigue_infer_at = 0.0
    last_login_click_log_at = 0.0
    last_lobby_click_log_at = 0.0
    last_recharge_click_log_at = 0.0
    login_click_done = 0
    login_click_next_at = 0.0
    login_auto_session_active = False
    lobby_auto_session_active = False
    lobby_click_done = 0
    lobby_click_next_at = 0.0
    recharge_auto_session_active = False
    recharge_click_phase = 0  # 0: wait click2, 1: wait click1, 2: done
    recharge_click_next_at = 0.0
    recharge_close_check_pending = False
    recharge_close_check_at = 0.0
    last_fatigue_result = FatigueResult(
        stable_value=None,
        stable_confidence=0.0,
        candidate_value=None,
        candidate_confidence=0.0,
        roi=(0, 0, 0, 0),
    )
    try:
        click_roi_vals = [float(x.strip()) for x in str(args.login_click_roi).split(",")]
        if len(click_roi_vals) != 4:
            raise ValueError("len!=4")
        login_click_roi = (
            click_roi_vals[0],
            click_roi_vals[1],
            click_roi_vals[2],
            click_roi_vals[3],
        )
    except Exception:
        logger.log(
            "W_LOGIN_CLICK_ROI_INVALID",
            "LOGIN 点击区域参数非法，回退默认值",
            login_click_roi=str(args.login_click_roi),
        )
        login_click_roi = (0.323, 0.367, 0.394, 0.343)
    try:
        lobby_roi_vals = [float(x.strip()) for x in str(args.lobby_click_roi).split(",")]
        if len(lobby_roi_vals) != 4:
            raise ValueError("len!=4")
        lobby_click_roi = (
            lobby_roi_vals[0],
            lobby_roi_vals[1],
            lobby_roi_vals[2],
            lobby_roi_vals[3],
        )
    except Exception:
        logger.log(
            "W_LOBBY_CLICK_ROI_INVALID",
            "LOBBY 点击区域参数非法，回退默认值",
            lobby_click_roi=str(args.lobby_click_roi),
        )
        lobby_click_roi = (0.238, 0.905, 0.130, 0.055)
    try:
        recharge1_roi_vals = [float(x.strip()) for x in str(args.recharge_click1_roi).split(",")]
        if len(recharge1_roi_vals) != 4:
            raise ValueError("len!=4")
        recharge_click1_roi = (
            recharge1_roi_vals[0],
            recharge1_roi_vals[1],
            recharge1_roi_vals[2],
            recharge1_roi_vals[3],
        )
    except Exception:
        logger.log(
            "W_RECHARGE_CLICK1_ROI_INVALID",
            "RECHARGE_1 点击区域参数非法，回退默认值",
            recharge_click1_roi=str(args.recharge_click1_roi),
        )
        recharge_click1_roi = (0.512, 0.708333, 0.154, 0.045)
    try:
        recharge2_roi_vals = [float(x.strip()) for x in str(args.recharge_click2_roi).split(",")]
        if len(recharge2_roi_vals) != 4:
            raise ValueError("len!=4")
        recharge_click2_roi = (
            recharge2_roi_vals[0],
            recharge2_roi_vals[1],
            recharge2_roi_vals[2],
            recharge2_roi_vals[3],
        )
    except Exception:
        logger.log(
            "W_RECHARGE_CLICK2_ROI_INVALID",
            "RECHARGE_2 点击区域参数非法，回退默认值",
            recharge_click2_roi=str(args.recharge_click2_roi),
        )
        recharge_click2_roi = (0.614500, 0.468333, 0.028, 0.034)

    logger.log(
        "I_FSM_STARTED",
        "状态机启动：SPLASH/LOGIN/CONNECTING/LOBBY/RECHARGE",
        threshold=args.threshold,
        lobby_threshold=args.lobby_threshold,
        login_priority_margin=args.login_priority_margin,
        stable_frames=args.stable_frames,
        fatigue_sample_dir=str(args.fatigue_sample_dir),
        fatigue_templates=fatigue_reader.template_count,
        fatigue_ocr_backend=fatigue_reader.ocr_name if fatigue_reader.template_count > 0 else "UNAVAILABLE",
        fatigue_interval=args.fatigue_interval,
        login_click_roi=",".join(f"{x:.6f}" for x in login_click_roi),
        lobby_click_roi=",".join(f"{x:.6f}" for x in lobby_click_roi),
        recharge_click1_roi=",".join(f"{x:.6f}" for x in recharge_click1_roi),
        recharge_click2_roi=",".join(f"{x:.6f}" for x in recharge_click2_roi),
        login_auto_clicks=max(0, int(args.login_auto_clicks)),
        login_click_interval=[click_interval_min, click_interval_max],
        lobby_fatigue_click_threshold=int(args.lobby_fatigue_click_threshold),
        lobby_click_interval=[lobby_click_interval_min, lobby_click_interval_max],
        window_title=args.window_title,
        proc_name=args.proc_name,
    )

    try:
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
                now_ts = time.time()
                if current_state == STATE_RECHARGE:
                    r1x, r1y, r1w, r1h = ratio_roi_to_rect(800, 600, recharge_click1_roi)
                    r2x, r2y, r2w, r2h = ratio_roi_to_rect(800, 600, recharge_click2_roi)
                    cv2.rectangle(frame_bgr, (r1x, r1y), (r1x + r1w, r1y + r1h), (0, 255, 255), 2)
                    cv2.putText(
                        frame_bgr,
                        f"RECHARGE_1 ({r1x},{r1y},{r1w},{r1h})",
                        (r1x, max(20, r1y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                    )
                    cv2.rectangle(frame_bgr, (r2x, r2y), (r2x + r2w, r2y + r2h), (0, 255, 0), 2)
                    cv2.putText(
                        frame_bgr,
                        f"RECHARGE_2 ({r2x},{r2y},{r2w},{r2h})",
                        (r2x, max(20, r2y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                    )
                    if (now_ts - last_recharge_click_log_at) >= 1.0:
                        last_recharge_click_log_at = now_ts
                        s1x = left + int(r1x * (width / 800.0))
                        s1y = top + int(r1y * (height / 600.0))
                        s1w = max(1, int(r1w * (width / 800.0)))
                        s1h = max(1, int(r1h * (height / 600.0)))
                        s2x = left + int(r2x * (width / 800.0))
                        s2y = top + int(r2y * (height / 600.0))
                        s2w = max(1, int(r2w * (width / 800.0)))
                        s2h = max(1, int(r2h * (height / 600.0)))
                        logger.log(
                            "RECHARGE_CLICK_REGIONS",
                            "RECHARGE 可点击区域",
                            recharge_1_frame_800x600={"x": r1x, "y": r1y, "w": r1w, "h": r1h},
                            recharge_1_window_pixels={"x": s1x, "y": s1y, "w": s1w, "h": s1h},
                            recharge_2_frame_800x600={"x": r2x, "y": r2y, "w": r2w, "h": r2h},
                            recharge_2_window_pixels={"x": s2x, "y": s2y, "w": s2w, "h": s2h},
                        )
                    if not recharge_auto_session_active:
                        recharge_auto_session_active = True
                        recharge_click_phase = 0
                        recharge_click_next_at = now_ts + 2.0
                        logger.log(
                            "RECHARGE_AUTO_CLICK_START",
                            "RECHARGE 自动点击开始：2秒后点击 RECHARGE_2 中心",
                            next_at_in_sec=2.0,
                        )
                    if recharge_auto_session_active and recharge_click_phase == 0 and now_ts >= recharge_click_next_at:
                        c2x = r2x + (r2w // 2)
                        c2y = r2y + (r2h // 2)
                        sc2x = left + int(c2x * (width / 800.0))
                        sc2y = top + int(c2y * (height / 600.0))
                        left_click_at_screen(sc2x, sc2y)
                        recharge_click_phase = 1
                        recharge_click_next_at = now_ts + 2.3
                        logger.log(
                            "RECHARGE_AUTO_CLICK_2",
                            "RECHARGE 点击 RECHARGE_2 中心",
                            click_time=datetime.now().astimezone().isoformat(timespec="milliseconds"),
                            frame_800x600={"x": c2x, "y": c2y},
                            window_pixels={"x": sc2x, "y": sc2y},
                            next_at_in_sec=2.3,
                        )
                    elif recharge_auto_session_active and recharge_click_phase == 1 and now_ts >= recharge_click_next_at:
                        c1x = r1x + (r1w // 2)
                        c1y = r1y + (r1h // 2)
                        press_enter_key()
                        recharge_click_phase = 2
                        logger.log(
                            "RECHARGE_AUTO_CLICK_1",
                            "RECHARGE 第二阶段改为按下 Enter",
                            click_time=datetime.now().astimezone().isoformat(timespec="milliseconds"),
                            frame_800x600={"x": c1x, "y": c1y},
                            action="press_enter",
                        )
                        recharge_close_check_pending = True
                        recharge_close_check_at = now_ts + 5.0
                        logger.log(
                            "RECHARGE_CLOSE_CHECK_SCHEDULED",
                            "RECHARGE 完成后等待5秒检查是否进入 CONNECTING",
                            check_after_sec=5.0,
                        )
                        logger.log(
                            "RECHARGE_AUTO_CLICK_DONE",
                            "RECHARGE 自动点击流程完成",
                            steps=2,
                        )
                elif recharge_auto_session_active:
                    logger.log(
                        "RECHARGE_AUTO_CLICK_STOP",
                        "RECHARGE 自动点击停止（状态已切换）",
                        phase=recharge_click_phase,
                        to_state=current_state,
                    )
                    recharge_auto_session_active = False
                    recharge_click_phase = 0
                    recharge_click_next_at = 0.0
                    recharge_close_check_pending = False
                    recharge_close_check_at = 0.0
                if current_state == STATE_LOBBY:
                    bx, by, bw, bh = ratio_roi_to_rect(800, 600, lobby_click_roi)
                    cv2.rectangle(frame_bgr, (bx, by), (bx + bw, by + bh), (0, 255, 255), 3)
                    cv2.putText(
                        frame_bgr,
                        f"LOBBY_CLICK ({bx},{by},{bw},{bh})",
                        (bx, max(20, by - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )
                    if (now_ts - last_lobby_click_log_at) >= 1.0:
                        last_lobby_click_log_at = now_ts
                        sx = left + int(bx * (width / 800.0))
                        sy = top + int(by * (height / 600.0))
                        sw = max(1, int(bw * (width / 800.0)))
                        sh = max(1, int(bh * (height / 600.0)))
                        logger.log(
                            "LOBBY_CLICK_REGION",
                            "LOBBY 可点击区域",
                            frame_800x600={"x": bx, "y": by, "w": bw, "h": bh},
                            window_pixels={"x": sx, "y": sy, "w": sw, "h": sh},
                        )
                if current_state == STATE_LOGIN:
                    lx, ly, lw, lh = ratio_roi_to_rect(800, 600, login_click_roi)
                    cv2.rectangle(frame_bgr, (lx, ly), (lx + lw, ly + lh), (0, 255, 255), 3)
                    cv2.putText(
                        frame_bgr,
                        f"LOGIN_CLICK ({lx},{ly},{lw},{lh})",
                        (lx, max(20, ly - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )
                    if (now_ts - last_login_click_log_at) >= 1.0:
                        last_login_click_log_at = now_ts
                        sx = left + int(lx * (width / 800.0))
                        sy = top + int(ly * (height / 600.0))
                        sw = max(1, int(lw * (width / 800.0)))
                        sh = max(1, int(lh * (height / 600.0)))
                        logger.log(
                            "LOGIN_CLICK_REGION",
                            "LOGIN 可点击区域",
                            frame_800x600={"x": lx, "y": ly, "w": lw, "h": lh},
                            window_pixels={"x": sx, "y": sy, "w": sw, "h": sh},
                        )

                    # LOGIN 自动点击：进入状态后执行固定次数，点击位置与间隔均随机。
                    if not login_auto_session_active:
                        login_auto_session_active = True
                        login_click_done = 0
                        login_click_next_at = now_ts + random.uniform(click_interval_min, click_interval_max)
                        logger.log(
                            "LOGIN_AUTO_CLICK_START",
                            "LOGIN 自动点击开始",
                            target_clicks="until_state_change",
                            interval_min=click_interval_min,
                            interval_max=click_interval_max,
                        )

                    if (
                        now_ts >= login_click_next_at
                        and lw > 2
                        and lh > 2
                    ):
                        rx = random.randint(lx + 1, lx + lw - 1)
                        ry = random.randint(ly + 1, ly + lh - 1)
                        sx = left + int(rx * (width / 800.0))
                        sy = top + int(ry * (height / 600.0))
                        left_click_at_screen(sx, sy)
                        login_click_done += 1
                        click_dt = datetime.now().astimezone().isoformat(timespec="milliseconds")
                        logger.log(
                            "LOGIN_AUTO_CLICK",
                            f"LOGIN 自动点击 #{login_click_done}",
                            click_time=click_dt,
                            frame_800x600={"x": rx, "y": ry},
                            window_pixels={"x": sx, "y": sy},
                        )
                        login_click_next_at = now_ts + random.uniform(click_interval_min, click_interval_max)
                else:
                    if login_auto_session_active:
                        logger.log(
                            "LOGIN_AUTO_CLICK_STOP",
                            "LOGIN 自动点击停止（状态已切换）",
                            clicked=login_click_done,
                            to_state=current_state,
                        )
                        login_auto_session_active = False
                        login_click_done = 0
                        login_click_next_at = 0.0
                if current_state != STATE_LOBBY and lobby_auto_session_active:
                    logger.log(
                        "LOBBY_AUTO_CLICK_STOP",
                        "LOBBY 自动点击停止（状态已切换）",
                        clicked=lobby_click_done,
                        to_state=current_state,
                    )
                    lobby_auto_session_active = False
                    lobby_click_done = 0
                    lobby_click_next_at = 0.0

                if recharge_close_check_pending and now_ts >= recharge_close_check_at:
                    recharge_close_check_pending = False
                    recharge_close_check_at = 0.0
                    if current_state != STATE_CONNECTING:
                        stopped = terminate_processes_by_name(args.proc_name)
                        logger.log(
                            "RECHARGE_CLOSE_GAME",
                            "5秒后未进入 CONNECTING，关闭游戏进程",
                            current_state=current_state,
                            proc_name=args.proc_name,
                            terminated_count=stopped,
                        )
                    else:
                        logger.log(
                            "RECHARGE_CLOSE_GAME_CANCEL",
                            "5秒检查通过：已进入 CONNECTING，不关闭游戏",
                            current_state=current_state,
                        )

                should_read_fatigue = current_state == STATE_LOBBY
                if should_read_fatigue:
                    if (now_ts - last_fatigue_infer_at) >= max(0.05, float(args.fatigue_interval)):
                        fatigue_worker.submit(frame_bgr)
                        last_fatigue_infer_at = now_ts
                    latest = fatigue_worker.latest()
                    if latest.roi != (0, 0, 0, 0) or latest.candidate_value is not None or latest.stable_value is not None:
                        last_fatigue_result = latest
                    fatigue = last_fatigue_result
                else:
                    fatigue = FatigueResult(
                        stable_value=None,
                        stable_confidence=0.0,
                        candidate_value=None,
                        candidate_confidence=0.0,
                        roi=(0, 0, 0, 0),
                    )

                fx, fy, fw, fh = fatigue.roi
                if should_read_fatigue and fw > 0 and fh > 0:
                    cv2.rectangle(frame_bgr, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 1)
                fatigue_text = "N/A"
                fatigue_conf = 0.0
                fatigue_color = (200, 200, 200)
                if not should_read_fatigue:
                    fatigue_text = "-"
                if fatigue.stable_value is not None:
                    fatigue_text = str(fatigue.stable_value)
                    fatigue_conf = fatigue.stable_confidence
                    fatigue_color = (80, 255, 120)
                elif fatigue.candidate_value is not None:
                    fatigue_text = f"{fatigue.candidate_value}?"
                    fatigue_conf = fatigue.candidate_confidence
                    fatigue_color = (0, 210, 255)
                cv2.putText(
                    frame_bgr,
                    f"FATIGUE={fatigue_text} conf={fatigue_conf:.3f}",
                    (10, 108),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    fatigue_color,
                    2,
                )
                if fatigue.stable_value is not None and fatigue.stable_value != last_logged_fatigue:
                    last_logged_fatigue = fatigue.stable_value
                    logger.log(
                        "FATIGUE_UPDATE",
                        f"体力更新：{fatigue.stable_value}",
                        state=current_state,
                        fatigue=fatigue.stable_value,
                        confidence=round(float(fatigue.stable_confidence), 4),
                    )
                if should_read_fatigue and (now_ts - last_fatigue_read_log_at) >= 2.0:
                    last_fatigue_read_log_at = now_ts
                    if fatigue.stable_value is not None:
                        logger.log(
                            "FATIGUE_READ",
                            f"体力识别：{fatigue.stable_value}",
                            state=current_state,
                            mode="stable",
                            fatigue=fatigue.stable_value,
                            confidence=round(float(fatigue.stable_confidence), 4),
                        )
                    elif fatigue.candidate_value is not None:
                        logger.log(
                            "FATIGUE_READ",
                            f"体力识别候选：{fatigue.candidate_value}",
                            state=current_state,
                            mode="candidate",
                            fatigue=fatigue.candidate_value,
                            confidence=round(float(fatigue.candidate_confidence), 4),
                        )
                    else:
                        logger.log(
                            "FATIGUE_READ",
                            "体力识别：N/A",
                            state=current_state,
                            mode="none",
                        )

                if current_state == STATE_LOBBY:
                    fatigue_for_click: Optional[int] = None
                    fatigue_mode = "none"
                    fatigue_conf_for_click = 0.0
                    if fatigue.stable_value is not None:
                        fatigue_for_click = int(fatigue.stable_value)
                        fatigue_mode = "stable"
                        fatigue_conf_for_click = float(fatigue.stable_confidence)
                    elif fatigue.candidate_value is not None:
                        fatigue_for_click = int(fatigue.candidate_value)
                        fatigue_mode = "candidate"
                        fatigue_conf_for_click = float(fatigue.candidate_confidence)

                    bx, by, bw, bh = ratio_roi_to_rect(800, 600, lobby_click_roi)
                    can_lobby_click = (
                        fatigue_for_click is not None
                        and fatigue_for_click >= int(args.lobby_fatigue_click_threshold)
                        and bw > 2
                        and bh > 2
                    )
                    if can_lobby_click and not lobby_auto_session_active:
                        lobby_auto_session_active = True
                        lobby_click_done = 0
                        lobby_click_next_at = now_ts + random.uniform(lobby_click_interval_min, lobby_click_interval_max)
                        logger.log(
                            "LOBBY_AUTO_CLICK_START",
                            "LOBBY 自动点击开始（体力满足阈值）",
                            fatigue=fatigue_for_click,
                            fatigue_mode=fatigue_mode,
                            confidence=round(fatigue_conf_for_click, 4),
                            threshold=int(args.lobby_fatigue_click_threshold),
                            interval_min=lobby_click_interval_min,
                            interval_max=lobby_click_interval_max,
                        )
                    elif (not can_lobby_click) and lobby_auto_session_active:
                        logger.log(
                            "LOBBY_AUTO_CLICK_PAUSE",
                            "LOBBY 自动点击暂停（体力不足或识别失败）",
                            clicked=lobby_click_done,
                            fatigue=fatigue_for_click,
                            fatigue_mode=fatigue_mode,
                            threshold=int(args.lobby_fatigue_click_threshold),
                        )
                        lobby_auto_session_active = False
                        lobby_click_next_at = 0.0

                    if can_lobby_click and lobby_auto_session_active and now_ts >= lobby_click_next_at:
                        rx = random.randint(bx + 1, bx + bw - 1)
                        ry = random.randint(by + 1, by + bh - 1)
                        sx = left + int(rx * (width / 800.0))
                        sy = top + int(ry * (height / 600.0))
                        left_click_at_screen(sx, sy)
                        lobby_click_done += 1
                        click_dt = datetime.now().astimezone().isoformat(timespec="milliseconds")
                        logger.log(
                            "LOBBY_AUTO_CLICK",
                            f"LOBBY 自动点击 #{lobby_click_done}",
                            click_time=click_dt,
                            fatigue=fatigue_for_click,
                            fatigue_mode=fatigue_mode,
                            frame_800x600={"x": rx, "y": ry},
                            window_pixels={"x": sx, "y": sy},
                        )
                        lobby_click_next_at = now_ts + random.uniform(lobby_click_interval_min, lobby_click_interval_max)

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
    finally:
        fatigue_worker.stop(timeout=2.0)
        cv2.destroyAllWindows()

    logger.log("I_FSM_STOPPED", "状态机已结束")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
