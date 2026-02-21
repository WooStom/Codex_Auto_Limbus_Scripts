# limbuscompany_Scripts3

目前提供一个最小版本日志系统：
- 运行后先检测 Steam 进程
- 检测到 Steam 后请求启动游戏
- 若未检测到 Steam，记录异常并结束程序
- 持续监控游戏进程的启动和关闭
- 当检测到游戏关闭时，监控自动结束（日志窗口随之关闭）

## 文件

- `game_lifecycle_monitor.py`：生命周期监控脚本（启动/关闭）
- `splash_login_fsm.py`：SPLASH/LOGIN 识别 + 状态机 + 实时画框
- `start_splash_login_fsm.bat`：SPLASH/LOGIN 状态机启动脚本（conda limbus）
- `start_monitor_and_fsm.bat`：一键同时启动 monitor + fsm
- `run_monitor_and_fsm.py`：单窗口聚合 monitor + fsm 日志输出
- `runtime/logs/limbus_lifecycle.jsonl`：机器可读日志
- `runtime/logs/limbus_lifecycle.log`：中文可读日志

## 依赖

```bash
pip install psutil
pip install opencv-python mss numpy
```

## 运行

```bash
python game_lifecycle_monitor.py
```

Windows 可直接双击：

```bat
start_lifecycle_monitor.bat
```

SPLASH/LOGIN 状态机可直接双击：

```bat
start_splash_login_fsm.bat
```

一键同时启动 monitor + fsm：

```bat
start_monitor_and_fsm.bat
```

说明：`start_monitor_and_fsm.bat` 现在只开一个日志窗口，日志会带 `[MONITOR]` / `[FSM]` 前缀。

可选参数：

```bash
python game_lifecycle_monitor.py --proc-name LimbusCompany.exe --poll 1.0
```

指定 Steam 参数：

```bash
python game_lifecycle_monitor.py --steam-name steam.exe --steam-app-id 1973530
```

如果你想在游戏关闭后继续保留监控，可加：

```bash
python game_lifecycle_monitor.py --keep-window-after-close
```

SPLASH/LOGIN 状态机（你的当前配置）：

```bash
python splash_login_fsm.py --window-title LimbusCompany --target-res 800_600 --threshold 0.80 --stable-frames 3
```

默认行为：游戏进程关闭后，monitor 与 fsm 都会自动退出并关闭各自窗口。

按 `Ctrl+C` 停止监控。
