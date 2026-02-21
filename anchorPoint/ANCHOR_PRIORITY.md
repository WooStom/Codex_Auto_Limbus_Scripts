# 锚点权限等级标准

本文档是 `anchorPoint` 的唯一优先级标准。  
后续新增锚点与状态机冲突处理，统一按本文件执行。

## 1. 权限等级（高 -> 低）

| 等级 | 状态 | 说明 | 当前阈值 |
|---|---|---|---|
| P5 | CONNECTING | 最高优先级，命中即抢占状态 | 0.80 |
| P4 | RECHARGE | 高优先级（保留位，后续接入） | 0.80 |
| P3 | LOBBY | 主界面稳定态 | 0.75 |
| P2 | LOGIN | 登录态 | 0.80 |
| P1 | SPLASH | 启动态 | 0.80 |
| P0 | UNKNOWN | 无有效锚点时回退态 | - |

## 2. 冲突处理规则

1. 先按权限等级比较，等级高者优先。
2. 同等级下，按匹配分数 `score` 高者优先。
3. 若所有状态都低于各自阈值，状态机强制为 `UNKNOWN`，并且不显示框。
4. `CONNECTING`（P5）属于抢占态：达到阈值时立即切换，不等待稳定帧。

## 3. 命名规则

- 文件名格式：`<STATE>_<WIDTH>_<HEIGHT>_<INDEX>.png`
- 例子：
  - `LOBBY_800_600_1.png`
  - `CONNECTING_800_600_1.png`

## 4. 当前锚点清单（已存在）

### CONNECTING
- `CONNECTING_800_600_1.png`

### RECHARGE
- `RECHARGE_800_600_1.png`（已入标准，脚本待接入）

### LOBBY
- `LOBBY_800_600_1.png`
- `LOBBY_800_600_2.png`
- `LOBBY_800_600_3.png`

### LOGIN
- `LOGIN_800_600_1.png`
- `LOGIN_800_600_2.png`
- `LOGIN_800_600_3.png`
- `LOGIN_1920_1009_1.png`
- `LOGIN_1920_1009_2.png`
- `LOGIN_1920_1009_3.png`

### SPLASH
- `SPLASH_800_600_1.png`
- `SPLASH_800_600_2.png`
- `SPLASH_1920_1009_1.png`
- `SPLASH_1920_1009_2.png`

