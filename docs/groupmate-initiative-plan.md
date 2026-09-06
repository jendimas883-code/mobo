# mobo 主动群友计划（偏好 · 主动性）

> 版本 v3（oracle 冷审两轮 + 评审团复审后定稿，修订记录见文末附）。前置文档：`docs/maibot-port-plan.md`（五阶段已全部交付）。
> 定位升级：从"能不违和地待在群里"升级为"**有偏好、会主动的群友**"——对不同用户有可见的好恶、
> 会避开讨厌的话题、会在冷场时主动开话题。自由文本印象与主动点名 @ 列为**预留阶段**，
> 默认不排期，凭上线观察数据决定是否执行（见 §5、§6）。

## 1. 背景与目标

### 1.1 现状基线（2026-09-05，主分支 51a3756，关键点已逐一到代码核实）

| 机制 | 位置 | 现状 |
|---|---|---|
| 四维关系（熟悉/信任/热情/疲劳） | `app/cognition.py:44-151`，表 `relationships` 主键 `(guild_id, user_id)` | **只在 direct 消息路径更新**（`discord_bot.py:1622-1629`）；主动回复（非 direct）不积累关系 |
| 关系文本注入 | `Relationship.description`（`cognition.py:28-41`）→ `ContextBuilder.build` :396-402 | 措辞是标量描述（"陌生/熟悉、冷淡/热情、信任、疲劳"），**无好恶态度语气**；公私两域都注入 |
| 心情 | `MoodService` `cognition.py:225-323` | 已注入 prompt（:527），不影响文风；关闭时回落基线值（:410-416） |
| 兴趣偏好 | 表 `bot_preferences`；`upsert` 已 `clamp(weight, -1, 1)`（`cognition.py:192`） | **负权重今天就能存**，但 `interest_for` 取 `max(weight)`（:221）使负值永远输给正值；学习分支只正增；`list(5)` 按 `weight DESC`（:157-159）——负行会被正值挤出渲染 |
| 主动发言决策 | `ProactiveService.decide` `app/behavior.py:189-313` | 纯响应式（别人发消息后才评估）；默认关；预算由 `_reserve_channel_slot`（:152-187）在 `BEGIN IMMEDIATE` 事务内原子记账进 `proactive_log`，**冷却/日限查询不按 reason 过滤** |
| 主动日志 | 表 `proactive_log(guild_id, channel_id, reason, created_at)` | reason 为自由文本，即事实上的类型字段（现有值："闸门(...)"、"偏好话题：…"、"自然参与"） |
| 出站 @ | `_send_public_reply` `discord_bot.py:1558-1591` | 首片可精确 ping `mention_user_ids`，其余片 `AllowedMentions.none()`；**签名收 `discord.Message` 并走 `source.reply`**；名字→ID 解析器不存在 |
| 消息持久化 | `messages` 表，索引 `(guild_id, channel_id, id DESC)`，含 `expires_at`；`role` CHECK 限 `('user','assistant','system')`（`database.py:76`） | 原始保存受 `save_raw_messages` 开关控制（`discord_bot.py:1105`） |
| 设置面板 | 声明式 `SettingField`（`runtime.py:14-28`），按 section 自动渲染 | 新增配置键零模板工作 |
| 死代码 | `ProactiveService.record`（`behavior.py:336-341`） | 无调用方，本期删除 |

### 1.2 不变量（沿用 port 计划，全部继续有效）

单进程单 SQLite 写者；所有新行为有界（限额/冷却/token 预算/kill switch）；**拟人化在安全检查
之前、安全检查在发送之前完成**（与现行代码一致：`:1380` humanize → `:1388` 逐片 safety）；
用户侧零新命令（好恶全部被动习得，管理台是运营者界面）；DM 上下文不进公开域；
新上下文一律按不可信数据包裹。

### 1.3 对 port 计划非目标的修订

"心情→文风强调制"不再是非目标：以提示词级软指令交付（Phase A 内，无独立开关，见 §4 A5）。

## 2. 总体决策

1. **标量优先**：偏好表达先用现成的关系四维标量（改 `description` 措辞 + 公开路径观测），
   不引入任何 LLM 生成的状态。自由文本印象留作预留 Phase C，有证据再上。
2. **负权重代替极性列**：`bot_preferences.weight` 本就 clamp 到 [-1,1]，避开话题 = 负权重。
   零迁移、零 API 参数新增，只改语义与两处行为。
3. **reason 前缀代替日志列**：心流日志用 `reason='flow:…'` 稳定前缀查询，不加 kind/user_id 列。
4. **心流复用回复路径原语**：话题生成是单次 utility 模型调用，下游 humanize → 逐片 safety →
   发送与回复路径同一套函数；预算与回复主动**共享同一池**（同一个 `_reserve_channel_slot`
   事务），从机制上排除双重记账。
5. **最少旋钮**：本期只新增 2 个配置键（`flow_enabled`、`flow_probability`）。
   其余阈值（冷场判定、冷却、窗口）作为模块常量硬编码，等遥测证明需要再配置化。
   所有行为默认关（`flow_enabled=False`）。

## 3. Phase A — 偏好可见化（0.5–1 天）

**目标**：bot 对不同用户表现出可见的喜欢/不喜欢；能避开讨厌的话题。零迁移、零新配置键。

- **A1 行为化 `Relationship.description`**（`cognition.py:28-41`）：把标量描述改写为
  **行为倾向**语气——只描述倾向，不声称情感（标量编码的是互动历史，不是感情；措辞禁用
  "喜欢/讨厌/烦"等情感断言词）：familiarity+warmth 高 → "和这位用户聊天很顺，你会更愿意
  接他的话头"；fatigue 高或 warmth 低 → "和他互动让你有点累，倾向简短回应、少接他的梗"；
  中间档保持中性。只映射标量，不生成新信息。**该文本同时注入公开域与 DM 域 prompt，
  两种 scope 下措辞都必须通顺**（测试覆盖两域）。
- **A2 公开路径关系观测**（`discord_bot.py:_learn_after_success`，:1622-1629 的 else 分支）：
  `not payload.direct` 且 `relationship_enabled` 时也调 `relationships.observe`，learning_rate
  固定取 direct 的 0.5（模块常量），且**只累积 familiarity**——observe 增加
  `familiarity_only` 参数（或轻量方法），把 warmth/trust/fatigue 的增量置 0，事务结构不变。
  理由与护栏：主动回复目前完全不积累关系，不补此路径则 bot 只会"偏爱"@过它的人；但 observe
  的正面/敌意词启发式在公开闲聊里会误伤（第三人对第三人说"滚"会被记成对 bot 的敌意），
  而偏好差异化需要的正是 familiarity——warmth/trust/fatigue 仍只在 direct 互动中演化。
  `interaction_count` **照常递增**（Phase C 的"互动 ≥5 次"门槛与管理台依赖它）；
  `last_interaction_at` 更新不扭曲另外三维的衰减（`_from_row` 每次写入物化衰减值）。
  写入走 observe 原有原子事务，只有 bot 实际回复事件才触发，量级有界。
- **A3 负权重语义**（`cognition.py:interest_for`），完整契约分三面：
  - **聚合**：消息同时匹配正负话题 → **负值优先**（取最负值）——回避由管理员显式设置，
    优先级高于被动学到的兴趣；多个正值仍取 max；无匹配 → 0。
  - **生效面**：负值经 `probability *= max(0.1, 0.7 + interest)` 把主动概率压到 0.1 因子下限；
    负值话题不进心流上下文（B4 只取正值话题）；prompt 中列为"想避开的话题"（A4）。
    直连路径（被 @ / reply / 对话窗内）不经此因子——被点名仍必须回应；**gate 强制回复
    （`score ≥ gate_threshold`）同样不查 probability，avoid 不压它**——这是有意语义，
    avoid 只治理"主动搭话"的概率路径。
  - **学习**：负权重行不参与被动学习，跳过在 **SQL 层**实现（`UPDATE … WHERE id = ? AND
    locked = 0 AND weight >= 0`），防止与管理台并发编辑竞态——即负权重等价于管理员所有；
    将来若要"被动学会回避"，单独立项。
- **A4 提示词**（`cognition.py:480-484`）：`preference_text` 拆两行——
  "较偏好的话题：…" / "想避开的话题：…"。**负值行单独查询**（`weight < 0 ORDER BY
  weight ASC LIMIT 3`），不依赖 `list(5)` 的 `weight DESC` 排序——否则 ≥5 个正值话题时
  避开行永远不渲染。
- **A5 心情→文风软指令**（`cognition.py:527` 之后、privacy/intent 行之后）：按 mood 数值
  映射表追加一行**倾向性描述**（非指令）：valence < -0.3 → "情绪低落，语气自然收敛、
  少用表情"；energy > 0.5 → "兴致高，可以更活泼"；valence > 0.5 → "心情好，语气放松"。
  `mood_enabled=False` 时按基线值推导（或不渲染与"情绪功能已关闭"矛盾的行）。纯映射表
  ~10 行，无模型调用，无独立开关。
- **管理台**：`behavior.html` 偏好面板的权重输入允许负值并加一句提示
  （"负值 = 想避开的话题"）；`web.py POST /api/preferences` 无需改（clamp 已支持）。
- **清理**：删除死代码 `ProactiveService.record`。
- **测试**（`tests/test_phase6_preference.py`）：description 各档映射（**公私两域**）、
  负值优先规则、SQL 层学习跳过（含并发编辑语义）、>5 正值行时避开行仍渲染、双行注入、
  公开路径 observe（familiarity-only：warmth/trust/fatigue 不动、familiarity 与
  interaction_count 增长、learning_rate 减半与 decay 交互）、文风映射行（含 mood 关闭回基线）。

## 4. Phase B — 心流：冷场主动开话题（1–1.5 天）

**目标**：频道"最近活跃但现在冷场"时，bot 低频、有预算地抛出**从上下文里长出来**的话题。

- **B1 循环**（`discord_bot.py`，仿 `maintenance` :2094 形态）：`@tasks.loop(minutes=15)`
  + `before_loop` 等 ready，`setup_hook` 启动，**`close()` 增加取消本循环**（仿 :2120-2123，
  与 maintenance 一并取消），**`@flow.error` 处理器记日志**，tick 开头检查 `_closing`。
  **tick 内整体 try/except + log.warning**（现维护循环无错误处理，本循环不能复制这个缺陷——
  未捕获异常会让 discord.py 静默停掉循环）。`flow_enabled=False` 时 tick 零开销返回。
  不复用 6 小时维护循环：6h 节奏无法探测"冷场 20 分钟"。
- **B2 资格判定**（每 tick 至多选 1 个频道，模块常量硬编码）：`proactive_global_enabled`
  开启（总 kill switch 对 flow 同样生效）；频道 `listen_enabled AND proactive_enabled`；
  非安静时段（复用 `ProactiveService._in_quiet_hours`）；活动查询**只数
  `role='user'`**（注意 `role` 列 CHECK 约束限 user/assistant/system，勿照抄
  `behavior.py:247` 永不匹配的 `role='bot'` 过滤）：走既有复合索引取回尾部 60 条后在
  Python 内过滤 `expires_at` 与时间窗（有效条不足 20 就少用，不追加查询）——最近 6h
  ≥ 10 条 且 最后一条用户消息距今 > 20 分钟（"聊过但冷了"）；mood social_budget ≥
  基线 × 0.8；token 日软预算未触顶；掷 `flow_probability`（默认 0.15）。
  **前提：`save_raw_messages=True`**——写入 `flow_enabled` 的帮助文本；tick 检测到原始
  保存关闭则记一条 log 并跳过（flow 永不触发的可诊断性）。
- **B3 共享预算**：走 `_reserve_channel_slot` 同一 `BEGIN IMMEDIATE` 事务（加 reason 前缀
  参数），心流行计入**同一个** `proactive_daily_limit` 池；心流专属冷却 = 距上一条
  `reason LIKE 'flow:%'` ≥ 120 分钟（常量），在同一事务内校验。无独立日志表、无双重记账。
  **显式契约（全部测试锁定）**：
  (a) flow 行同样重启通用 `proactive_cooldown_minutes` 窗口——flow 之后的 45 分钟内
  回复主动不会再触发，这是挤占契约的一部分；
  (b) flow 自身也必须通过通用冷却检查；
  (c) 槽位在生成前保留：输出被丢弃（hook 非子串 / 安全拦截 / 空输出）**仍消耗槽位**——
  一次失败生成 = 120 分钟 flow 静默，接受该代价，不删已记账行。
- **B4 话题生成**（新私有方法，单次 utility 调用，`≤300` token 上限，**调用必须走
  `usage.record`**——否则 token 对 `daily_soft_token_budget` 不可见）：输入 = 该频道摘要
  （`MemoryService.channel_summary`）+ 最近 ≤20 条有效消息（B2 的取回结果）+ 偏好话题
  （正值）+ open_loops（**过滤 `public_safe` 且限同 guild**——DM follow-up 不得漏进
  公开话题）；全部按不可信数据包裹。**输出为结构化 JSON：
  `{"hook": "<所供上下文中的原文引用>", "text": "<≤80 字开场白>"}`**；`hook` 不是摘要或
  最近消息的逐字子串即整体丢弃——由头判据是机械的、可测试的（测试喂伪造 hook，断言
  不发送、不记账）。
- **B5 发送**：`humanize_fragments` → **逐片** `safety.check_output` → `typing_delay` →
  发送（顺序与回复路径一致：拟人化在安全检查之前）。**发送层**：把 `_send_public_reply`
  的碎片发送循环抽为可复用 helper；flow 没有入站消息，以 `channel.send` +
  `AllowedMentions.none()` 走同一 helper（不经 `source.reply`）。flow 开场白按回复
  同一路径持久化（`discord_bot.py:1443-1458` 的 `_save`/`_remember` 调用），后续 tick
  的上下文与回复追踪才能看到它。
- **B6 记账**：`reason='flow:话题：<hook 摘要>'`。**不调 `mood.observe`**——回复路径观察
  的是用户消息，flow 观察自己的输出会让正/敌意词启发式自我强化情绪。
- **配置**（section `主动发言/回复决策`）：`flow_enabled`(False，帮助文本写明
  `save_raw_messages` 前提)、`flow_probability`(0.15)。
- **测试**（`tests/test_phase6_flow.py`）：B2 各条件正反例（含 `proactive_global_enabled`
  关闭、`save_raw_messages=False` 跳过且记日志）、role 过滤只数用户消息、伪造 hook 丢弃
  不发送不记账、hook 子串通过、共享池记账（回复+心流合计不超日限）、(a) flow 行重启通用
  冷却、(b) flow 过通用冷却、(c) 丢弃仍消耗槽位、事务内 flow 冷却、安静时段、
  open_loops 过滤（DM 来源不进上下文）、usage 记账、异常后循环存活、上下文不可信包裹、
  flow 输出持久化后下一 tick 可见。

## 5. 预留 Phase C — 自由文本印象（默认不排期）

**触发条件**：M2 上线观察 ≥1 周后，若标量态度（A1）被证明表达力不足（例如群友反馈
"它对谁都一个样"、或审计发现它对讨厌的人照聊不误），才执行本阶段。
规格存档（避免将来重新推导）：

- `relationships` 加 `impression TEXT` + `impression_updated_at TEXT` 两列；
- `RelationshipService.refresh_impression`：utility 模型从**该 guild 公开频道上下文**
  （DM 内容一律不参与，长度 ≤120 字）改写印象，24h 冷却、interaction_count 每 +10
  触发一次（不做概率掷；interaction_count 自 Phase A 起含公开路径递增）；
- 注入 `【当前内部状态】`（`cognition.py:526` 后）："- 你对他的印象：…"
  （bot 侧态度行，公开域可见性需按 A2 同样的共享段逻辑证明不泄漏无关用户）；
- 配置键仅 1 个：`impression_enabled`(False)。

## 6. 预留 Phase D — 主动点名 @（默认不排期）

**触发条件**：M2 心流数据证明"开话题"形态成立（命中率可接受、无人反感反馈）后才做。
规格存档：

- 白名单解析器：仅从 prompt 列出的近期参与者显示名中唯一前缀匹配，歧义/无匹配 → 纯文本不 ping；
- 硬约束：目标需近 2h 内发言过且 familiarity+warmth ≥ 0.4（常量）、每目标 24h 冷却、
  每 guild 日限 3、每条至多 1 个；配额与审计记 `audit_log`（复用 bot_bridge 审计形态），
  `proactive_log.reason='ping:<目标显示名>'`，不加列；
- 实现细节（名字→`<@id>` 归一化与 safety/humanize 的先后、发送层按碎片放宽
  allowed_mentions、模型伪造 ID 的消毒）在执行期实现计划中定稿——本节只是决策门与约束，
  不是可执行规格；
- 配置键仅 2 个：`ping_enabled`(False)、`ping_daily_limit`(3)。

## 7. 风险与对策

| 风险 | 对策 |
|---|---|
| 心流变成客服推送感 | hook 子串判据缺失即丢弃；"聊过但冷了"双条件；与回复共享日限池；`flow_enabled` kill switch |
| 模型幻觉话题 | 结构化 hook 必须为上下文逐字子串；上下文按不可信包裹；输出过安全引擎 |
| token 成本漂移 | utility 模型 + 300 token 上限 + `usage.record` 入账 + 日软预算复用 + 概率节流 |
| 公开路径关系观测改变衰减语义 | familiarity-only 参数把三维增量置 0；仅 bot 回复事件触发；Phase A 测试覆盖 decay 交互 |
| 15 分钟循环静默死亡 | tick try/except + `@flow.error` 处理器 + 日志（修正维护循环的既有缺陷） |
| messages 表活动查询慢 / 语义错 | 索引尾部取 60 条 Python 判窗；只数 `role='user'`；不照抄永不匹配的 `role='bot'` 过滤 |
| `save_raw_messages=False` 时 flow 永不触发 | 前提写进帮助文本；tick 检测并记日志（可诊断） |
| A1 行为化措辞被截图放大 | 映射表只输出"少接话/简短回应"级措辞，禁用情感断言与敌意词；负向表达 = 概率与长度，不是攻击性内容 |

## 8. 执行顺序与里程碑

- **M1 = Phase A**：偏好可见 + 话题避开上线，管理台可运营。可独立交付。
- **M2 = Phase B**：心流灰度（先单频道手动开 `proactive_enabled` + `flow_enabled`），观察 ≥1 周。
- **M3 = 决策点**：按 M2 遥测决定是否执行预留 Phase C / D（默认都不执行）。
- 执行走 deepwork（进度文件 `.deepwork/groupmate-initiative.md`），每阶段结束送 oracle 审。

## 9. 明确不做（非目标）

- 向量库 / embedding / FTS5（维持 port 计划口径）
- 主动私聊/DM 发起、定时私信、"提醒我"类功能
- 对不喜欢的用户演出敌意（不喜欢的表达 = 降概率 + 简短回应 + 不接梗）
- 好恶的手动指派（管理员只能管理兴趣话题正负权重；对人的态度只能被动习得）
- 心情→文风的机制级强制（只做提示词软指令，无独立开关）
- 用户侧任何新命令；多进程化；流式输出；本期新增超过 2 个的常驻配置键

## 附: oracle 冷审修订记录（v1→v2）

**采纳**：自由文本印象移出本期（→ 预留 C，标量态度先行的替代方案成立，`description`
本就由标量派生）；主动点名 @ 移出本期（→ 预留 D，风险最高、收益待心流验证）；
`polarity` 列 → 负权重（`upsert` 已 clamp [-1,1]，`cognition.py:192` 核实属实）；
`proactive_log` 不加 kind/user_id 列（reason 前缀 + audit 元数据足够）；
配置键 ~20 → 2（阈值硬编码为模块常量）；`mood_style_enabled` 键删除（硬编码开）；
文档/清理并入阶段门而非独立 Phase；心流循环必须补错误处理；活动查询限定索引尾部取回；
flow 上下文按不可信包裹；文档与测试工时计入各阶段估算。

**保留（附理由）**：
1. **公开路径 `relationship.observe`**（oracle 建议删）——保留。主动回复目前完全不积累关系，
   不补此路径则 bot 只会"偏爱"@过它的人，用户核心诉求（对不同群友有不同态度）无法成立。
   已加界：仅 bot 回复事件触发、learning_rate 减半、原子事务不变。
2. **专用 15 分钟 `tasks.loop`**（oracle 建议考虑复用 6h 维护循环）——保留专用循环。
   6h 节奏无法实现"冷场 20 分钟"检测；模板已存在，边际成本是一个声明；采纳了 oracle
   对错误处理的要求作为前置条件。
3. **点名 @ 未整体删除而是转预留**——用户在需求陈述中明确点名（"甚至 at 某些用户"），
   完整规格保留在 §6 并明确默认不排期、以 M2 遥测为执行门槛，避免将来重新推导设计。

**第二轮复核（v2 定稿）**：三个保留点均判 defensible（A2 需护栏、B1 无条件成立、
D 降级为纯决策门文档）。按复核意见收尾：A2 加 familiarity-only 护栏（公开闲聊中
正面/敌意词启发式会误伤第三人的对话，warmth/trust/fatigue 回归 direct-only）；
A1 措辞改为行为倾向、禁用情感断言词；A3 聚合/生效/学习三面契约显式化（负值优先限
管理员设置的回避，负权重行不参与学习 = 有意策略）；B3 显式接受"心流挤占回复配额"
并以测试锁定该契约。判定：**可执行**。

## 附: 评审团复审修订记录（v2→v3）

评审对象为 v2 计划（三席 council，二席 provider 未配置未出报告），2/2 判 **FIX FIRST**，
合成报告见 `.review/20260905-main.md`。全部阻塞项与裁决已写入 v3 正文：

- B5：发送层重构为可复用碎片 helper（flow 无入站消息，走 `channel.send` + none()）；
  flow 输出按回复同路径持久化；**修正顺序笔误**为 humanize → 逐片 safety → 发送；
- B4：由头判据机械化（结构化 `{"hook","text"}`，hook 须为上下文逐字子串，伪造即丢弃可测）；
  flow 调用走 `usage.record`；open_loops 过滤 `public_safe` + 同 guild；
- B2：只数 `role='user'`（勿照抄 `behavior.py:247` 永不匹配的 `role='bot'`）；尾部取 60 条
  Python 判窗；`save_raw_messages=True` 前提写入帮助文本、tick 检测记日志；尊重
  `proactive_global_enabled`；
- B3：三条契约显式化并测试锁定（flow 重启通用冷却、flow 过通用冷却、丢弃仍消耗槽位）；
- B6：flow 路径不调 `mood.observe`（自我强化情绪）；
- A2：familiarity_only 实现规格（observe 加参数置零三维增量）、`interaction_count`
  照常递增；A3：gate 强回不经概率写入语义、learn-skip 下沉 SQL；A4：负值行单独查询
  （`list(5)` 的 `weight DESC` 会挤出负行）+ >5 正值行测试；A5：mood 关闭回基线、
  置于 privacy 之后、倾向性措辞；A1：公私两域措辞测试。

## 附: 执行期修订记录（v3 → 交付态，2026-09-05 深度工作执行）

Phase A（commit 0e5f68b）与 Phase B（commit fcb65d5）交付，最终 347 tests 全绿。
执行中经评审（sec-reviewer / reviewer-1 / oracle 共五轮）确认的计划修订：

- **B4 open_loops 移出心流上下文**（sec 阻塞）：`open_loops` 表无来源频道列，溯源不可建立；
  `public_safe=True` 对全 guild 授权，受限频道话题会经心流泄漏到公开频道。修复 = 整体移除，
  待将来加溯源列再启用。
- **B5 心流产出不持久化到 messages 表**（sec 阻塞）：多人派生文本以 `user_id=None` 落库
  会让 `/忘记我` 清不掉参与者信息（与动态总结"不持久化多人派生内容"同一 rationale）。
  防重复改用同频道最近 5 条 `flow:` 记录注入（不可信标注）。附带动 `purge_user` 扩展：
  连带删除该用户所在服务器的 flow 行与无归属（`user_id=''`）安全事件。
- **B4 机械化收口**：`_FLOW_MAX_TOKENS`（≤300）真实接线到 utility 调用；`text` >80 字
  机械丢弃（原为纯提示词约束）；闲置判定为严格大于 20 分钟。
- **B2/B4 单快照**：资格判定与生成上下文共用一次有界快照（60 条尾部），资格只数
  `role='user'`，上下文含全部角色——bot 自己的回复因此能进入下一轮上下文。
- **B5 发送路径统一**：`_send_public_reply` 的后续碎片委托给抽取的 `_send_fragments`
  helper（首片仍 `source.reply` + first_mentions，外部行为不变）。
- **杂项**：`@flow.error` 签名按已装 discord.py 契约修正为 `(self, exc)`；
  Phase A 顺带修复 4 个既有时钟敏感测试（安静时段在测试配置内置 `00:00`/`00:00` 禁用）。
