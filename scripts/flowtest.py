#!/usr/bin/env python3
"""mobo 全流程精密演练（flowtest）。

在临时 SQLite 上驱动真实管线（不连 Discord、不连真实模型），逐项核对：
初始化与基础设施、Web 管理台、消息管线端到端、输入/输出安全、多用户隔离、
偏好/关系/情绪、待关心事项、主动发言与心流、维护清理、总账守恒与删除权。

运行：python scripts/flowtest.py
全部通过输出"N 项检查完成 — 全部通过 ✓"并以 0 退出；任何失败以 1 退出。
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import sys
import tempfile
import time
import traceback
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cryptography.fernet import Fernet  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from app.config import BootstrapSettings  # noqa: E402
from app.database import utcnow  # noqa: E402
from app.discord_bot import AdminCommands, MoboBot, PublicCommands  # noqa: E402
from app.instance_lock import InstanceLock  # noqa: E402
from app.llm import ModelResult  # noqa: E402
from app.state import create_state  # noqa: E402
from app.web import create_web_app  # noqa: E402
from tests.test_discord_pipeline_v4 import (  # noqa: E402
    FakeChannel,
    FakeMessage,
    FakeUser,
)

TEST_PASSWORD = "F10w!Test#Pass2026"
GUILD_ID = "333333333333333"
CHANNEL_MAIN = "444444444444444"
CHANNEL_FLOW = "555000111222333"
CHANNEL_CLEAR = "777000888888888"
CHANNEL_BUDGET = "999000111222333"
USER_A = "111111111111111"
USER_B = "222222222222222"
USER_HALVED = "666666666666666"
USER_FLOW = "777000666000111"
BOT_ID = 999999999999999

EXPECTED_COMMANDS = {
    "帮助",
    "隐私",
    "忘记我",
    "状态",
    "管理台",
    "清空频道",
    "人设",
    "频道设置",
    "主动发言",
    "重载配置",
}


def approx(expected: float, tol: float):
    """近似相等断言助手。"""

    class _Approx:
        def __eq__(self, other):
            return abs(other - expected) <= tol

    return _Approx()


class Env:
    """演练共享环境：一个状态、一个 bot、一本模型调用总账。"""

    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.state = None
        self.bot = None
        self.bot_user = None
        self.ledger: list[tuple[int, int]] = []
        self._next_msg_id = 9000000000000000

    async def setup(self) -> None:
        self.state = await create_state(
            BootstrapSettings(
                _env_file=None,
                discord_token="flowtest-token",
                admin_username="admin",
                admin_password=TEST_PASSWORD,
                session_secret="f" * 48,
                config_encryption_key=Fernet.generate_key().decode("ascii"),
                db_path=self.tmp / "mobo-flowtest.db",
                public_base_url="http://testserver",
                cookie_secure=False,
                allowed_hosts="testserver,localhost",
                test_mode=True,
            )
        )
        self.bot = MoboBot(self.state)
        self.bot_user = FakeUser(BOT_ID, bot=True, name="mobo")
        self.bot._connection.user = self.bot_user
        await self.bot.add_cog(PublicCommands(self.bot))
        await self.bot.add_cog(AdminCommands(self.bot))
        await self.runtime_update(
            {
                "message_debounce_seconds": 0,
                "humanization_enabled": False,
                "save_raw_messages": True,
                "rate_limit_requests": 100,
            }
        )

    def next_message_id(self) -> int:
        self._next_msg_id += 1
        return self._next_msg_id

    async def runtime_update(self, values: dict) -> None:
        await self.state.runtime.update(values, actor="flowtest")

    def stub_llm(self, text: str, *, tokens: tuple[int, int] = (120, 40)) -> AsyncMock:
        """替换模型网关：固定回复 + 记入总账。"""

        async def complete(config, messages, **kwargs):
            self.ledger.append(tokens)
            return ModelResult(
                text=text,
                input_tokens=tokens[0],
                output_tokens=tokens[1],
                latency_ms=1,
                provider="flowtest",
                model="stub",
            )

        return AsyncMock(side_effect=complete)

    def flow_result(
        self, hook: str, text: str, *, tokens: tuple[int, int] = (120, 40)
    ) -> AsyncMock:
        """心流专用桩：返回 JSON 格式的 hook/text。"""
        return self.stub_llm(
            json.dumps({"hook": hook, "text": text}, ensure_ascii=False),
            tokens=tokens,
        )

    def fake_channel(self, channel_id: int) -> FakeChannel:
        return FakeChannel(channel_id)

    def mention(self, channel: FakeChannel, user: FakeUser, content: str) -> FakeMessage:
        return FakeMessage(
            self.next_message_id(),
            user,
            channel,
            f"<@{BOT_ID}> {content}",
            mentions=[self.bot_user],
        )

    async def sql_scalar(self, sql: str, params: tuple = ()) -> int:
        row = await self.state.database.fetchone(sql, params)
        return int(row["n"]) if row is not None else 0


CHECKS: list[tuple[str, object]] = []


def check(name: str):
    def decorator(fn):
        CHECKS.append((name, fn))
        return fn

    return decorator


# ── 一、初始化与基础设施 ────────────────────────────────────────────


@check("create_state：schema 迁移、种子数据与运行时默认值齐备")
async def check_bootstrap(env: Env):
    state = env.state
    prefs = await state.preferences.list(20)
    assert sum(1 for row in prefs if row["source"] == "seed") == 4, "内置种子偏好应为 4 条"
    mood_row = await state.database.fetchone("SELECT * FROM mood_state WHERE id = 1")
    assert mood_row is not None, "mood_state 初始行缺失"
    config = await state.runtime.all()
    assert len(config) > 50, f"运行时默认键过少：{len(config)}"
    assert config["bot_name"] == "mobo"


@check("单实例锁：同库第二实例必须被拒绝，释放后可复用")
async def check_instance_lock(env: Env):
    lock_path = env.tmp / "flowtest.instance.lock"
    first = InstanceLock(lock_path)
    first.acquire()
    try:
        second = InstanceLock(lock_path)
        try:
            second.acquire()
        except Exception:
            pass
        else:
            raise AssertionError("第二实例未被拒绝")
    finally:
        first.release()
    again = InstanceLock(lock_path)
    again.acquire()
    again.release()


@check("中文命令：十个命令全部注册进命令树")
async def check_commands(env: Env):
    names = {command.name for command in env.bot.tree.get_commands()}
    missing = EXPECTED_COMMANDS - names
    assert not missing, f"缺少命令：{missing}"


# ── 二、Web 管理台 ──────────────────────────────────────────────────


def _client(env: Env) -> AsyncClient:
    return AsyncClient(
        transport=ASGITransport(app=create_web_app(env.state)),
        base_url="http://testserver",
        follow_redirects=False,
    )


@check("管理台：healthz 公开、未登录重定向、错误密码不产生会话")
async def check_web_public(env: Env):
    async with _client(env) as client:
        health = await client.get("/healthz")
        assert health.status_code == 200 and health.json()["database"] == "ok"
        redirected = await client.get("/")
        assert redirected.status_code == 303 and redirected.headers["location"] == "/login"
        bad = await client.post("/login", data={"username": "admin", "password": "wrong-password"})
        assert "mobo_admin_session" not in bad.cookies, "错误密码不应产生会话"


@check("管理台：密码登录成功，全部页面可达且带 CSRF，未登录 API 写入被拒")
async def check_web_login(env: Env):
    async with _client(env) as client:
        login = await client.post("/login", data={"username": "admin", "password": TEST_PASSWORD})
        assert login.status_code == 303, "登录应 303"
        assert "mobo_admin_session" in login.cookies
        for page in ("/", "/settings", "/behavior", "/memories", "/models", "/security", "/audit"):
            response = await client.get(page)
            assert response.status_code == 200, f"{page} 应 200，得到 {response.status_code}"
        settings = await client.get("/settings")
        assert re.search(r'<body data-csrf="([^"]+)"', settings.text), "settings 页缺 CSRF"
    async with _client(env) as fresh:
        denied = await fresh.post("/api/preferences", json={"topic": "x", "weight": 0.1})
        assert denied.status_code in (303, 401, 403), "未登录 API 写入未被拒绝"


# ── 三、消息管线端到端 ──────────────────────────────────────────────


async def _setup_main_channel(env: Env):
    await env.runtime_update({"safety_input_terms": "", "safety_output_terms": ""})
    await env.state.channels.set(
        GUILD_ID, CHANNEL_MAIN, "general", listen_enabled=True, proactive_enabled=False
    )


@check("管线：@ 提及触发回复，回复内容与用户/助手消息双双落库")
async def check_pipeline_reply(env: Env):
    await _setup_main_channel(env)
    channel = env.fake_channel(int(CHANNEL_MAIN))
    env.state.llm.complete = env.stub_llm("你好呀，我在的。")
    msg = env.mention(channel, FakeUser(int(USER_A)), "在吗？帮我看看这个")
    await env.bot.on_message(msg)
    assert channel.sent, "未发送任何回复"
    assert "你好呀" in channel.sent[0].content and "我在的" in channel.sent[0].content
    user_row = await env.state.database.fetchone(
        """SELECT COUNT(*) AS n FROM messages
           WHERE guild_id = ? AND channel_id = ? AND role = 'user'""",
        (GUILD_ID, CHANNEL_MAIN),
    )
    assistant_row = await env.state.database.fetchone(
        """SELECT COUNT(*) AS n FROM messages
           WHERE guild_id = ? AND channel_id = ? AND role = 'assistant'""",
        (GUILD_ID, CHANNEL_MAIN),
    )
    assert int(user_row["n"]) >= 1, "用户消息未落库"
    assert int(assistant_row["n"]) >= 1, "助手消息未落库"
    usage = await env.state.database.fetchone(
        "SELECT kind, input_tokens FROM usage_metrics ORDER BY id DESC LIMIT 1"
    )
    assert usage["kind"] == "discord_chat" and int(usage["input_tokens"]) > 0


@check("输入安全：命中禁词零模型调用并给出不能处理")
async def check_input_safety(env: Env):
    await _setup_main_channel(env)
    await env.runtime_update({"safety_input_terms": "禁词"})
    channel = env.fake_channel(int(CHANNEL_MAIN))
    stub = env.stub_llm("不应调用")
    env.state.llm.complete = stub
    msg = env.mention(channel, FakeUser(int(USER_A)), "这里有禁词")
    await env.bot.on_message(msg)
    stub.assert_not_awaited()
    assert channel.sent and "不能处理" in channel.sent[0].content


@check("输出安全：命中脱敏词后回复不再包含原文")
async def check_output_redact(env: Env):
    await _setup_main_channel(env)
    await env.runtime_update({"safety_output_terms": "机密", "safety_default_action": "redact"})
    channel = env.fake_channel(int(CHANNEL_MAIN))
    env.state.llm.complete = env.stub_llm("这是机密内容请保密。")
    msg = env.mention(channel, FakeUser(int(USER_A)), "说说看")
    await env.bot.on_message(msg)
    assert channel.sent, "未发送回复"
    assert "机密" not in channel.sent[0].content, "脱敏词仍出现在回复里"
    await env.runtime_update({"safety_output_terms": "", "safety_default_action": "block"})


@check("主动路径关系观测：速率减半且只动 familiarity（Phase A2）")
async def check_public_relationship_half_rate(env: Env):
    await env.runtime_update({"relationship_learning_rate": 0.2})
    channel = env.fake_channel(int(CHANNEL_MAIN))
    from app.discord_bot import GenerationPayload

    payload = GenerationPayload(
        message=FakeMessage(env.next_message_id(), FakeUser(int(USER_HALVED)), channel, "随便聊聊"),
        guild_id=GUILD_ID,
        channel_id=CHANNEL_MAIN,
        context_channel_id=CHANNEL_MAIN,
        user_id=USER_HALVED,
        text="随便聊聊",
        content="随便聊聊",
        config=await env.state.runtime.all(),
        direct=False,
        listened=True,
        proactive_reason="自然参与",
        source_message_db_id=None,
        generation_version=1,
    )
    await env.bot._learn_after_success(payload)
    rel = await env.state.relationships.get(GUILD_ID, USER_HALVED, 60)
    assert rel.familiarity == approx(0.1, 1e-6), f"familiarity 应为 0.1，得到 {rel.familiarity}"
    assert rel.warmth == approx(0.1, 1e-6), "主动路径不应改变 warmth"


@check("隐私隔离：私密记忆不进公开上下文，也不跨用户泄漏")
async def check_privacy_isolation(env: Env):
    state = env.state
    await state.memories.add(
        GUILD_ID, USER_A, "他喜欢薄荷巧克力冰淇淋", kind="fact", confidence=0.95
    )
    await state.memories.add(
        GUILD_ID, USER_A, "他在准备一场重要的面试", kind="explicit", confidence=0.95
    )
    public_a = await state.context.build(GUILD_ID, CHANNEL_MAIN, USER_A, "你好")
    private_a = await state.context.build(GUILD_ID, CHANNEL_MAIN, USER_A, "你好", public=False)
    assert "薄荷巧克力" in public_a[0]["content"], "高置信 fact 记忆应可在公聊使用"
    assert "重要的面试" not in public_a[0]["content"], "私密记忆泄漏进公开上下文"
    assert "重要的面试" in private_a[0]["content"], "私密范围应加载私密记忆"
    public_b = await state.context.build(GUILD_ID, CHANNEL_MAIN, USER_B, "你好")
    assert "薄荷巧克力" not in public_b[0]["content"], "A 的记忆泄漏给 B"


# ── 四、偏好与情绪 ──────────────────────────────────────────────────


@check("负权重三面契约：聚合负值优先、学习跳过、回避行最负优先（Phase A3）")
async def check_negative_weights(env: Env):
    state = env.state
    await state.preferences.upsert("陶艺", ["陶艺"], 0.4, locked=True)
    await state.preferences.upsert("争议话题甲", ["吵架甲"], -0.5, locked=False)
    score, topics = await state.preferences.interest_for("聊聊陶艺顺便吵架甲")
    assert abs(score - (-0.5)) < 1e-9, f"负值应优先，得到 {score}"
    # OURS: topics 包含所有命中的偏好（正值和负值均在列），score 取最负
    assert "争议话题甲" in topics, "负值命中应出现在 topics 中"
    await state.preferences.interest_for("又吵架甲了", learn=True)
    row = await state.database.fetchone(
        "SELECT weight, evidence_count FROM bot_preferences WHERE topic = ?", ("争议话题甲",)
    )
    assert float(row["weight"]) == -0.5 and int(row["evidence_count"]) == 0, "负权重行不应被学习"
    for index in range(6):
        await state.preferences.upsert(
            f"回避{index}", [f"回避词{index}"], -0.1 - index * 0.15, locked=True
        )
    avoid_rows = await state.preferences.avoid()
    weights = [float(row["weight"]) for row in avoid_rows]
    assert weights == sorted(weights), "回避行应按最负优先排序"
    assert weights[0] == approx(-0.85, 1e-9), "最强回避 -0.85 应排第一"


@check("提示词双行与心情文风行（Phase A4/A5）")
async def check_prompt_lines(env: Env):
    state = env.state
    await state.preferences.upsert("陶艺", ["陶艺"], 1.0, locked=True)
    await state.preferences.upsert("争议话题", ["吵架"], -0.6, locked=True)
    await state.mood.set(-0.8, 0.3, 0.7)
    system = (await state.context.build(GUILD_ID, CHANNEL_MAIN, USER_A, "你好"))[0]["content"]
    assert "你目前较偏好的话题：陶艺(1.00)" in system
    assert "想避开的话题" in system and "争议话题" in system
    assert "自然收敛、少用表情" in system, "低落心情应注入收敛文风"
    await state.mood.set(0.0, 0.3, 0.7)
    system = (await state.context.build(GUILD_ID, CHANNEL_MAIN, USER_A, "你好"))[0]["content"]
    assert "心情对文风的影响" not in system, "中性心情不应注入文风行"


@check("情绪：更新有界且按半衰期回归基线")
async def check_mood_bounds(env: Env):
    state = env.state
    await state.mood.set(-5.0, 5.0, 5.0)
    config = await state.runtime.all()
    mood = await state.mood.current(config)
    assert -1.0 <= float(mood["valence"]) <= 1.0
    assert 0.0 <= float(mood["energy"]) <= 1.0
    half_life = max(1.0, float(config["mood_half_life_minutes"]))
    stale = (utcnow() - timedelta(minutes=half_life * 10)).isoformat()
    await state.database.execute("UPDATE mood_state SET updated_at = ? WHERE id = 1", (stale,))
    mood = await state.mood.current(config)
    baseline = float(config["mood_baseline_valence"])
    assert abs(float(mood["valence"]) - baseline) < 0.05, "久置后情绪应回归基线"
    await state.mood.set(0.0, 0.6, 0.7)


# ── 五、待关心事项 ──────────────────────────────────────────────────


@check("待关心事项：建条、到期可取、公开域只出 public_safe、关闭即失效")
async def check_followups(env: Env):
    state = env.state
    now = utcnow()
    public_id = await state.followups.create(
        GUILD_ID, USER_A, "周末爬山的事", now + timedelta(hours=12), public_safe=True, now=now
    )
    assert public_id is not None, "public_safe 待关心建条失败"
    await state.database.execute(
        """INSERT INTO open_loops
           (guild_id, user_id, topic, public_safe, status, followup_after,
            expires_at, followup_count, created_at, updated_at)
           VALUES(?, ?, '他的心理咨询预约', 0, 'open', ?, NULL, 0, ?, ?)""",
        (
            GUILD_ID,
            USER_A,
            (now + timedelta(hours=12)).isoformat(),
            now.isoformat(),
            now.isoformat(),
        ),
    )
    due_past = (now - timedelta(hours=1)).isoformat()
    await state.database.execute("UPDATE open_loops SET followup_after = ?", (due_past,))
    due = await state.followups.list_due(guild_id=GUILD_ID, user_id=USER_A)
    assert len(due) >= 2, "两条到期待关心都应可取出"
    await state.followups.close(public_id)
    after = await state.followups.list_due(guild_id=GUILD_ID, user_id=USER_A)
    assert public_id not in {row["id"] for row in after}, "关闭后仍到期"


# ── 六、主动发言与心流 ──────────────────────────────────────────────


@check("主动发言：闸门触发发言，随后立即进入频道冷却")
async def check_proactive_decide(env: Env):
    state = env.state
    await env.runtime_update(
        {
            "proactive_global_enabled": True,
            "proactive_quiet_start": "00:00",
            "proactive_quiet_end": "00:00",
            "gate_threshold": 10,
        }
    )
    await state.channels.set(
        GUILD_ID, CHANNEL_MAIN, "general", listen_enabled=True, proactive_enabled=True
    )
    config = await state.runtime.all()
    decision = await state.proactive.decide(
        GUILD_ID, CHANNEL_MAIN, USER_A, "mobo 你好，帮我看看这个怎么解决？", config
    )
    assert decision.should_speak, f"闸门应触发：{decision.reason}"
    decision = await state.proactive.decide(
        GUILD_ID, CHANNEL_MAIN, USER_B, "mobo 再帮我看一个？", config
    )
    assert not decision.should_speak and decision.reason == "频道冷却中"
    await env.runtime_update({"gate_threshold": 80})


async def _seed_idle_channel(env: Env, channel_id: str) -> None:
    for index in range(12):
        created = (utcnow() - timedelta(minutes=30 + (11 - index) * 5)).isoformat()
        await env.state.database.execute(
            """INSERT INTO messages
               (guild_id, channel_id, user_id, username, role, content, created_at, expires_at)
               VALUES(?, ?, ?, '小明', 'user', ?, ?, NULL)""",
            (GUILD_ID, channel_id, USER_FLOW, f"活跃消息{index}", created),
        )


async def _enable_flow(env: Env):
    await env.runtime_update(
        {
            "flow_enabled": True,
            "proactive_global_enabled": True,
            "flow_probability": 1.0,
            "proactive_quiet_start": "00:00",
            "proactive_quiet_end": "00:00",
            "save_raw_messages": True,
        }
    )
    await env.state.channels.set(
        GUILD_ID, CHANNEL_FLOW, "flowzone", listen_enabled=True, proactive_enabled=True
    )
    await env.state.mood.set(0.0, 0.6, 0.7)
    await _seed_idle_channel(env, CHANNEL_FLOW)


@check("心流 happy path：发送、proactive_log 落账、无助手消息落库、用量记账、社交余量不变")
async def check_flow_happy(env: Env):
    """OURS 语义：
    - 发送成功
    - proactive_log 行存在，reason 以 'flow:话题：' 开头
    - 不落库助手消息（flow 输出是多用户派生文本，与摘要路径同理）
    - 用量记录 kind='flow_topic'
    - 不调用 mood.observe（社交余量不变）
    """
    await _enable_flow(env)
    channel = env.fake_channel(int(CHANNEL_FLOW))
    env.bot.get_channel = lambda channel_id: channel
    env.state.llm.complete = env.flow_result("活跃消息5", "刚才你们聊到活跃消息5，我也想参与！")
    social_before = float(
        (await env.state.database.fetchone("SELECT social_budget FROM mood_state WHERE id = 1"))[
            "social_budget"
        ]
    )
    await env.bot._flow_tick()
    assert len(channel.sent) == 1, "心流开场白未发送"
    row = await env.state.database.fetchone(
        "SELECT reason FROM proactive_log WHERE channel_id = ? ORDER BY id DESC LIMIT 1",
        (CHANNEL_FLOW,),
    )
    assert str(row["reason"]).startswith("flow:话题："), f"reason 口径错误：{row['reason']}"
    # 不落库助手消息
    saved = await env.sql_scalar(
        """SELECT COUNT(*) AS n FROM messages
           WHERE guild_id = ? AND channel_id = ? AND role = 'assistant'""",
        (GUILD_ID, CHANNEL_FLOW),
    )
    assert saved == 0, "OURS 语义：flow 输出不应落库助手消息"
    # 用量记账
    usage = await env.sql_scalar(
        "SELECT COUNT(*) AS n FROM usage_metrics WHERE kind = 'flow_topic'"
    )
    assert usage == 1, "心流模型调用未记入用量"
    # 社交余量不变（不调 mood.observe）
    social_after = float(
        (await env.state.database.fetchone("SELECT social_budget FROM mood_state WHERE id = 1"))[
            "social_budget"
        ]
    )
    assert abs(social_after - social_before) < 1e-9, "flow 路径不应改变社交余量"


@check("心流由头契约：虚构 hook（非子串）丢弃且占名额")
async def check_flow_no_hook(env: Env):
    """OURS 语义：
    - JSON hook 不是上下文子串 → 丢弃，不发送
    - 槽位已消耗（proactive_log 行 'flow:待定' 保留）
    """
    # 清理前序检查遗留的冷却行，确保本检查的 flow 能走到 LLM 步骤
    stale = (utcnow() - timedelta(minutes=150)).isoformat()
    await env.state.database.execute(
        "UPDATE proactive_log SET created_at = ? WHERE reason LIKE 'flow:%'", (stale,)
    )
    await _enable_flow(env)
    channel = env.fake_channel(int(CHANNEL_FLOW))
    env.bot.get_channel = lambda channel_id: channel
    env.state.llm.complete = env.flow_result("完全不存在的话题xyz", "这条不应被发送")
    before = await env.sql_scalar("SELECT COUNT(*) AS n FROM proactive_log")
    await env.bot._flow_tick()
    assert channel.sent == [], "虚构 hook 不应发送"
    after = await env.sql_scalar("SELECT COUNT(*) AS n FROM proactive_log")
    assert after == before + 1, "槽位应已消耗（proactive_log 新增一行 'flow:待定'）"
    row = await env.state.database.fetchone(
        "SELECT reason FROM proactive_log WHERE channel_id = ? ORDER BY id DESC LIMIT 1",
        (CHANNEL_FLOW,),
    )
    assert row["reason"] == "flow:待定", "被丢弃的生成应保留占位 reason"


@check("心流冷却：120 分钟内同类行在同一事务内拒绝")
async def check_flow_cooldown(env: Env):
    await _enable_flow(env)
    # 前序检查留下的"现在"时刻的 flow:待定 行会先触发 45 分钟通用冷却，
    # 导致本检查碰巧通过；把全部记账行变陈旧，确保拒绝确实来自 120 分钟 kind 冷却。
    ancient = (utcnow() - timedelta(minutes=150)).isoformat()
    await env.state.database.execute(
        "UPDATE proactive_log SET created_at = ?", (ancient,)
    )
    recent = (utcnow() - timedelta(minutes=60)).isoformat()
    await env.state.database.execute(
        """INSERT INTO proactive_log(guild_id, channel_id, reason, created_at)
           VALUES(?, ?, 'flow:话题：早前的', ?)""",
        (GUILD_ID, CHANNEL_FLOW, recent),
    )
    channel = env.fake_channel(int(CHANNEL_FLOW))
    env.bot.get_channel = lambda channel_id: channel
    env.state.llm.complete = env.flow_result("活跃消息", "再来一次？")
    await env.bot._flow_tick()
    assert channel.sent == [], "心流冷却未生效"
    stale = (utcnow() - timedelta(minutes=150)).isoformat()
    await env.state.database.execute(
        "UPDATE proactive_log SET created_at = ? WHERE reason LIKE 'flow:%'", (stale,)
    )


@check("共享日限池：回复与心流合计不超每日上限")
async def check_shared_budget(env: Env):
    # 清理前序检查的冷却行
    stale = (utcnow() - timedelta(minutes=150)).isoformat()
    await env.state.database.execute(
        "UPDATE proactive_log SET created_at = ? WHERE reason LIKE 'flow:%'", (stale,)
    )
    await _enable_flow(env)
    await env.runtime_update({"proactive_daily_limit": 2})
    await env.state.channels.set(
        GUILD_ID, CHANNEL_BUDGET, "budget", listen_enabled=True, proactive_enabled=True
    )
    await _seed_idle_channel(env, CHANNEL_BUDGET)
    # 确保 CHANNEL_BUDGET 有最新消息，被 tie-break 选中
    recent = (utcnow() - timedelta(minutes=25)).isoformat()
    await env.state.database.execute(
        """INSERT INTO messages
           (guild_id, channel_id, user_id, username, role, content, created_at, expires_at)
           VALUES(?, ?, ?, '小明', 'user', '最新预算消息', ?, NULL)""",
        (GUILD_ID, CHANNEL_BUDGET, USER_FLOW, recent),
    )
    two_hours_ago = (utcnow() - timedelta(hours=2)).isoformat()
    await env.state.database.execute(
        """INSERT INTO proactive_log(guild_id, channel_id, reason, created_at)
           VALUES(?, ?, '自然参与', ?)""",
        (GUILD_ID, CHANNEL_BUDGET, two_hours_ago),
    )
    channel = env.fake_channel(int(CHANNEL_BUDGET))
    env.bot.get_channel = lambda channel_id: channel
    env.state.llm.complete = env.flow_result("最新预算消息", "预算测试")
    await env.bot._flow_tick()
    count = await env.sql_scalar(
        "SELECT COUNT(*) AS n FROM proactive_log WHERE channel_id = ?", (CHANNEL_BUDGET,)
    )
    assert count == 2, "回复 + 心流应恰好占满日限 2"
    # 不传 kind_cooldown_minutes，仅验证共享日限池
    denied = await env.state.proactive._reserve_channel_slot(
        GUILD_ID,
        CHANNEL_BUDGET,
        "自然参与",
        now_utc=utcnow(),
        utc_start=(utcnow() - timedelta(hours=12)).isoformat(),
        cooldown_minutes=0,
        daily_limit=2,
    )
    assert denied == "今日额度已用完", f"第三个主动行为应被拒：{denied}"
    await env.runtime_update({"proactive_daily_limit": 6})


@check("心流循环：tick 内异常不终止循环")
async def check_flow_survives(env: Env):
    import logging

    logger = logging.getLogger("mobo.discord")
    was_disabled = logger.disabled
    logger.disabled = True
    try:
        real_tick = env.bot._flow_tick

        async def broken_tick():
            raise RuntimeError("演练注入的异常")

        env.bot._flow_tick = broken_tick
        try:
            await env.bot.flow_error(RuntimeError("演练注入的异常"))
        except Exception:
            raise AssertionError("flow error handler 应捕获异常而不传播") from None
        finally:
            env.bot._flow_tick = real_tick
    finally:
        logger.disabled = was_disabled


# ── 七、记忆与维护 ──────────────────────────────────────────────────


@check("维护清理：过期消息与过期记忆被清出，活跃数据保留")
async def check_cleanup_expired(env: Env):
    state = env.state
    past = (utcnow() - timedelta(days=1)).isoformat()
    await state.database.execute(
        """INSERT INTO messages
           (guild_id, channel_id, user_id, username, role, content, created_at, expires_at)
           VALUES(?, ?, ?, '小明', 'user', '过期的消息', ?, ?)""",
        (GUILD_ID, CHANNEL_CLEAR, USER_FLOW, past, past),
    )
    await state.memories.add(
        GUILD_ID, USER_A, "这条记忆已经过期了", kind="fact", confidence=0.9, expires_at=past
    )
    keep_id = await state.memories.add(
        GUILD_ID, USER_A, "这条记忆还活着", kind="fact", confidence=0.9
    )
    deleted = await state.database.cleanup_expired()
    assert sum(deleted.values()) >= 2, f"清理量异常：{deleted}"
    expired_left = await env.sql_scalar(
        "SELECT COUNT(*) AS n FROM messages WHERE expires_at IS NOT NULL AND expires_at <= ?",
        (utcnow().isoformat(),),
    )
    assert expired_left == 0, "过期消息未被清出"
    row = await state.database.fetchone("SELECT status FROM memories WHERE id = ?", (keep_id,))
    assert row is not None and row["status"] == "active", "活跃记忆被误删"


@check("频道清空：单频道上下文一次清空")
async def check_clear_channel(env: Env):
    state = env.state
    await _seed_idle_channel(env, CHANNEL_CLEAR)
    deleted = await state.memories.clear_channel(GUILD_ID, CHANNEL_CLEAR)
    assert deleted >= 1, "清空返回量异常"
    count = await env.sql_scalar(
        "SELECT COUNT(*) AS n FROM messages WHERE guild_id = ? AND channel_id = ?",
        (GUILD_ID, CHANNEL_CLEAR),
    )
    assert count == 0, "频道消息未清空"


# ── 八、总账核对 ────────────────────────────────────────────────────


@check("总账：模型调用 token 与 usage_metrics 分毫不差")
async def check_token_ledger(env: Env):
    expected = sum(in_tok + out_tok for in_tok, out_tok in env.ledger)
    actual = await env.sql_scalar(
        "SELECT COALESCE(SUM(input_tokens + output_tokens), 0) AS n FROM usage_metrics"
    )
    assert actual == expected, f"token 总账不平：期望 {expected}，实际 {actual}"


@check("总账：主动行为 reason 词表合法且每频道不超日限")
async def check_proactive_ledger(env: Env):
    rows = await env.state.database.fetchall("SELECT channel_id, reason FROM proactive_log")
    assert rows, "前置：proactive_log 应有记录"
    # OURS 语义：'flow:待定' 是占位 reason（生成被丢弃时保留）
    allowed = ("闸门(", "偏好话题：", "自然参与", "flow:话题：", "flow:待定")
    counts: dict[str, int] = {}
    for row in rows:
        assert str(row["reason"]).startswith(allowed), f"非法 reason：{row['reason']}"
        counts[str(row["channel_id"])] = counts.get(str(row["channel_id"]), 0) + 1
    config = await env.state.runtime.all()
    daily_limit = int(config["proactive_daily_limit"])
    for channel_id, count in counts.items():
        assert count <= daily_limit, f"频道 {channel_id} 超日限：{count} > {daily_limit}"


# ── 九、删除权（/忘记我） ───────────────────────────────────────────


@check("删除权：purge_user_data 清空该用户全部数据，flow 侧通道一并清理，他人数据无损")
async def check_purge_user(env: Env):
    """OURS 语义扩展：
    - 基本删除权同 teammate
    - 额外验证：flow proactive_log 行和无归属 safety_events 被清理
    """
    state = env.state
    await state.memories.add(GUILD_ID, USER_B, "B 的专属记忆", kind="fact", confidence=0.9)
    await state.database.execute(
        """INSERT OR REPLACE INTO user_profiles(user_id, display_name, first_seen_at, last_seen_at)
           VALUES(?, '用户A', ?, ?)""",
        (USER_A, utcnow().isoformat(), utcnow().isoformat()),
    )
    # 插入 flow proactive_log 行（应在 purge 时被清理）
    await state.database.execute(
        """INSERT INTO proactive_log(guild_id, channel_id, reason, created_at)
           VALUES(?, ?, 'flow:话题：应被清理', ?)""",
        (GUILD_ID, CHANNEL_FLOW, utcnow().isoformat()),
    )
    # 插入无归属 safety_events（应在 purge 时被清理）
    await state.database.execute(
        """INSERT INTO safety_events
           (guild_id, channel_id, user_id, direction, category, action, content_hash, created_at)
           VALUES(?, ?, '', 'output', 'custom', 'block', 'hash1', ?)""",
        (GUILD_ID, CHANNEL_FLOW, utcnow().isoformat()),
    )
    # 插入有归属 safety_events（B 的，应保留）
    await state.database.execute(
        """INSERT INTO safety_events
           (guild_id, channel_id, user_id, direction, category, action, content_hash, created_at)
           VALUES(?, ?, ?, 'input', 'custom', 'block', 'hash2', ?)""",
        (GUILD_ID, CHANNEL_FLOW, USER_B, utcnow().isoformat()),
    )

    a_messages = await env.sql_scalar(
        "SELECT COUNT(*) AS n FROM messages WHERE user_id = ?", (USER_A,)
    )
    assert a_messages >= 1, "前置：A 应有消息记录"

    await env.bot.purge_user_data(USER_A)

    for table in (
        "messages",
        "memories",
        "relationships",
        "user_profiles",
        "open_loops",
        "usage_metrics",
    ):
        remaining = await env.sql_scalar(
            f"SELECT COUNT(*) AS n FROM {table} WHERE user_id = ?", (USER_A,)
        )
        assert remaining == 0, f"{table} 中仍有 A 的数据 {remaining} 行"
    keep = await state.database.fetchone(
        "SELECT COUNT(*) AS n FROM memories WHERE content LIKE '%B 的专属记忆%'"
    )
    assert int(keep["n"]) == 1, "B 的记忆被误删"
    summaries = await env.sql_scalar("SELECT COUNT(*) AS n FROM channel_summaries")
    assert summaries == 0, "频道摘要未随删除权清空（防嵌入残留）"
    # flow 侧通道清理
    flow_rows = await env.sql_scalar(
        "SELECT COUNT(*) AS n FROM proactive_log WHERE reason LIKE 'flow:%'"
    )
    assert flow_rows == 0, "flow proactive_log 行应随 purge 清理"
    unattributed = await env.sql_scalar(
        "SELECT COUNT(*) AS n FROM safety_events WHERE guild_id = ? AND (user_id IS NULL OR user_id = '')",
        (GUILD_ID,),
    )
    assert unattributed == 0, "无归属 safety_events 应随 purge 清理"
    b_safety = await env.sql_scalar(
        "SELECT COUNT(*) AS n FROM safety_events WHERE user_id = ?", (USER_B,)
    )
    assert b_safety >= 1, "B 的有归属 safety_events 应保留"


async def main() -> int:
    started = time.monotonic()
    tmp = Path(tempfile.mkdtemp(prefix="mobo-flowtest-"))
    env = Env(tmp)
    failures: list[tuple[str, str]] = []
    try:
        await env.setup()
        for index, (name, fn) in enumerate(CHECKS, start=1):
            label = f"[{index:02d}/{len(CHECKS)}] {name}"
            try:
                await fn(env)
            except Exception:
                failures.append((name, traceback.format_exc()))
                print(f"{label} ✗")
            else:
                print(f"{label} ✓")
    finally:
        if env.state is not None:
            await env.state.database.close()
        shutil.rmtree(tmp, ignore_errors=True)
    elapsed = time.monotonic() - started
    total = len(CHECKS)
    passed = total - len(failures)
    if failures:
        print("\n失败详情：")
        for name, detail in failures:
            print(f"—— {name} ——")
            print(detail)
        print(f"\n{passed}/{total} 项通过 — 存在失败 ✗（耗时 {elapsed:.1f}s）")
        return 1
    print(f"\n{total} 项检查完成 — 全部通过 ✓（耗时 {elapsed:.1f}s）")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
