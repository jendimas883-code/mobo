#!/usr/bin/env python3
"""mobo 边缘精密测试（edgetest）。

专挑容易出问题、容易被忽视的地方做逐项核对：
重复投递幂等、并发预算预留、安静时段跨午夜边界、超长回复分片守恒、
空白回复兜底、过期时间边界（含恰好相等的 off-by-one）、偏好权重钳制、
purge 对 NULL 用户行的保留、心流上下文裁剪、reason 分型冷却互不干扰。

运行：python scripts/edgetest.py
全部通过输出"N 项边缘检查完成 — 全部通过 ✓"并以 0 退出；任何失败以 1 退出。
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
import time
import traceback
from datetime import UTC, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.database import utcnow  # noqa: E402
from scripts.flowtest import (  # noqa: E402
    CHANNEL_MAIN,
    GUILD_ID,
    USER_A,
    USER_B,
    Env,
)
from tests.test_discord_pipeline_v4 import FakeUser  # noqa: E402

MAX_DISCORD_MESSAGE = 1980

# FakeSent 的消息 ID 是"9000 + 频道内序号"，产生消息的检查各用独立频道，
# 避免 discord_message_id 跨检查撞车触发落库幂等（真实 Discord ID 是全局雪花）
CH_DUP = "111000111000111"
CH_LONG = "222000222000222"
CH_BLANK = "333000333000333"


async def _setup_channel(env: Env, channel_id: str):
    await env.runtime_update({"safety_input_terms": "", "safety_output_terms": ""})
    await env.state.channels.set(
        GUILD_ID, channel_id, "general", listen_enabled=True, proactive_enabled=False
    )
    return env.fake_channel(int(channel_id))


CHECKS: list[tuple[str, object]] = []


def check(name: str):
    def decorator(fn):
        CHECKS.append((name, fn))
        return fn

    return decorator


@check("重复投递幂等：同一条 Discord 消息投两次，只回复一次、落库一次、入账一次")
async def check_duplicate_delivery(env: Env):
    channel = await _setup_channel(env, CH_DUP)
    env.state.llm.complete = env.stub_llm("在的。")
    msg = env.mention(channel, FakeUser(int(USER_A)), "在吗")
    await env.bot.on_message(msg)
    await env.bot.on_message(msg)
    assert len(channel.sent) == 1, f"重复投递应只回复一次，实际 {len(channel.sent)}"
    assert len(env.ledger) == 1, f"重复投递应只调用一次模型，实际 {len(env.ledger)}"
    saved = await env.sql_scalar(
        """SELECT COUNT(*) AS n FROM messages
           WHERE guild_id = ? AND channel_id = ? AND role = 'assistant'""",
        (GUILD_ID, CH_DUP),
    )
    assert saved == 1, f"助手消息应只落库一次，实际 {saved}"


@check("并发预算预留：10 个并发预留挤同一个日限 5 的频道，恰好 5 成 5 拒")
async def check_concurrent_reserve(env: Env):
    now_utc = utcnow()
    utc_start = (now_utc - timedelta(hours=12)).isoformat()

    async def reserve(index: int):
        return await env.state.proactive._reserve_channel_slot(
            GUILD_ID,
            CHANNEL_MAIN,
            f"自然参与{index}",
            now_utc=now_utc,
            utc_start=utc_start,
            cooldown_minutes=0,
            daily_limit=5,
        )

    results = await asyncio.gather(*(reserve(index) for index in range(10)))
    assert all(r is None or r == "今日额度已用完" for r in results), f"异常拒绝原因：{results}"
    successes = [r for r in results if r is None]
    denied = [r for r in results if r is not None]
    assert len(successes) == 5, f"应恰好 5 个成功，实际 {len(successes)}"
    assert len(denied) == 5 and set(denied) == {"今日额度已用完"}
    rows = await env.sql_scalar(
        "SELECT COUNT(*) AS n FROM proactive_log WHERE channel_id = ?", (CHANNEL_MAIN,)
    )
    assert rows == 5, "成功数应与日志行数一致"


@check("安静时段边界：跨午夜窗口含起点不含终点，正午在窗外")
async def check_quiet_hours_boundaries(env: Env):
    from app.behavior import ProactiveService

    def at(hour: int, minute: int = 0) -> datetime:
        return datetime(2026, 9, 6, hour, minute, tzinfo=UTC)

    overnight_start, overnight_end = "23:00", "08:00"
    assert ProactiveService._in_quiet_hours(at(23, 0), overnight_start, overnight_end) is True
    assert ProactiveService._in_quiet_hours(at(23, 30), overnight_start, overnight_end) is True
    assert ProactiveService._in_quiet_hours(at(7, 59), overnight_start, overnight_end) is True
    assert ProactiveService._in_quiet_hours(at(8, 0), overnight_start, overnight_end) is False
    assert ProactiveService._in_quiet_hours(at(12, 0), overnight_start, overnight_end) is False
    # start == end 语义为"永不安静"（测试密闭化的依赖）
    assert ProactiveService._in_quiet_hours(at(2, 0), "00:00", "00:00") is False
    # start < end 的普通窗口
    assert ProactiveService._in_quiet_hours(at(14, 0), "13:00", "15:00") is True
    assert ProactiveService._in_quiet_hours(at(15, 0), "13:00", "15:00") is False


@check("超长回复分片：5000 字拆片每片不超限，内容分毫不差")
async def check_long_reply_chunks(env: Env):
    channel = await _setup_channel(env, CH_LONG)
    long_text = "好" * 5000
    env.state.llm.complete = env.stub_llm(long_text)
    msg = env.mention(channel, FakeUser(int(USER_A)), "长文测试")
    await env.bot.on_message(msg)
    assert len(channel.sent) >= 2, "超长回复应被拆成多片"
    for sent in channel.sent:
        assert len(sent.content) <= MAX_DISCORD_MESSAGE, f"片段超长：{len(sent.content)}"
    joined = "".join(sent.content for sent in channel.sent)
    assert joined == long_text, "分片拼合后应与原文完全一致"


@check("空白回复兜底：模型返回纯空白时发送兜底文案且正常入账")
async def check_blank_reply_fallback(env: Env):
    channel = await _setup_channel(env, CH_BLANK)
    env.state.llm.complete = env.stub_llm("   ")
    msg = env.mention(channel, FakeUser(int(USER_A)), "在吗")
    ledger_before = len(env.ledger)
    await env.bot.on_message(msg)
    assert len(channel.sent) == 1, "空白回复应触发兜底"
    assert channel.sent[0].content == "模型没有返回文字。"
    assert len(env.ledger) == ledger_before + 1, "兜底前的一次模型调用应已入账"
    saved = await env.sql_scalar(
        """SELECT COUNT(*) AS n FROM messages
           WHERE guild_id = ? AND channel_id = ? AND role = 'assistant'
             AND content = '模型没有返回文字。'""",
        (GUILD_ID, CH_BLANK),
    )
    assert saved == 1, "兜底文案也应落库"


@check("过期边界：expires_at 恰好等于当下的记忆不可检索、消息必被清理")
async def check_expiry_boundary(env: Env):
    state = env.state
    now = utcnow()
    boundary = now.isoformat()
    await state.memories.add(
        GUILD_ID, USER_A, "恰好过期的记忆", kind="fact", confidence=0.95, expires_at=boundary
    )
    keep_id = await state.memories.add(
        GUILD_ID,
        USER_A,
        "还差一秒过期的记忆",
        kind="fact",
        confidence=0.95,
        expires_at=(now + timedelta(minutes=5)).isoformat(),
    )
    hits = await state.memories.retrieve(GUILD_ID, USER_A, "过期 记忆", limit=10)
    contents = [row["content"] for row in hits]
    assert "恰好过期的记忆" not in contents, "边界过期记忆不应被检索"
    assert "还差一秒过期的记忆" in contents, "未过期记忆应可检索"
    await state.database.execute(
        """INSERT INTO messages
           (guild_id, channel_id, user_id, username, role, content, created_at, expires_at)
           VALUES(?, ?, ?, '小明', 'user', '边界过期消息', ?, ?)""",
        (GUILD_ID, CHANNEL_MAIN, USER_B, boundary, boundary),
    )
    await state.database.cleanup_expired()
    left = await env.sql_scalar(
        "SELECT COUNT(*) AS n FROM messages WHERE expires_at <= ?", (utcnow().isoformat(),)
    )
    assert left == 0, "边界过期的消息应被清理（expires_at <= now）"
    row = await state.database.fetchone("SELECT id FROM memories WHERE id = ?", (keep_id,))
    assert row is not None, "未过期记忆不应被清理"


@check("偏好权重钳制：越界输入收敛到 [-1, 1]，-1.0 仍参与回避与负值优先")
async def check_weight_clamping(env: Env):
    state = env.state
    await state.preferences.upsert("极限正", ["钳正词"], 5.0, locked=True)
    await state.preferences.upsert("极限负", ["钳负词"], -5.0, locked=True)
    pos = await state.database.fetchone(
        "SELECT weight FROM bot_preferences WHERE topic = ?", ("极限正",)
    )
    neg = await state.database.fetchone(
        "SELECT weight FROM bot_preferences WHERE topic = ?", ("极限负",)
    )
    assert float(pos["weight"]) == 1.0, f"正权重应钳到 1.0，实际 {pos['weight']}"
    assert float(neg["weight"]) == -1.0, f"负权重应钳到 -1.0，实际 {neg['weight']}"
    score, topics = await state.preferences.interest_for("钳正词 钳负词")
    assert score == -1.0, f"混合命中应取最负，实际 {score}"
    assert topics == []
    avoid = await state.preferences.list(5, below_weight=0.0)
    assert "极限负" in [row["topic"] for row in avoid], "钳到 -1.0 的回避行应可见"


@check("purge 保留 NULL 用户行：心流用量（user_id NULL）不随删除权误删")
async def check_purge_keeps_null_rows(env: Env):
    channel = await _setup_channel(env, CHANNEL_MAIN)
    env.state.llm.complete = env.stub_llm("你好。")
    msg = env.mention(channel, FakeUser(int(USER_A)), "在吗")
    await env.bot.on_message(msg)
    await env.state.usage.record("flow")  # 心流用量不携带 user_id
    a_usage = await env.sql_scalar(
        "SELECT COUNT(*) AS n FROM usage_metrics WHERE user_id = ?", (USER_A,)
    )
    assert a_usage >= 1, "前置：A 应有用量行"
    await env.bot.purge_user_data(USER_A)
    a_left = await env.sql_scalar(
        "SELECT COUNT(*) AS n FROM usage_metrics WHERE user_id = ?", (USER_A,)
    )
    assert a_left == 0, "A 的用量行应被清除"
    null_rows = await env.sql_scalar(
        "SELECT COUNT(*) AS n FROM usage_metrics WHERE user_id IS NULL AND kind = 'flow'"
    )
    assert null_rows == 1, "user_id 为 NULL 的心流用量行不应被误删"


@check("心流上下文裁剪：30 条活跃消息只取最新 20 条做话题素材")
async def check_flow_context_trimming(env: Env):
    channel_id = "888000777666555"
    for index in range(30):
        created = (utcnow() - timedelta(minutes=30 + (29 - index) * 5)).isoformat()
        await env.state.database.execute(
            """INSERT INTO messages
               (guild_id, channel_id, user_id, username, role, content, created_at, expires_at)
               VALUES(?, ?, '777000666000111', '小明', 'user', ?, ?, NULL)""",
            (GUILD_ID, channel_id, f"消息{index}", created),
        )
    prepared = await env.bot._flow_prepare_channel(GUILD_ID, channel_id)
    assert prepared is not None, "30 条活跃消息应判定合格"
    _last_at, _guild, _channel, rows = prepared
    assert len(rows) == 20, f"上下文应裁剪到 20 条，实际 {len(rows)}"
    assert rows[0]["content"] == "消息10", "应保留最新的 20 条（最旧的 10 条被裁掉）"
    assert rows[-1]["content"] == "消息29", "最新一条应保留在末尾"


@check("reason 分型冷却：心流冷却不波及主动回复，两类互不误伤")
async def check_kind_cooldown_isolation(env: Env):
    now_utc = utcnow()
    minute_ago = (now_utc - timedelta(minutes=1)).isoformat()
    await env.state.database.execute(
        """INSERT INTO proactive_log(guild_id, channel_id, reason, created_at)
           VALUES(?, ?, 'flow:话题：刚才的', ?)""",
        (GUILD_ID, CHANNEL_MAIN, minute_ago),
    )
    denied = await env.state.proactive._reserve_channel_slot(
        GUILD_ID,
        CHANNEL_MAIN,
        "flow:话题：再来一次",
        now_utc=now_utc,
        utc_start=(now_utc - timedelta(hours=12)).isoformat(),
        cooldown_minutes=0,
        daily_limit=99,
        kind_cooldown_minutes=120,
    )
    assert denied == "同类冷却中", f"1 分钟前的心流行应触发同类冷却：{denied}"
    allowed = await env.state.proactive._reserve_channel_slot(
        GUILD_ID,
        CHANNEL_MAIN,
        "自然参与",
        now_utc=now_utc,
        utc_start=(now_utc - timedelta(hours=12)).isoformat(),
        cooldown_minutes=0,
        daily_limit=99,
        kind_cooldown_minutes=120,
    )
    assert allowed is None, "回复类不应被心流的同类冷却误伤"


async def main() -> int:
    started = time.monotonic()
    tmp = Path(tempfile.mkdtemp(prefix="mobo-edgetest-"))
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
    print(f"\n{total} 项边缘检查完成 — 全部通过 ✓（耗时 {elapsed:.1f}s）")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
