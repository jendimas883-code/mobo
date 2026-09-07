"""回复必要性评分 — 纯函数，零副作用。

移植自 MaiBot (Mai-with-u/MaiBot) reply_necessity.py，GPL-3.0。
已删除 QQ 特有噪声（CQ 码、合并转发、发言榜、其他 bot 检测）。
"""

from __future__ import annotations

from collections.abc import Sequence
from math import log1p

# ── 常量（与 MaiBot 口径对齐，仅暴露阈值为配置项）───────────────────
SCORE_NAME_MENTION = 80
SCORE_CONTENT_QUESTION = 15
SCORE_CONTENT_REQUEST = 20
SCORE_CONTENT_OPINION = 20
SCORE_CONTENT_LONG_TEXT = 5
SCORE_CONTENT_LONGER_TEXT = 10
SCORE_SHORT_REACTION_PENALTY = -25

PRESSURE_FULL_RATIO = 5.0
PRESSURE_MAX_SCORE = 40

PRESENCE_FREE_RATIO = 0.25
PRESENCE_FULL_RATIO = 0.60
# MaiBot 原值 25；mobo 闸门不含 @提及/DM 分量（由 direct 路径短路），
# 分值整体分布更低，取 40 补偿以维持同等抑制强度。
PRESENCE_PENALTY_MAX = 40

SHORT_REACTIONS = {"哈哈", "哈哈哈", "草", "笑死", "好", "嗯", "啊", "哦", "6", "666", "？", "?"}
QUESTION_TERMS = ("怎么", "如何", "为什么", "有没有", "是什么", "啥意思")
REQUEST_TERMS = ("帮我", "帮忙", "能不能", "可以吗", "要不要", "需要", "求", "看看", "试试")
OPINION_TERMS = ("你觉得", "你认为", "咋看", "怎么看", "有什么建议")


def _is_chinese_char(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"


def _has_name_mention(text: str, bot_name: str) -> bool:
    """检查 bot 名字是否出现在文本中（容忍词边界）。"""
    if not bot_name or not text:
        return False
    return bot_name in text


def _score_content(texts: Sequence[str]) -> tuple[int, list[str]]:
    """计算内容加分。"""
    combined = "\n".join(t for t in texts if t)
    score = 0
    reasons: list[str] = []

    # 疑问
    if any("？" in t or "?" in t for t in texts) and any(
        term in combined for term in QUESTION_TERMS
    ):
        score += SCORE_CONTENT_QUESTION
        reasons.append("问题")

    # 请求
    request_hits = [term for term in REQUEST_TERMS if term in combined]
    if request_hits:
        score += SCORE_CONTENT_REQUEST
        reasons.append("请求")

    # 观点
    opinion_hits = [term for term in OPINION_TERMS if term in combined]
    if opinion_hits:
        score += SCORE_CONTENT_OPINION
        reasons.append("观点")

    # 长文本
    total_len = len(combined)
    if total_len >= 120:
        score += SCORE_CONTENT_LONGER_TEXT
        reasons.append("较长文本")
    elif total_len >= 40:
        score += SCORE_CONTENT_LONG_TEXT
        reasons.append("长文本")

    # 短反应惩罚
    normalized = [" ".join(t.split()).strip() for t in texts if t.strip()]
    if normalized and all(len(t) <= 8 and t in SHORT_REACTIONS for t in normalized):
        score += SCORE_SHORT_REACTION_PENALTY
        reasons.append("短反应")

    return score, reasons


def _score_pressure(pending_count: int) -> int:
    """积压压力分：对数增长，从 0 升至上限 PRESSURE_MAX_SCORE。"""
    if pending_count <= 0:
        return 0
    factor = min(1.0, log1p(pending_count) / log1p(PRESSURE_FULL_RATIO))
    return int(round(PRESSURE_MAX_SCORE * factor))


def _score_presence_penalty(recent_self_messages: int, recent_total_messages: int) -> int:
    """在场惩罚：bot 最近发言占比过高时扣分。"""
    if recent_self_messages <= 0 or recent_total_messages <= 0:
        return 0
    ratio = min(1.0, recent_self_messages / recent_total_messages)
    if ratio <= PRESENCE_FREE_RATIO:
        return 0
    span = PRESENCE_FULL_RATIO - PRESENCE_FREE_RATIO
    progress = min(1.0, (ratio - PRESENCE_FREE_RATIO) / span)
    return int(round(PRESENCE_PENALTY_MAX * progress))


def score_gate(
    text: str,
    bot_name: str,
    *,
    pending_count: int = 0,
    recent_self_messages: int = 0,
    recent_total_messages: int = 0,
) -> tuple[int, str]:
    """计算回复必要性分数和原因。

    返回 (score, detail)。score 无上限裁剪，由调用方与阈值比较。
    """
    parts: list[str] = []
    total = 0

    # 名字提及
    if _has_name_mention(text, bot_name):
        total += SCORE_NAME_MENTION
        parts.append(f"点名={SCORE_NAME_MENTION}")

    # 内容加分
    content_score, content_reasons = _score_content([text])
    if content_score:
        total += content_score
        parts.append(f"内容={content_score}({','.join(content_reasons)})")

    # 积压压力
    pressure = _score_pressure(pending_count)
    if pressure:
        total += pressure
        parts.append(f"压力={pressure}")

    # 在场惩罚
    penalty = _score_presence_penalty(recent_self_messages, recent_total_messages)
    if penalty:
        total -= penalty
        parts.append(f"在场=-{penalty}")

    total = max(0, total)
    detail = f"总分={total} " + " ".join(parts) if parts else f"总分={total}"
    return total, detail
