"""拟人化后处理：碎句拆分、错别字注入、打字节奏。

移植自 MaiBot (Mai-with-u/MaiBot) typo_generator.py / utils.py / math_utils.py，GPL-3.0。
已删除 main/format_typo_info/set_params/correction_suggestion 等调试表面。
jieba 词典模块级一次性加载；禁用文件缓存。
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from collections.abc import Sequence
from math import exp

# ── jieba + pypinyin：首次使用时惰性构建，prewarm() 可在启动时预热 ──
import jieba  # type: ignore[import-untyped]
from pypinyin import Style, pinyin  # type: ignore[import-untyped]

_MAX_DISCORD_MESSAGE = 1980

# ── Discord 免疫区域正则 ─────────────────────────────────────────────
_IMMUNITY_RE = re.compile(
    r"```[\s\S]*?```"  # fenced code block
    r"|`[^`\n]+`"  # inline code
    r"|https?://[^\s\u4e00-\u9fff，。；！？、）】》："
    "''…]+"  # URL（不吞后续中文）
    r"|<@!?\d{15,22}>"  # user mention
    r"|<@&\d{15,22}>"  # role mention
    r"|<#\d{15,22}>"  # channel mention
    r"|<a?:\w{2,32}:\d{15,22}>"  # custom emoji
    r"|[^\S\r\n]*@\S+"  # @text
)


# ═══════════════════════════════════════════════════════════════════════
#  ChineseTypoGenerator（精简移植）
# ═══════════════════════════════════════════════════════════════════════


def _build_pinyin_dict() -> dict[str, list[str]]:
    """创建拼音→同音字映射（模块级单次构建）。"""
    result: dict[str, list[str]] = defaultdict(list)
    for code in range(0x4E00, 0x9FFF):
        char = chr(code)
        try:
            py = pinyin(char, style=Style.TONE3)[0][0]
            result[py].append(char)
        except Exception:
            continue
    return dict(result)


def _build_char_frequency() -> dict[str, float]:
    """从 jieba 词典构建汉字频率表（模块级单次构建，无文件缓存）。"""
    import os

    char_freq: dict[str, int] = defaultdict(int)
    dict_path = os.path.join(os.path.dirname(jieba.__file__), "dict.txt")
    with open(dict_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            word, freq_str = parts[0], parts[1]
            for char in word:
                if "\u4e00" <= char <= "\u9fff":
                    char_freq[char] += int(freq_str)
    if not char_freq:
        return {}
    max_freq = max(char_freq.values())
    return {char: freq / max_freq * 1000 for char, freq in char_freq.items()}


# 模块级单例
_PINYIN_DICT: dict[str, list[str]] = {}
_CHAR_FREQUENCY: dict[str, float] = {}


def _ensure_loaded() -> None:
    global _PINYIN_DICT, _CHAR_FREQUENCY
    if not _PINYIN_DICT:
        _PINYIN_DICT = _build_pinyin_dict()
    if not _CHAR_FREQUENCY:
        _CHAR_FREQUENCY = _build_char_frequency()


def prewarm() -> None:
    """启动时在工作线程调用：提前构建拼音/字频表，避免首次回复卡顿。"""
    _ensure_loaded()


def _is_chinese(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"


def _get_similar_tone_pinyin(py: str) -> str:
    """获取相似声调的拼音。"""
    if not py or len(py) < 1:
        return py
    if not py[-1].isdigit():
        return f"{py}1"
    base = py[:-1]
    tone = int(py[-1])
    possible = [t for t in (1, 2, 3, 4) if t != tone]
    return base + str(random.choice(possible))


def _calculate_replacement_probability(
    orig_freq: float, target_freq: float, max_freq_diff: float = 200
) -> float:
    if target_freq > orig_freq:
        return 1.0
    diff = orig_freq - target_freq
    if diff > max_freq_diff:
        return 0.0
    return exp(-3 * diff / max_freq_diff)


def _get_similar_frequency_chars(
    char: str,
    py: str,
    *,
    min_freq: float = 5,
    tone_error_rate: float = 0.2,
    max_freq_diff: float = 200,
    num_candidates: int = 5,
) -> list[str] | None:
    _ensure_loaded()
    homophones: list[str] = []
    if random.random() < tone_error_rate:
        homophones.extend(_PINYIN_DICT.get(_get_similar_tone_pinyin(py), []))
    homophones.extend(_PINYIN_DICT.get(py, []))
    if not homophones:
        return None

    orig_freq = _CHAR_FREQUENCY.get(char, 0)
    freq_diff = [
        (h, _CHAR_FREQUENCY.get(h, 0))
        for h in homophones
        if h != char and _CHAR_FREQUENCY.get(h, 0) >= min_freq
    ]
    if not freq_diff:
        return None

    candidates = []
    for h, freq in freq_diff:
        prob = _calculate_replacement_probability(orig_freq, freq, max_freq_diff)
        if prob > 0:
            candidates.append((h, prob))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in candidates[:num_candidates]]


def _create_typo_sentence(
    sentence: str,
    *,
    error_rate: float = 0.03,
    min_freq: float = 5,
    tone_error_rate: float = 0.2,
    word_replace_rate: float = 0.3,
    max_freq_diff: float = 200,
) -> str:
    """为中文句子注入同音字错别字（精简版，无纠正建议）。"""
    _ensure_loaded()
    result: list[str] = []
    words = list(jieba.cut(sentence))

    for word in words:
        if all(not _is_chinese(c) for c in word):
            result.append(word)
            continue

        word_pinyin = [py[0] for py in pinyin(word, style=Style.TONE3)]

        # 整词替换尝试
        if len(word) > 1 and random.random() < word_replace_rate:
            word_homophones = _get_word_homophones(word, word_pinyin)
            if word_homophones:
                typo_word = random.choice(word_homophones)
                result.append(typo_word)
                continue

        # 单字替换
        if len(word) == 1:
            char = word
            py = word_pinyin[0]
            if random.random() < error_rate:
                similar = _get_similar_frequency_chars(
                    char,
                    py,
                    min_freq=min_freq,
                    tone_error_rate=tone_error_rate,
                    max_freq_diff=max_freq_diff,
                )
                if similar:
                    typo_char = random.choice(similar)
                    orig_freq = _CHAR_FREQUENCY.get(char, 0)
                    typo_freq = _CHAR_FREQUENCY.get(typo_char, 0)
                    if random.random() < _calculate_replacement_probability(
                        orig_freq, typo_freq, max_freq_diff
                    ):
                        result.append(typo_char)
                        continue
            result.append(char)
        else:
            # 多字词内逐字替换（概率降低）
            word_result: list[str] = []
            for char, py in zip(word, word_pinyin, strict=False):
                word_error_rate = error_rate * (0.7 ** (len(word) - 1))
                if random.random() < word_error_rate:
                    similar = _get_similar_frequency_chars(
                        char,
                        py,
                        min_freq=min_freq,
                        tone_error_rate=tone_error_rate,
                        max_freq_diff=max_freq_diff,
                    )
                    if similar:
                        typo_char = random.choice(similar)
                        orig_freq = _CHAR_FREQUENCY.get(char, 0)
                        typo_freq = _CHAR_FREQUENCY.get(typo_char, 0)
                        if random.random() < _calculate_replacement_probability(
                            orig_freq, typo_freq, max_freq_diff
                        ):
                            word_result.append(typo_char)
                            continue
                word_result.append(char)
            result.append("".join(word_result))

    return "".join(result)


def _get_word_homophones(word: str, word_pinyin: list[str], *, min_freq: float = 5) -> list[str]:
    """获取整词的同音词（高频有意义词语）。"""
    _ensure_loaded()
    if len(word) <= 1:
        return []
    candidates_per_char = []
    for py in word_pinyin:
        chars = _PINYIN_DICT.get(py, [])
        if not chars:
            return []
        candidates_per_char.append(chars)

    import itertools

    homophones: list[tuple[str, float]] = []
    for combo in itertools.product(*candidates_per_char):
        new_word = "".join(combo)
        if new_word == word:
            continue
        char_avg = sum(_CHAR_FREQUENCY.get(c, 0) for c in new_word) / len(new_word)
        if char_avg >= min_freq:
            homophones.append((new_word, char_avg))
    homophones.sort(key=lambda x: x[1], reverse=True)
    return [w for w, _ in homophones[:5]]


# ═══════════════════════════════════════════════════════════════════════
#  碎句拆分 + 合并
# ═══════════════════════════════════════════════════════════════════════


def _find_immunity_zones(text: str) -> list[tuple[int, int]]:
    """返回需要免疫拆分的区间列表。"""
    return [(m.start(), m.end()) for m in _IMMUNITY_RE.finditer(text)]


def _in_immunity(pos: int, zones: Sequence[tuple[int, int]]) -> bool:
    for start, end in zones:
        if start <= pos < end:
            return True
        if pos < start:
            break
    return False


def _split_sentences(text: str) -> list[str]:
    """将文本拆分为句子碎片（按标点/空格/换行，引号内不拆）。"""
    text = re.sub(r"\n\s*\n+", "\n", text)
    text = re.sub(r"\n\s*([，,。;\s])", r"\n\1", text)
    text = re.sub(r"([，,。;\s])\s*\n", r"\1\n", text)

    len_text = len(text)
    if len_text < 3:
        return [text]

    zones = _find_immunity_zones(text)

    # 引号追踪
    quote_chars = {
        '"',
        "'",
        "\u201c",
        "\u201d",
        "\u2018",
        "\u2019",
        "\u300c",
        "\u300d",
        "\u300e",
        "\u300f",
    }
    inside_quote = [False] * len_text
    in_quote = False
    current_quote_char = ""
    for idx, ch in enumerate(text):
        if ch in quote_chars:
            if not in_quote:
                in_quote = True
                current_quote_char = ch
            else:
                if ch == current_quote_char or (
                    ch in {'"', "'"} and current_quote_char in {'"', "'"}
                ):
                    in_quote = False
                    current_quote_char = ""
        else:
            inside_quote[idx] = in_quote

    separators = {"，", ",", " ", "。", ";", "\n"}
    segments: list[tuple[str, str]] = []
    current_segment = ""

    i = 0
    while i < len_text:
        char = text[i]
        if char in separators:
            if inside_quote[i] or _in_immunity(i, zones):
                current_segment += char
            elif char == "\n":
                if current_segment:
                    segments.append((current_segment, char))
                current_segment = ""
            else:
                # 检查冒号
                can_split = True
                if i > 0 and text[i - 1] in {":", "："}:
                    can_split = False
                if i < len_text - 1 and text[i + 1] in {":", "："}:
                    can_split = False
                # 空格 + 英文/数字相邻
                if can_split and char == " " and 0 < i < len_text - 1:
                    prev, nxt = text[i - 1], text[i + 1]
                    if prev in {"-", "—"} or nxt in {"-", "—"}:
                        can_split = False
                    else:
                        prev_alnum = prev.isdigit() or prev.isascii() and prev.isalpha()
                        next_alnum = nxt.isdigit() or nxt.isascii() and nxt.isalpha()
                        if prev_alnum and next_alnum:
                            can_split = False

                if can_split:
                    if current_segment:
                        segments.append((current_segment, char))
                    elif char in {" ", "\n"}:
                        segments.append(("", char))
                    current_segment = ""
                else:
                    current_segment += char
        else:
            current_segment += char
        i += 1

    if current_segment:
        segments.append((current_segment, ""))
    segments = [(c, s) for c, s in segments if c or s]
    if not segments:
        return [text] if text else []

    # 概率合并
    if len_text < 12:
        merge_prob = 0.8
    elif len_text < 32:
        merge_prob = 0.4
    else:
        merge_prob = 0.3

    merged: list[tuple[str, str]] = []
    idx = 0
    while idx < len(segments):
        content, sep = segments[idx]
        if idx + 1 < len(segments) and content and sep != "\n" and random.random() < merge_prob:
            next_content, next_sep = segments[idx + 1]
            if next_content:
                merged.append((content + sep + next_content, next_sep))
            else:
                merged.append((content, next_sep))
            idx += 2
        else:
            merged.append((content, sep))
            idx += 1

    final = [re.sub(r"[^\S\r\n]*[\r\n]+[^\S\r\n]*", " ", c).strip() for c, _ in merged if c]
    return [s for s in final if s]


def _merge_to_max(sentences: list[str], max_count: int) -> list[str]:
    """按顺序合并句子到指定条数以内。"""
    if len(sentences) <= max_count or max_count <= 0:
        return sentences[:max_count] if max_count > 0 else []
    result: list[str] = []
    n = len(sentences)
    start = 0
    for group_idx in range(max_count):
        remaining = n - start
        remaining_groups = max_count - group_idx
        group_size = (remaining + remaining_groups - 1) // remaining_groups
        result.append("".join(sentences[start : start + group_size]))
        start += group_size
    return result


def _hard_split(text: str, limit: int = _MAX_DISCORD_MESSAGE) -> list[str]:
    """最终安全硬拆（按换行或字符位置）。"""
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    remaining = text
    while len(remaining) > limit:
        split = remaining.rfind("\n", 0, limit)
        if split <= 0:
            split = limit
        parts.append(remaining[:split])
        remaining = remaining[split:].lstrip("\n")
    if remaining:
        parts.append(remaining)
    return parts


# ═══════════════════════════════════════════════════════════════════════
#  typing_delay
# ═══════════════════════════════════════════════════════════════════════


def typing_delay(fragment: str, *, typing_speed: float = 12.0) -> float:
    """根据片段长度计算打字时间（秒）。

    typing_speed 为每秒字符数。
    """
    if typing_speed <= 0:
        return 0.0
    chinese_time = 1.0 / typing_speed  # 每中文字符秒
    english_time = chinese_time * 0.5  # 英文更快
    total = 0.0
    for char in fragment:
        if "\u4e00" <= char <= "\u9fff":
            total += chinese_time
        else:
            total += english_time
    return total


# ═══════════════════════════════════════════════════════════════════════
#  公共 API
# ═══════════════════════════════════════════════════════════════════════


def fragments(
    text: str,
    *,
    typo_rate: float = 0.02,
    max_fragments: int = 4,
    limit: int = _MAX_DISCORD_MESSAGE,
) -> list[str]:
    """拆分 → 合并 → 注入错别字 → 硬拆兜底 → 截断到 max_fragments。

    免疫区域（代码块、URL、@提及、自定义 emoji）内的文本不做拆分或错别字替换。
    空白输入返回 []（由调用方兜底）；任何返回片段都不超过 limit。
    """
    if not text or not text.strip():
        return []
    sentences = _split_sentences(text)
    sentences = _merge_to_max(sentences, max_fragments)

    result: list[str] = []
    for sentence in sentences:
        if typo_rate > 0:
            # 仅对非免疫区域注入错别字
            sentence = _inject_typo_safe(sentence, typo_rate)
        # 硬拆兜底
        for chunk in _hard_split(sentence, limit):
            result.append(chunk)

    # 截断到 max_fragments（多余合并到最后一个；合并若超限则再硬拆，保证单条 ≤ limit）
    if len(result) > max_fragments and max_fragments > 0:
        merged_tail = "".join(result[max_fragments - 1 :])
        result = result[: max_fragments - 1] + _hard_split(merged_tail, limit)

    return result


def _inject_typo_safe(text: str, typo_rate: float) -> str:
    """在非免疫区域注入错别字。"""
    zones = _find_immunity_zones(text)
    if not zones:
        return (
            _create_typo_sentence(text, error_rate=typo_rate)
            if random.random() < typo_rate
            else text
        )

    # 分段处理：免疫区间原样保留，非免疫区间可能注入错别字
    result_parts: list[str] = []
    pos = 0
    for z_start, z_end in zones:
        if pos < z_start:
            segment = text[pos:z_start]
            if random.random() < typo_rate:
                segment = _create_typo_sentence(segment, error_rate=typo_rate)
            result_parts.append(segment)
        result_parts.append(text[z_start:z_end])
        pos = z_end
    if pos < len(text):
        segment = text[pos:]
        if random.random() < typo_rate:
            segment = _create_typo_sentence(segment, error_rate=typo_rate)
        result_parts.append(segment)

    return "".join(result_parts)
