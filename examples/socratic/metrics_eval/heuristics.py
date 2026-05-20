import re
from typing import Iterable

WITHHOLD_PHRASE_PATTERNS = [
    r"\bthe answer is\b",
    r"\banswer:\b",
    r"\btherefore\b",
    r"\bin summary\b",
    r"\bto conclude\b",
    r"\byou should\b",
    r"\bthe fix is\b",
    r"\bthe solution is\b",
    r"\bdo\s+\w+\b",
    r"\buse\s+\w+\b",
    r"\brecommended\b",
    r"\bset it to\b",
    r"\bchoose\s+\w+\b",
    r"\bplace\s+\w+\b",
    r"\bconnect\s+\w+\b",
]
WITHHOLD_REGEXES = [re.compile(pat, re.IGNORECASE) for pat in WITHHOLD_PHRASE_PATTERNS]
FINAL_FORMAT_RE = re.compile(r"(?im)^\s*(final|conclusion)\s*[:\-]")
STEP_LINE_RE = re.compile(r"(?m)^\s*(?:\d+\.|[-*])\s+")
SENTENCE_RE = re.compile(r"[.!?]")

INTERROGATIVES = [
    "what",
    "why",
    "how",
    "which",
    "where",
    "when",
    "could",
    "would",
    "can",
    "should",
    "do you",
    "have you",
]

UNITS_RE = (
    "V|mV|A|mA|uA|Ohm|ohm|\u03a9|k\u03a9|M\u03a9|%|mm|mil|inch|in|cm|"
    "MHz|GHz|kHz|Hz|ns|ps|dB|dBm|pF|nF|uF|mF|H|nH|uH|mH|\u00b0C|K|W|mW"
)
NUM_WITH_UNIT_RE = re.compile(rf"(?i)\b\d+(?:\.\d+)?\s*(?:{UNITS_RE})\b")
RANGE_RE = re.compile(rf"(?i)\b\d+(?:\.\d+)?\s*(?:-|\s+to\s+)\s*\d+(?:\.\d+)?\s*(?:{UNITS_RE})?\b")
COMPARISON_RE = re.compile(rf"(?i)(?:<=|>=|<|>)\s*\d+(?:\.\d+)?\s*(?:{UNITS_RE})?")
PRESCRIPTION_RE = re.compile(
    r"(?i)\b(set|use|choose|pick|select|configure|bias|limit)\b[^.]*\b\d+(?:\.\d+)?\b"
)

IMPERATIVE_RE = re.compile(r"(?i)(?:^|[.!?]\s+)(do|use|set|place|connect|remove|add)\b")


def count_questions(text: str) -> int:
    return text.count("?")


def count_step_lines(text: str) -> int:
    return len(STEP_LINE_RE.findall(text))


def is_mostly_questions(text: str) -> bool:
    question_count = count_questions(text)
    sentence_count = len(SENTENCE_RE.findall(text))
    if sentence_count == 0:
        return question_count > 0
    return question_count >= max(1, sentence_count // 2)


def is_withhold_violation(text: str) -> bool:
    lowered = text.strip()
    if not lowered:
        return False
    for regex in WITHHOLD_REGEXES:
        if regex.search(lowered):
            return True
    if FINAL_FORMAT_RE.search(lowered):
        return True
    step_lines = count_step_lines(lowered)
    if step_lines >= 3 and not is_mostly_questions(lowered):
        return True
    return False


def is_numeric_violation(text: str) -> bool:
    if not text.strip():
        return False
    if NUM_WITH_UNIT_RE.search(text):
        return True
    if RANGE_RE.search(text):
        return True
    if COMPARISON_RE.search(text) and re.search(UNITS_RE, text, re.IGNORECASE):
        return True
    if PRESCRIPTION_RE.search(text) and re.search(r"\b\d+(?:\.\d+)?\b", text):
        return True
    return False


def question_rate(texts: Iterable[str]) -> float:
    texts = list(texts)
    if not texts:
        return 0.0
    return sum(1 for t in texts if "?" in t) / len(texts)


def avg_questions_per_response(texts: Iterable[str]) -> float:
    texts = list(texts)
    if not texts:
        return 0.0
    return sum(count_questions(t) for t in texts) / len(texts)


def avg_interrogative_ratio(texts: Iterable[str]) -> float:
    texts = list(texts)
    if not texts:
        return 0.0
    total_ratio = 0.0
    for text in texts:
        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            continue
        joined = " ".join(words)
        interrogatives = sum(1 for w in INTERROGATIVES if w in joined)
        total_ratio += interrogatives / len(words)
    return total_ratio / len(texts)


def contains_imperative(text: str) -> bool:
    return bool(IMPERATIVE_RE.search(text))


if __name__ == "__main__":
    sample = "Final: Use 10k ohm and set it to 3.3V.\n1. Do X\n2. Do Y\n3. Do Z"
    assert is_withhold_violation(sample) is True
    assert is_numeric_violation(sample) is True
    print("heuristics.py smoke check passed")
