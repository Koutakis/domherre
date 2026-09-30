import re

# YY(YY)MMDD, separator -/+/none, 4 trailing digits. Day covers 01-31 (normal)
# and 61-91 (samordningsnummer, day+60). Shape only, checksum validated separately.
_MONTH = r"(?:0[1-9]|1[0-2])"
_DAY = r"(?:0[1-9]|[12]\d|3[01]|6[1-9]|[78]\d|9[01])"
PERSONNUMMER_RE = re.compile(rf"\b(?:\d{{2}})?\d{{2}}{_MONTH}{_DAY}[-+]?\d{{4}}\b")

_SEPARATORS = re.compile(r"[-+]")


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(digits):
        n = int(ch) * (2 if i % 2 == 0 else 1)
        total += n - 9 if n > 9 else n
    return total % 10 == 0


def is_valid_personnummer(raw: str) -> bool:
    digits = _SEPARATORS.sub("", raw)
    if len(digits) == 12:
        digits = digits[2:]  # control digit is computed without century
    return len(digits) == 10 and _luhn_ok(digits)


def find_personnummer(text: str, validate: bool = True) -> list[tuple[int, int]]:
    return [
        (m.start(), m.end())
        for m in PERSONNUMMER_RE.finditer(text)
        if not validate or is_valid_personnummer(m.group())
    ]
