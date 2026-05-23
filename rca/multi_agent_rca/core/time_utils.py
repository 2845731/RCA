from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

UTC_PLUS_8 = timezone(timedelta(hours=8))

MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def normalize_epoch(value: object) -> Optional[int]:
    try:
        if value is None:
            return None
        ts = int(float(value))
        if abs(ts) >= 10_000_000_000:
            ts //= 1000
        return ts
    except Exception:
        return None


def epoch_to_local(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), UTC_PLUS_8).strftime("%Y-%m-%d %H:%M:%S")


def local_to_epoch(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC_PLUS_8)
    return int(dt.timestamp())


def _parse_clock(text: str) -> Tuple[int, int]:
    m = re.match(r"\s*(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?", text)
    if not m:
        raise ValueError(f"Invalid clock value: {text}")
    hour = int(m.group(1))
    minute = int(m.group(2))
    suffix = m.group(3)
    if suffix:
        suffix = suffix.lower()
        if suffix == "pm" and hour != 12:
            hour += 12
        if suffix == "am" and hour == 12:
            hour = 0
    return hour, minute


def parse_instruction_time_range(instruction: str) -> Tuple[str, str, int, int]:
    """Parse OpenRCA natural-language time ranges into local datetimes.

    OpenRCA query templates are regular enough that date + first two clock
    values recover the diagnostic window. The timestamps in the benchmark are
    labeled in UTC+8.
    """

    text = " ".join(str(instruction).replace("\n", " ").split())
    date_match = re.search(
        r"\b(" + "|".join(MONTHS.keys()) + r")\s+(\d{1,2}),\s*(\d{4})\b",
        text,
        re.IGNORECASE,
    )
    if not date_match:
        raise ValueError(f"Cannot parse date from instruction: {instruction}")
    month = MONTHS[date_match.group(1).lower()]
    day = int(date_match.group(2))
    year = int(date_match.group(3))

    clock_matches = re.findall(r"\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?", text)
    if len(clock_matches) < 2:
        raise ValueError(f"Cannot parse time range from instruction: {instruction}")
    start_h, start_m = _parse_clock(clock_matches[0])
    end_h, end_m = _parse_clock(clock_matches[1])

    start = datetime(year, month, day, start_h, start_m, tzinfo=UTC_PLUS_8)
    end = datetime(year, month, day, end_h, end_m, tzinfo=UTC_PLUS_8)
    if end <= start:
        end += timedelta(days=1)
    return (
        start.strftime("%Y-%m-%d %H:%M:%S"),
        end.strftime("%Y-%m-%d %H:%M:%S"),
        local_to_epoch(start),
        local_to_epoch(end),
    )


def day_dir_from_epoch(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), UTC_PLUS_8).strftime("%Y_%m_%d")
