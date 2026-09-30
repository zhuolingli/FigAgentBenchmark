from __future__ import annotations


def validate_month(value: str) -> str:
    if len(value) != 4 or not value.isdigit() or not 1 <= int(value[2:]) <= 12:
        raise ValueError(f"invalid YYMM month: {value!r}")
    return value


def previous_month(value: str) -> str:
    validate_month(value)
    year, month = int(value[:2]), int(value[2:])
    month -= 1
    if month == 0:
        year -= 1
        month = 12
    return f"{year:02d}{month:02d}"


def descending_months(start: str, stop: str) -> list[str]:
    start = validate_month(start)
    stop = validate_month(stop)
    result = []
    current = start
    while True:
        result.append(current)
        if current == stop:
            return result
        current = previous_month(current)
        if len(result) > 1200 or current > start:
            raise ValueError(f"stop month {stop} must not be newer than start month {start}")
