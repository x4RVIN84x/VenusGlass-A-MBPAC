from __future__ import annotations

import json
import sqlite3
from bisect import bisect_right
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional


class ReportStore:
    """Small durable inspection history for the local workstation."""

    TIMELINE_CATEGORIES = (
        "PASS",
        "BASEPLATE NOT FOUND",
        "X",
        "Y",
        "ANGLE",
        "OTHER FAIL",
    )

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(str(self.db_path), timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self):
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS inspection_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    recipe TEXT NOT NULL DEFAULT '',
                    result TEXT NOT NULL CHECK(result IN ('PASS', 'FAIL')),
                    cause TEXT NOT NULL DEFAULT '',
                    metrics_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_inspection_events_time "
                "ON inspection_events(timestamp)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_inspection_events_result "
                "ON inspection_events(result)"
            )

    @staticmethod
    def _timestamp(value: Optional[datetime] = None, *, milliseconds: bool = False) -> str:
        value = value or datetime.now()
        if milliseconds:
            return value.isoformat(sep=" ", timespec="milliseconds")
        # Query boundaries deliberately remain second-based so legacy rows
        # stored without fractional seconds remain included at an exact bound.
        return value.replace(microsecond=0).isoformat(sep=" ")

    def record_result(
        self,
        *,
        result: str,
        recipe: str = "",
        cause: str = "",
        metrics: Optional[Any] = None,
        timestamp: Optional[datetime] = None,
    ) -> int:
        result = str(result or "").upper().strip()
        if result not in {"PASS", "FAIL"}:
            raise ValueError("Report results must be PASS or FAIL")

        payload = metrics if metrics is not None else []
        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO inspection_events(timestamp, recipe, result, cause, metrics_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    self._timestamp(timestamp, milliseconds=True),
                    str(recipe or ""),
                    result,
                    str(cause or ""),
                    json.dumps(payload, ensure_ascii=False, default=str),
                ),
            )
            return int(cursor.lastrowid)

    @staticmethod
    def _bounds(start: datetime, end: datetime):
        if end <= start:
            end = start + timedelta(days=1)
        return ReportStore._timestamp(start), ReportStore._timestamp(end)

    @staticmethod
    def _recipe_clause(recipe: Optional[str]):
        recipe = str(recipe or "").strip()
        return ("", []) if not recipe or recipe == "All recipes" else (" AND recipe = ?", [recipe])

    def summary(self, *, start: datetime, end: datetime, recipe: Optional[str] = None) -> dict:
        start_text, end_text = self._bounds(start, end)
        recipe_clause, recipe_args = self._recipe_clause(recipe)
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN result = 'PASS' THEN 1 ELSE 0 END) AS pass_count,
                    SUM(CASE WHEN result = 'FAIL' THEN 1 ELSE 0 END) AS fail_count
                FROM inspection_events
                WHERE timestamp >= ? AND timestamp < ?
                """ + recipe_clause,
                [start_text, end_text, *recipe_args],
            ).fetchone()

        total = int(row["total"] or 0)
        passed = int(row["pass_count"] or 0)
        failed = int(row["fail_count"] or 0)
        return {
            "total": total,
            "pass": passed,
            "fail": failed,
            "pass_rate": (100.0 * passed / total) if total else 0.0,
        }

    def failure_causes(
        self, *, start: datetime, end: datetime, recipe: Optional[str] = None
    ) -> dict[str, int]:
        start_text, end_text = self._bounds(start, end)
        recipe_clause, recipe_args = self._recipe_clause(recipe)
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT cause, metrics_json
                FROM inspection_events
                WHERE timestamp >= ? AND timestamp < ? AND result = 'FAIL'
                """ + recipe_clause,
                [start_text, end_text, *recipe_args],
            ).fetchall()

        output: dict[str, int] = {}
        for row in rows:
            metric_causes = []
            try:
                metrics = json.loads(row["metrics_json"] or "[]")
                if isinstance(metrics, list):
                    metric_causes = [
                        str(metric.get("label", "")).replace(" OFFSET", "").strip()
                        for metric in metrics
                        if isinstance(metric, dict) and metric.get("passed") is False
                    ]
                    metric_causes = [cause for cause in metric_causes if cause]
            except (TypeError, ValueError, json.JSONDecodeError):
                metric_causes = []

            causes = metric_causes or [str(row["cause"] or "Unspecified failure").strip()]
            for cause in dict.fromkeys(causes):
                cause = cause or "Unspecified failure"
                output[cause] = output.get(cause, 0) + 1
        return output

    def failure_breakdown(
        self, *, start: datetime, end: datetime, recipe: Optional[str] = None
    ) -> dict[str, dict[str, int]]:
        """Return per-cause totals split by signed direction where available."""
        start_text, end_text = self._bounds(start, end)
        recipe_clause, recipe_args = self._recipe_clause(recipe)
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT cause, metrics_json
                FROM inspection_events
                WHERE timestamp >= ? AND timestamp < ? AND result = 'FAIL'
                """ + recipe_clause,
                [start_text, end_text, *recipe_args],
            ).fetchall()

        output: dict[str, dict[str, int]] = {}

        def add(label: str, direction: str = "other"):
            label = str(label or "Unspecified failure").strip() or "Unspecified failure"
            direction = direction if direction in {"positive", "negative", "zero"} else "other"
            bucket = output.setdefault(label, {"positive": 0, "negative": 0, "zero": 0, "other": 0, "total": 0})
            bucket[direction] += 1
            bucket["total"] += 1

        for row in rows:
            found_metric = False
            try:
                metrics = json.loads(row["metrics_json"] or "[]")
                if isinstance(metrics, list):
                    for metric in metrics:
                        if not isinstance(metric, dict) or metric.get("passed") is not False:
                            continue
                        label = str(metric.get("label", "")).replace(" OFFSET", "").strip()
                        if not label:
                            continue
                        add(label, str(metric.get("direction", "other")).lower())
                        found_metric = True
            except (TypeError, ValueError, json.JSONDecodeError):
                pass

            if not found_metric:
                raw_cause = str(row["cause"] or "Unspecified failure")
                for cause in raw_cause.split("+"):
                    add(cause.strip())
        return output

    def time_buckets(
        self,
        *,
        start: datetime,
        end: datetime,
        recipe: Optional[str] = None,
        granularity: str = "day",
        week_start: int = 0,
        bucket_boundaries: Optional[list[datetime]] = None,
    ) -> list[dict[str, Any]]:
        """Return complete time buckets, including zero-count buckets.

        ``segments`` is deliberately mutually exclusive: an inspection with
        multiple failed measurements is assigned to its highest-priority
        primary cause so stacked columns still add up to total inspections.
        The detailed failure view continues to count every failed metric.
        """
        start_text, end_text = self._bounds(start, end)
        recipe_clause, recipe_args = self._recipe_clause(recipe)
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT timestamp, result, cause, metrics_json FROM inspection_events "
                "WHERE timestamp >= ? AND timestamp < ?" + recipe_clause,
                [start_text, end_text, *recipe_args],
            ).fetchall()

        granularity = str(granularity).lower()
        if granularity not in {"hour", "day", "week", "month"}:
            granularity = "day"

        week_start = int(week_start) % 7

        def bucket_start(value: datetime) -> datetime:
            if granularity == "hour":
                return value.replace(minute=0, second=0, microsecond=0)
            if granularity == "week":
                day = value.date() - timedelta(days=(value.weekday() - week_start) % 7)
                return datetime.combine(day, datetime.min.time())
            if granularity == "month":
                return datetime(value.year, value.month, 1)
            return datetime.combine(value.date(), datetime.min.time())

        start_bucket = bucket_start(start)
        end_bucket = end
        if granularity == "hour":
            step = timedelta(hours=1)
        elif granularity == "week":
            step = timedelta(days=7)
        elif granularity == "month":
            step = None
        else:
            step = timedelta(days=1)

        def next_bucket(value: datetime) -> datetime:
            if granularity != "month":
                return value + step
            if value.month == 12:
                return value.replace(year=value.year + 1, month=1)
            return value.replace(month=value.month + 1)

        def empty_bucket():
            return {
                "total": 0,
                "pass": 0,
                "fail": 0,
                "segments": {category: 0 for category in self.TIMELINE_CATEGORIES},
            }

        counts: dict[datetime, dict[str, Any]] = {}
        custom_boundaries = None
        if bucket_boundaries:
            custom_boundaries = sorted({value for value in bucket_boundaries if start <= value <= end})
            if not custom_boundaries or custom_boundaries[0] != start:
                custom_boundaries.insert(0, start)
            if custom_boundaries[-1] != end:
                custom_boundaries.append(end)
            for boundary in custom_boundaries[:-1]:
                counts[boundary] = empty_bucket()
        else:
            cursor = start_bucket
            while cursor < end_bucket:
                counts[cursor] = empty_bucket()
                cursor = next_bucket(cursor)

        def primary_failure_category(row) -> str:
            labels = []
            try:
                metrics = json.loads(row["metrics_json"] or "[]")
                if isinstance(metrics, list):
                    labels.extend(
                        str(metric.get("label", "")).replace(" OFFSET", "").strip().upper()
                        for metric in metrics
                        if isinstance(metric, dict) and metric.get("passed") is False
                    )
            except (TypeError, ValueError, json.JSONDecodeError):
                pass

            if not labels:
                labels = [part.strip().upper() for part in str(row["cause"] or "").split("+")]
            labels = [label for label in labels if label]
            for category in ("BASEPLATE NOT FOUND", "X", "Y", "ANGLE"):
                if any(label == category or label.startswith(category + " ") for label in labels):
                    return category
            return "OTHER FAIL"

        for row in rows:
            try:
                timestamp = datetime.fromisoformat(str(row["timestamp"]))
            except (TypeError, ValueError):
                continue
            if custom_boundaries is not None:
                bucket_index = bisect_right(custom_boundaries, timestamp) - 1
                if bucket_index < 0 or bucket_index >= len(custom_boundaries) - 1:
                    continue
                bucket = custom_boundaries[bucket_index]
            else:
                bucket = bucket_start(timestamp)
            if bucket not in counts:
                counts[bucket] = empty_bucket()
            counts[bucket]["total"] += 1
            result = str(row["result"] or "").lower()
            if result == "pass":
                counts[bucket]["pass"] += 1
                counts[bucket]["segments"]["PASS"] += 1
            elif result == "fail":
                counts[bucket]["fail"] += 1
                counts[bucket]["segments"][primary_failure_category(row)] += 1

        output = []
        for bucket in sorted(counts):
            values = counts[bucket]
            output.append({
                "bucket": bucket,
                "label": (
                    bucket.strftime("%H:00")
                    if granularity == "hour"
                    else bucket.strftime("Wk %b %d")
                    if granularity == "week"
                    else bucket.strftime("%b")
                    if granularity == "month"
                    else bucket.strftime("%a %b %d")
                ),
                **values,
                "pass_rate": (100.0 * values["pass"] / values["total"]) if values["total"] else 0.0,
            })
        return output

    def available_recipes(self) -> list[str]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT DISTINCT recipe FROM inspection_events WHERE recipe <> '' ORDER BY recipe"
            ).fetchall()
        return [str(row["recipe"]) for row in rows]

    def inspection_events(
        self,
        *,
        start: datetime,
        end: datetime,
        recipe: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Return the confirmed per-glass inspection history, newest first."""
        start_text, end_text = self._bounds(start, end)
        recipe_clause, recipe_args = self._recipe_clause(recipe)
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT id, timestamp, recipe, result, cause, metrics_json
                FROM inspection_events
                WHERE timestamp >= ? AND timestamp < ?
                """ + recipe_clause + " ORDER BY timestamp DESC, id DESC",
                [start_text, end_text, *recipe_args],
            ).fetchall()

        events: list[dict[str, Any]] = []
        for row in rows:
            raw_timestamp = str(row["timestamp"] or "")
            try:
                timestamp = datetime.fromisoformat(raw_timestamp)
            except ValueError:
                timestamp = None
            events.append(
                {
                    "id": int(row["id"]),
                    "timestamp": timestamp,
                    "timestamp_text": raw_timestamp,
                    "recipe": str(row["recipe"] or ""),
                    "result": str(row["result"] or "").upper(),
                    "cause": str(row["cause"] or ""),
                }
            )
        return events
