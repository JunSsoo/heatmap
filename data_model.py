"""
data_model.py
=============

Phase 1 of the scientific graph generator: explicit data table types inspired
by GraphPad Prism.

The existing app (heatmap.py / charts.py / stats_utils.py / presets.py) assumes
a simple wide DataFrame where:
    - row 0 = column headers
    - column 0 = row labels
    - the remaining cells are numeric values, with each column treated as a
      separate group.

For Prism-level features (Two-way ANOVA, LC50/EC50 curve fitting, Kaplan-Meier,
etc.) we need to know what each column *means*. This module defines five table
types and provides helpers to validate, parse, and route DataFrames into the
correct analyses and chart renderers.

Table types
-----------
- COLUMN         : one column per group (current default).
- XY             : first numeric column is X, remaining columns are Y series.
- GROUPED        : 2-factor design (factor1 in col 1, factor2 in cols 2+).
- DOSE_RESPONSE  : dose in col 1, response replicates in cols 2+.
- SURVIVAL       : time/event/group table for Kaplan-Meier (placeholder, Phase 6+).

Typical usage
-------------
    import pandas as pd
    from data_model import (
        DataTableType, TABLE_SPECS,
        parse_table, validate_table, detect_table_type,
        to_long, available_analyses, available_charts,
    )

    df = pd.read_csv("experiment.csv", header=None)
    ttype = detect_table_type(df)
    ok, msg = validate_table(df, ttype)
    if not ok:
        raise ValueError(msg)
    parsed = parse_table(df, ttype)
    long_df = to_long(df, ttype)
    analyses = available_analyses(ttype)
    charts = available_charts(ttype)

Note
----
The input DataFrames follow the project convention: the *first row* holds
column headers and the *first column* holds row labels. Callers typically load
a CSV with ``header=None`` and pass the raw frame; this module handles header
extraction internally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Enum and spec dataclass
# ---------------------------------------------------------------------------


class DataTableType(Enum):
    """Supported data table types (Prism-inspired)."""

    COLUMN = "column"
    XY = "xy"
    GROUPED = "grouped"
    DOSE_RESPONSE = "dose_response"
    SURVIVAL = "survival"


@dataclass
class TableSpec:
    """Describes what each column means for a given table type.

    Attributes
    ----------
    table_type : DataTableType
        The table type this spec describes.
    column_roles : dict
        Maps column index (0-based, relative to the raw DataFrame including
        the row-label column) to a human-readable role name. The special
        key -1 means "all remaining columns share this role".
    required_min_cols : int
        Minimum number of columns (including row labels) for the table to
        make sense.
    description : str
        Korean description used in the UI.
    """

    table_type: DataTableType
    column_roles: Dict[int, str] = field(default_factory=dict)
    required_min_cols: int = 2
    description: str = ""


# One spec per table type. The key -1 indicates "all remaining columns".
TABLE_SPECS: Dict[DataTableType, TableSpec] = {
    DataTableType.COLUMN: TableSpec(
        table_type=DataTableType.COLUMN,
        column_roles={0: "row_name", -1: "group"},
        required_min_cols=2,
        description="Column 표: 첫 열은 행 이름, 이후 각 열은 하나의 그룹(반복 측정치). "
                    "t-검정, 일원배치 분산분석 등에 사용합니다.",
    ),
    DataTableType.XY: TableSpec(
        table_type=DataTableType.XY,
        column_roles={0: "row_name", 1: "x", -1: "y_series"},
        required_min_cols=3,
        description="XY 표: 첫 열은 행 이름, 두 번째 열은 X값(수치), 이후 열들은 Y 시리즈. "
                    "산점도, 선형/비선형 회귀, 상관분석에 사용합니다.",
    ),
    DataTableType.GROUPED: TableSpec(
        table_type=DataTableType.GROUPED,
        column_roles={0: "row_name", 1: "factor1", -1: "factor2_level"},
        required_min_cols=3,
        description="Grouped 표: 2-요인 설계. 첫 열은 행 이름, 두 번째 열은 요인1(범주), "
                    "이후 열들은 요인2의 수준별 측정치. 이원배치 분산분석, 그룹 막대그래프에 사용합니다.",
    ),
    DataTableType.DOSE_RESPONSE: TableSpec(
        table_type=DataTableType.DOSE_RESPONSE,
        column_roles={0: "row_name", 1: "dose", -1: "response_replicate"},
        required_min_cols=3,
        description="Dose-Response 표: 첫 열은 행 이름, 두 번째 열은 용량(수치, 로그스케일 권장), "
                    "이후 열들은 반응(% 치사율/효과) 반복측정. LC50/EC50 곡선 적합에 사용합니다.",
    ),
    DataTableType.SURVIVAL: TableSpec(
        table_type=DataTableType.SURVIVAL,
        column_roles={0: "subject_id", 1: "time", 2: "event", 3: "group"},
        required_min_cols=4,
        description="Survival 표: 첫 열은 대상 ID, 두 번째 열은 시간(수치), "
                    "세 번째 열은 사건(0=중도절단, 1=발생), 네 번째 열은 그룹. "
                    "Kaplan-Meier 생존분석에 사용합니다. (Phase 6+)",
    ),
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_header(df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
    """Split off the header row.

    The project convention is that the first row of the raw DataFrame holds
    column names. If the DataFrame already has meaningful (non-RangeIndex)
    column labels, we use those instead.

    Returns
    -------
    header : list of str
        Column names, one per column.
    body : pd.DataFrame
        DataFrame with the header row removed and reset index.
    """
    if df is None or df.shape[0] == 0:
        return [], pd.DataFrame()

    # If columns look like default RangeIndex (0, 1, 2, ...), treat row 0 as
    # the header row. Otherwise, columns already carry the headers.
    cols_are_default = list(df.columns) == list(range(df.shape[1]))
    if cols_are_default:
        header = [str(x) for x in df.iloc[0].tolist()]
        body = df.iloc[1:].reset_index(drop=True).copy()
    else:
        header = [str(c) for c in df.columns]
        body = df.reset_index(drop=True).copy()
    return header, body


def _to_numeric_array(series: pd.Series) -> np.ndarray:
    """Coerce a pandas Series to float ndarray, turning unparseable entries
    into ``np.nan``."""
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _column_is_numeric(series: pd.Series, threshold: float = 0.7) -> bool:
    """Return True if at least ``threshold`` fraction of non-null entries
    parse as numbers."""
    s = series.dropna()
    if len(s) == 0:
        return False
    coerced = pd.to_numeric(s, errors="coerce")
    return coerced.notna().mean() >= threshold


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_table(df: pd.DataFrame, table_type: DataTableType) -> Tuple[bool, str]:
    """Validate ``df`` against the given table type.

    Returns
    -------
    (is_valid, error_message_in_korean)
        If valid, the message is an empty string.
    """
    if df is None or df.empty:
        return False, "데이터가 비어 있습니다."

    spec = TABLE_SPECS.get(table_type)
    if spec is None:
        return False, f"지원하지 않는 표 유형입니다: {table_type!r}"

    header, body = _split_header(df)
    ncols = df.shape[1]
    if ncols < spec.required_min_cols:
        return (
            False,
            f"{table_type.value} 표는 최소 {spec.required_min_cols}개의 열이 필요합니다. "
            f"(현재 {ncols}개)",
        )
    if body.shape[0] == 0:
        return False, "데이터 행이 없습니다. 헤더 외에 최소 1개의 데이터 행이 필요합니다."

    # Type-specific checks.
    if table_type == DataTableType.COLUMN:
        # All columns from index 1 onwards should be (mostly) numeric.
        for j in range(1, ncols):
            if not _column_is_numeric(body.iloc[:, j]):
                name = header[j] if j < len(header) else f"col{j}"
                return False, f"'{name}' 열에 숫자로 변환할 수 없는 값이 너무 많습니다."
        return True, ""

    if table_type == DataTableType.XY:
        if not _column_is_numeric(body.iloc[:, 1]):
            return False, "XY 표의 두 번째 열(X값)은 숫자여야 합니다."
        for j in range(2, ncols):
            if not _column_is_numeric(body.iloc[:, j]):
                name = header[j] if j < len(header) else f"col{j}"
                return False, f"Y 시리즈 '{name}' 열에 숫자가 아닌 값이 너무 많습니다."
        return True, ""

    if table_type == DataTableType.GROUPED:
        # factor1 (col 1) should be categorical; factor2 levels (cols 2+) numeric.
        factor1 = body.iloc[:, 1].dropna()
        if factor1.empty:
            return False, "요인1(두 번째 열)에 값이 없습니다."
        for j in range(2, ncols):
            if not _column_is_numeric(body.iloc[:, j]):
                name = header[j] if j < len(header) else f"col{j}"
                return False, f"요인2 수준 '{name}' 열은 숫자여야 합니다."
        if factor1.nunique() < 2:
            return False, "이원배치 분석을 위해서는 요인1에 최소 2개의 수준이 필요합니다."
        if ncols - 2 < 2:
            return False, "이원배치 분석을 위해서는 요인2에 최소 2개의 수준(열)이 필요합니다."
        return True, ""

    if table_type == DataTableType.DOSE_RESPONSE:
        if not _column_is_numeric(body.iloc[:, 1]):
            return False, "용량(두 번째 열)은 숫자여야 합니다."
        dose_vals = _to_numeric_array(body.iloc[:, 1])
        if np.all(np.isnan(dose_vals)):
            return False, "용량 열에 유효한 숫자가 없습니다."
        for j in range(2, ncols):
            if not _column_is_numeric(body.iloc[:, j]):
                name = header[j] if j < len(header) else f"col{j}"
                return False, f"반응 반복측정 '{name}' 열은 숫자여야 합니다."
        # Need at least 3 distinct doses for meaningful curve fitting.
        unique_doses = np.unique(dose_vals[~np.isnan(dose_vals)])
        if len(unique_doses) < 3:
            return False, "용량-반응 곡선 적합을 위해서는 최소 3개의 서로 다른 용량이 필요합니다."
        return True, ""

    if table_type == DataTableType.SURVIVAL:
        if ncols < 4:
            return False, "생존 분석 표는 최소 4개 열(ID, 시간, 사건, 그룹)이 필요합니다."
        if not _column_is_numeric(body.iloc[:, 1]):
            return False, "시간(두 번째 열)은 숫자여야 합니다."
        event_col = _to_numeric_array(body.iloc[:, 2])
        valid_events = event_col[~np.isnan(event_col)]
        if not np.all(np.isin(valid_events.astype(int), [0, 1])):
            return False, "사건(세 번째 열) 값은 0(중도절단) 또는 1(사건 발생)이어야 합니다."
        if body.iloc[:, 3].dropna().empty:
            return False, "그룹(네 번째 열)에 값이 없습니다."
        return True, ""

    return False, f"알 수 없는 표 유형입니다: {table_type!r}"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_table(df: pd.DataFrame, table_type: DataTableType) -> dict:
    """Parse ``df`` according to the specified table type.

    Returns a dictionary whose keys depend on ``table_type`` (see module
    docstring and the function-level comments below). Raises ``ValueError``
    with a Korean message if the data fails validation.
    """
    ok, msg = validate_table(df, table_type)
    if not ok:
        raise ValueError(msg)

    header, body = _split_header(df)
    row_names = body.iloc[:, 0].astype(str).tolist()

    if table_type == DataTableType.COLUMN:
        groups: Dict[str, np.ndarray] = {}
        for j in range(1, body.shape[1]):
            name = header[j] if j < len(header) else f"col{j}"
            arr = _to_numeric_array(body.iloc[:, j])
            groups[name] = arr[~np.isnan(arr)]
        return {
            "table_type": table_type,
            "row_names": row_names,
            "groups": groups,
        }

    if table_type == DataTableType.XY:
        x = _to_numeric_array(body.iloc[:, 1])
        y_series: Dict[str, np.ndarray] = {}
        for j in range(2, body.shape[1]):
            name = header[j] if j < len(header) else f"y{j - 1}"
            y_series[name] = _to_numeric_array(body.iloc[:, j])
        return {
            "table_type": table_type,
            "row_names": row_names,
            "x": x,
            "x_name": header[1] if len(header) > 1 else "x",
            "y_series": y_series,
        }

    if table_type == DataTableType.GROUPED:
        factor1_name = header[1] if len(header) > 1 else "factor1"
        factor1_values = body.iloc[:, 1].astype(str).tolist()
        factor2_names = [
            header[j] if j < len(header) else f"level{j - 1}"
            for j in range(2, body.shape[1])
        ]
        # Group rows by factor1 level; each level maps factor2 name -> np.array
        # of values drawn from rows sharing that factor1 level.
        levels: Dict[str, Dict[str, np.ndarray]] = {}
        long_records: List[dict] = []
        for row_idx, lvl in enumerate(factor1_values):
            bucket = levels.setdefault(lvl, {name: [] for name in factor2_names})
            for j, name in enumerate(factor2_names, start=2):
                val = pd.to_numeric(body.iat[row_idx, j], errors="coerce")
                bucket[name].append(val)
                if not pd.isna(val):
                    long_records.append(
                        {
                            "row_name": row_names[row_idx],
                            "factor1": lvl,
                            "factor2": name,
                            "value": float(val),
                        }
                    )
        # Convert list buckets to numpy arrays (keeping NaNs so shape is preserved).
        levels_arr: Dict[str, Dict[str, np.ndarray]] = {
            lvl: {name: np.asarray(vals, dtype=float) for name, vals in buckets.items()}
            for lvl, buckets in levels.items()
        }
        long_df = pd.DataFrame(long_records, columns=["row_name", "factor1", "factor2", "value"])
        return {
            "table_type": table_type,
            "row_names": row_names,
            "factor1_name": factor1_name,
            "factor2_names": factor2_names,
            "levels": levels_arr,
            "long_df": long_df,
        }

    if table_type == DataTableType.DOSE_RESPONSE:
        dose = _to_numeric_array(body.iloc[:, 1])
        responses: List[np.ndarray] = []
        for j in range(2, body.shape[1]):
            responses.append(_to_numeric_array(body.iloc[:, j]))
        # Heuristic: if dose spans more than ~2 orders of magnitude, treat as log.
        finite_dose = dose[np.isfinite(dose) & (dose > 0)]
        if finite_dose.size >= 2:
            ratio = finite_dose.max() / finite_dose.min()
            dose_log = bool(ratio >= 100)
        else:
            dose_log = False
        return {
            "table_type": table_type,
            "row_names": row_names,
            "dose": dose,
            "dose_name": header[1] if len(header) > 1 else "dose",
            "responses": responses,
            "response_names": [
                header[j] if j < len(header) else f"rep{j - 1}"
                for j in range(2, body.shape[1])
            ],
            "dose_log": dose_log,
        }

    if table_type == DataTableType.SURVIVAL:
        time = _to_numeric_array(body.iloc[:, 1])
        event = _to_numeric_array(body.iloc[:, 2])
        group = body.iloc[:, 3].astype(str).to_numpy()
        return {
            "table_type": table_type,
            "row_names": row_names,
            "subject_id": row_names,  # alias
            "time": time,
            "event": event.astype(int, copy=False) if not np.any(np.isnan(event)) else event,
            "group": group,
        }

    raise ValueError(f"알 수 없는 표 유형입니다: {table_type!r}")


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


_XY_HEADER_HINTS = {
    "x", "time", "t", "dose", "conc", "concentration", "ppm", "ppb",
    "log_dose", "logdose", "log(dose)", "hour", "day", "min", "sec",
}


def detect_table_type(df: pd.DataFrame) -> DataTableType:
    """Best-guess the table type based on column structure and numeric patterns.

    Heuristics (in order):
    1. 4+ columns, col 1 numeric, col 2 values all in {0, 1}, col 3 categorical
       -> SURVIVAL.
    2. 3+ columns, col 1 numeric spanning >=100x, cols 2+ numeric
       -> DOSE_RESPONSE.
    3. 3+ columns, col 1 numeric with a recognizable X-axis header
       (``x``, ``time``, ``dose``, ``conc``...), cols 2+ numeric -> XY.
    4. 3+ columns, col 1 categorical (non-numeric), cols 2+ numeric -> GROUPED.
    5. Otherwise -> COLUMN. (This is the conservative default that matches the
       existing app's "every column is a group" assumption.)
    """
    if df is None or df.empty:
        return DataTableType.COLUMN

    header, body = _split_header(df)
    ncols = df.shape[1]
    if ncols < 2 or body.shape[0] == 0:
        return DataTableType.COLUMN

    # 1. Survival: col 2 all 0/1 and col 1 numeric and col 3 categorical.
    if ncols >= 4 and _column_is_numeric(body.iloc[:, 1]):
        event_vals = pd.to_numeric(body.iloc[:, 2], errors="coerce").dropna()
        if not event_vals.empty and set(event_vals.astype(int).unique()).issubset({0, 1}):
            # Require col 3 to look categorical (not purely numeric) so we don't
            # accidentally catch plain numeric tables.
            if not _column_is_numeric(body.iloc[:, 3]):
                return DataTableType.SURVIVAL

    col1_numeric = ncols >= 2 and _column_is_numeric(body.iloc[:, 1])
    rest_numeric = ncols >= 3 and all(
        _column_is_numeric(body.iloc[:, j]) for j in range(2, ncols)
    )

    if ncols >= 3 and col1_numeric and rest_numeric:
        # Distinguish dose-response from plain XY by dose range.
        dose_vals = pd.to_numeric(body.iloc[:, 1], errors="coerce").dropna()
        positive = dose_vals[dose_vals > 0]
        if len(positive) >= 2 and (positive.max() / positive.min()) >= 100:
            return DataTableType.DOSE_RESPONSE
        # Only call this XY if the col-1 header explicitly advertises an X axis.
        col1_header = header[1].strip().lower() if len(header) > 1 else ""
        if col1_header in _XY_HEADER_HINTS:
            return DataTableType.XY
        # Otherwise fall through to the COLUMN default (current app convention:
        # every non-label column is a group of replicates).
        return DataTableType.COLUMN

    # Grouped: col 1 is categorical text, cols 2+ numeric.
    if ncols >= 3 and not col1_numeric and rest_numeric:
        return DataTableType.GROUPED

    return DataTableType.COLUMN


# ---------------------------------------------------------------------------
# Wide -> long conversion
# ---------------------------------------------------------------------------


def to_long(df: pd.DataFrame, table_type: DataTableType) -> pd.DataFrame:
    """Convert wide-format ``df`` to long format suitable for statsmodels.

    Column conventions per table type:
      - COLUMN        : ['row_name', 'group', 'value']
      - XY            : ['row_name', 'x', 'series', 'value']  (value = y)
      - GROUPED       : ['row_name', 'factor1', 'factor2', 'value']
      - DOSE_RESPONSE : ['row_name', 'dose', 'replicate', 'value']
      - SURVIVAL      : ['subject_id', 'time', 'event', 'group']
    """
    ok, msg = validate_table(df, table_type)
    if not ok:
        raise ValueError(msg)

    header, body = _split_header(df)

    if table_type == DataTableType.COLUMN:
        records = []
        for i in range(body.shape[0]):
            row_name = str(body.iat[i, 0])
            for j in range(1, body.shape[1]):
                name = header[j] if j < len(header) else f"col{j}"
                val = pd.to_numeric(body.iat[i, j], errors="coerce")
                if not pd.isna(val):
                    records.append({"row_name": row_name, "group": name, "value": float(val)})
        return pd.DataFrame(records, columns=["row_name", "group", "value"])

    if table_type == DataTableType.XY:
        records = []
        for i in range(body.shape[0]):
            row_name = str(body.iat[i, 0])
            x_val = pd.to_numeric(body.iat[i, 1], errors="coerce")
            for j in range(2, body.shape[1]):
                name = header[j] if j < len(header) else f"y{j - 1}"
                y_val = pd.to_numeric(body.iat[i, j], errors="coerce")
                if not pd.isna(y_val):
                    records.append(
                        {
                            "row_name": row_name,
                            "x": float(x_val) if not pd.isna(x_val) else np.nan,
                            "series": name,
                            "value": float(y_val),
                        }
                    )
        return pd.DataFrame(records, columns=["row_name", "x", "series", "value"])

    if table_type == DataTableType.GROUPED:
        parsed = parse_table(df, table_type)
        return parsed["long_df"].copy()

    if table_type == DataTableType.DOSE_RESPONSE:
        records = []
        for i in range(body.shape[0]):
            row_name = str(body.iat[i, 0])
            dose_val = pd.to_numeric(body.iat[i, 1], errors="coerce")
            for j in range(2, body.shape[1]):
                name = header[j] if j < len(header) else f"rep{j - 1}"
                resp = pd.to_numeric(body.iat[i, j], errors="coerce")
                if not pd.isna(resp):
                    records.append(
                        {
                            "row_name": row_name,
                            "dose": float(dose_val) if not pd.isna(dose_val) else np.nan,
                            "replicate": name,
                            "value": float(resp),
                        }
                    )
        return pd.DataFrame(records, columns=["row_name", "dose", "replicate", "value"])

    if table_type == DataTableType.SURVIVAL:
        records = []
        for i in range(body.shape[0]):
            subj = str(body.iat[i, 0])
            t = pd.to_numeric(body.iat[i, 1], errors="coerce")
            e = pd.to_numeric(body.iat[i, 2], errors="coerce")
            g = str(body.iat[i, 3])
            records.append(
                {
                    "subject_id": subj,
                    "time": float(t) if not pd.isna(t) else np.nan,
                    "event": int(e) if not pd.isna(e) else np.nan,
                    "group": g,
                }
            )
        return pd.DataFrame(records, columns=["subject_id", "time", "event", "group"])

    raise ValueError(f"알 수 없는 표 유형입니다: {table_type!r}")


# ---------------------------------------------------------------------------
# Analysis / chart routing
# ---------------------------------------------------------------------------


_ANALYSES: Dict[DataTableType, List[str]] = {
    DataTableType.COLUMN: [
        "descriptive",
        "t-test",
        "one-way ANOVA",
        "Kruskal-Wallis",
        "normality",
        "variance_equality",
    ],
    DataTableType.XY: [
        "descriptive",
        "linear_regression",
        "correlation",
        "nonlinear_regression",
    ],
    DataTableType.GROUPED: [
        "two-way ANOVA",
        "tukey_hsd",
        "dunnetts",
    ],
    DataTableType.DOSE_RESPONSE: [
        "hill_4pl",
        "probit",
        "ec50",
    ],
    DataTableType.SURVIVAL: [
        "kaplan_meier",
        "log_rank",
    ],
}


_CHARTS: Dict[DataTableType, List[str]] = {
    DataTableType.COLUMN: ["bar", "box", "violin", "heatmap"],
    DataTableType.XY: ["scatter", "line", "xy_error"],
    DataTableType.GROUPED: ["grouped_bar", "interaction"],
    DataTableType.DOSE_RESPONSE: ["dose_response_curve"],
    DataTableType.SURVIVAL: ["kaplan_meier"],
}


def available_analyses(table_type: DataTableType) -> List[str]:
    """Return the list of analyses applicable to ``table_type``."""
    return list(_ANALYSES.get(table_type, []))


def available_charts(table_type: DataTableType) -> List[str]:
    """Return the list of chart types applicable to ``table_type``."""
    return list(_CHARTS.get(table_type, []))


__all__ = [
    "DataTableType",
    "TableSpec",
    "TABLE_SPECS",
    "parse_table",
    "validate_table",
    "detect_table_type",
    "to_long",
    "available_analyses",
    "available_charts",
]
