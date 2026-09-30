import os
import sys
import warnings
from typing import Any

import polars as pl

_PANDAS_WARNED = False


class PandasPerformanceWarning(UserWarning):
    pass


def _pandas() -> Any:
    return sys.modules.get("pandas")


def is_polars(df: Any) -> bool:
    return isinstance(df, pl.DataFrame)


def is_pandas(df: Any) -> bool:
    pd = _pandas()  # never imports pandas just to answer the question
    return pd is not None and isinstance(df, pd.DataFrame)


def _caller_stacklevel() -> int:
    # walk out of the package so the warning points at the caller, not our internals
    frame = sys._getframe(1)
    level = 1
    package = os.path.dirname(__file__)
    while frame is not None and os.path.dirname(frame.f_code.co_filename) == package:
        frame = frame.f_back
        level += 1
    return level


def _warn_pandas(df: Any) -> None:
    global _PANDAS_WARNED
    if _PANDAS_WARNED or os.environ.get("DOMHERRE_QUIET"):
        return
    _PANDAS_WARNED = True

    pd = _pandas()
    arrow_backed = any(
        isinstance(getattr(dt, "storage", None), str) or "arrow" in str(dt).lower()
        for dt in df.dtypes
    )
    extra = (
        ""
        if arrow_backed
        else " Columns are numpy-object backed; string[pyarrow] or polars is faster."
    )
    warnings.warn(
        f"domherre is polars-first; pandas {pd.__version__} input is converted internally."
        + extra,
        PandasPerformanceWarning,
        stacklevel=_caller_stacklevel(),
    )


def columns(df: Any) -> list[str]:
    return list(df.columns)


def check_string_columns(df: Any, cols: list[str]) -> list[str]:
    if is_polars(df):
        return [c for c in cols if df.schema[c] != pl.Utf8]

    pd = _pandas()
    bad: list[str] = []
    for c in cols:
        dtype = df[c].dtype
        if pd.api.types.is_string_dtype(dtype):
            continue
        if dtype == object and df[c].dropna().map(lambda v: isinstance(v, str)).all():
            continue  # object column holding only str, the classic pandas case
        bad.append(c)
    return bad


def to_list(df: Any, col: str) -> list[str | None]:
    if is_polars(df):
        return df[col].to_list()

    pd = _pandas()
    return [None if v is None or pd.isna(v) else str(v) for v in df[col]]


def set_column(df: Any, col: str, values: list[str | None]) -> Any:
    if is_polars(df):
        return df.with_columns(pl.Series(col, values, dtype=pl.Utf8))

    out = df.copy()
    out[col] = _pandas().Series(values, index=out.index, dtype=df[col].dtype)
    return out


def rename(df: Any, mapping: dict[str, str]) -> Any:
    if is_polars(df):
        return df.rename(mapping)
    return df.rename(columns=mapping)


def validate(df: Any) -> None:
    if is_polars(df):
        return
    if is_pandas(df):
        _warn_pandas(df)
        return
    raise TypeError(
        f"expected a polars or pandas DataFrame, got {type(df).__name__}. "
        "polars is the supported path; pandas works but is converted internally."
    )
