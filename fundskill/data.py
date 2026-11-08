"""Loading factors and fund returns.

Everything downstream expects:
  factors : index = month-end Timestamp, columns e.g. MKT_RF, SMB, HML, MOM, RF
            values in DECIMALS (0.012 = 1.2%)
  returns : wide, index = month-end Timestamp, one column per fund,
            monthly EXCESS returns (fund return - RF), NaN when not alive.

Sources this is written for
---------------------------
* US factors: Ken French's data library, files
  "F-F_Research_Data_Factors_CSV.zip" and "F-F_Momentum_Factor_CSV.zip".
  Download and pass the path (zip or extracted csv).
* UK factors: Exeter (Gregory, Tharyan & Christidis) monthly UK factors.
  Column names differ by vintage, so use `load_factor_csv` with a mapping.
* Funds, CRSP Mutual Fund database on WRDS: monthly returns table with
  columns crsp_fundno, caldt, mret -> use `long_to_wide`.
* Funds, Bloomberg / Yahoo export: one column per fund -> `load_wide_prices`
  or `load_wide_returns`.
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import pandas as pd


def _month_end(s) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(s)) + pd.offsets.MonthEnd(0)


def _read_text(path: str | Path) -> str:
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
            return z.read(name).decode("latin-1")
    return path.read_text(encoding="latin-1")


def parse_ken_french(text: str) -> pd.DataFrame:
    """Parse the MONTHLY block of a Ken French CSV (percent -> decimal).

    The files have a text preamble, a header row starting with a comma,
    YYYYMM rows, then an annual block that we ignore.
    """
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines)
                 if l.strip().startswith(",") and i + 1 < len(lines)
                 and re.match(r"^\s*\d{6}\s*,", lines[i + 1]))
    header = [h.strip() for h in lines[start].split(",")][1:]
    rows = []
    for l in lines[start + 1:]:
        if not re.match(r"^\s*\d{6}\s*,", l):
            break
        parts = [p.strip() for p in l.split(",")]
        rows.append([parts[0]] + [float(x) for x in parts[1:len(header) + 1]])
    df = pd.DataFrame(rows, columns=["yyyymm"] + header)
    df.index = _month_end(pd.to_datetime(df.pop("yyyymm"), format="%Y%m"))
    df = df / 100.0
    rename = {"Mkt-RF": "MKT_RF", "Mom": "MOM", "WML": "MOM"}
    return df.rename(columns=lambda c: rename.get(c, c.replace("-", "_")))


def load_ken_french(three_factor_path, momentum_path=None) -> pd.DataFrame:
    """US Fama-French 3 factors (+ momentum if given) and RF."""
    ff = parse_ken_french(_read_text(three_factor_path))
    if momentum_path is not None:
        mom = parse_ken_french(_read_text(momentum_path))
        ff = ff.join(mom[["MOM"]], how="inner")
    return ff


def load_factor_csv(path, date_col: str, mapping: dict, date_format: str | None = None,
                    percent: bool = False) -> pd.DataFrame:
    """Generic loader, e.g. for the Exeter UK factors.

    mapping: {source column -> our name}, our names among
             MKT_RF, SMB, HML, MOM, RF. If the file gives the raw market
             return (not excess), map it to 'MKT' and it will be converted.
    """
    raw = pd.read_csv(path)
    df = raw[list(mapping)].rename(columns=mapping).astype(float)
    df.index = _month_end(pd.to_datetime(raw[date_col].astype(str), format=date_format))
    if percent:
        df = df / 100.0
    if "MKT" in df and "MKT_RF" not in df:
        df["MKT_RF"] = df.pop("MKT") - df["RF"]
    return df.sort_index()


def long_to_wide(df: pd.DataFrame, fund_col="crsp_fundno", date_col="caldt",
                 ret_col="mret") -> pd.DataFrame:
    """CRSP-style long table -> wide returns (raw, not yet excess)."""
    d = df[[fund_col, date_col, ret_col]].dropna(subset=[ret_col]).copy()
    d[date_col] = _month_end(d[date_col])
    wide = d.pivot_table(index=date_col, columns=fund_col, values=ret_col, aggfunc="last")
    return wide.sort_index()


def load_wide_prices(path, date_col: str = "Date") -> pd.DataFrame:
    """One column of prices (ideally total-return / accumulation units) per
    fund -> monthly simple returns."""
    px = pd.read_csv(path, parse_dates=[date_col]).set_index(date_col).sort_index()
    monthly = px.resample("ME").last()
    return monthly.pct_change(fill_method=None)


def to_excess(returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Subtract the risk-free rate month by month."""
    rf = factors["RF"].reindex(returns.index)
    return returns.sub(rf, axis=0)
