"""Data-contract validation against the provider's data structure spec (v2.1.0).

The contract (configs/data_contract.yaml) is a machine-readable copy of
docs/Data_structure_CSV_files.docx. Every pipeline run validates the raw files
against it and writes:

  reports/metrics/data_contract_report.csv    one row per check (table, column, check, severity, result)
  reports/metrics/data_contract_summary.json  counts, missing files, coverage vs the spec

Severity: a missing required file or column, a duplicate key or a broken foreign key is an
`error` (the run stops when `contract.strict` is true, the default for errors). Type, null,
range, allowed-value, pattern and row-count deviations are `warning`s: the business rules
(R1-R15) decide how such rows are treated, and the report documents how many there are.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import PROJECT_ROOT, Config
from .data import SPREADSHEET_ROW_LIMIT, resolve_files

TRUE_FALSE = {"TRUE", "FALSE", "T", "F", "1", "0", "1.0", "0.0"}


def load_contract(path: str | Path | None = None) -> dict:
    path = Path(path) if path else PROJECT_ROOT / "configs" / "data_contract.yaml"
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def discover(cfg: Config, table: str, spec: dict) -> tuple[list[Path], list[Path]]:
    """(used, superseded) files for a contract table: the pipeline's `raw_files` entry when it has
    one (so the validator checks exactly what the pipeline reads), else the contract patterns."""
    return resolve_files(cfg.path("raw"), cfg["raw_files"].get(table, spec["patterns"]))


def _row(table, file, column, check, severity, ok, n_bad=0, n=0, detail=""):
    return {"table": table, "file": file, "column": column, "check": check, "severity": severity,
            "result": "pass" if ok else "fail", "n_bad": int(n_bad), "share_bad": (n_bad / n) if n else 0.0,
            "detail": detail}


def _type_ok(s: pd.Series, kind: str) -> tuple[bool, str]:
    v = s.dropna()
    if v.empty:
        return True, "all null"
    if kind == "int":
        if pd.api.types.is_integer_dtype(v):
            return True, str(s.dtype)
        if pd.api.types.is_float_dtype(v):
            ok = bool(np.all(np.isclose(v, np.round(v))))
            return ok, "float column with integer values (nullable int)" if ok else "non-integer values"
        return False, str(s.dtype)
    if kind == "float":
        return bool(pd.api.types.is_numeric_dtype(v) and not pd.api.types.is_bool_dtype(v)), str(s.dtype)
    if kind == "bool":
        if pd.api.types.is_bool_dtype(v):
            return True, "bool"
        vals = set(v.astype(str).str.upper().str.strip())
        return vals <= TRUE_FALSE, f"codes {sorted(vals)[:6]}"
    if kind == "datetime":
        parsed = pd.to_datetime(v.head(5000), errors="coerce", format="ISO8601")
        return bool(parsed.notna().all()), "ISO 8601"
    return bool(not pd.api.types.is_numeric_dtype(v) or pd.api.types.is_object_dtype(v)), str(s.dtype)


def validate_frame(table: str, file: str, spec: dict, df: pd.DataFrame) -> list[dict]:
    """Column-level checks of one file (or file part) against its contract entry."""
    rows = []
    cols = spec["columns"]
    n = len(df)
    missing = [c for c in cols if c not in df.columns]
    extra = [c for c in df.columns if c not in cols]
    rows.append(_row(table, file, "*", "columns_present", "error", not missing, len(missing), len(cols),
                     f"missing {missing}" if missing else f"{len(cols)} columns as specified"))
    if extra:
        rows.append(_row(table, file, "*", "no_extra_columns", "warning", False, len(extra), len(df.columns), f"extra {extra}"))
    for col, c in cols.items():
        if col not in df.columns:
            continue
        s = df[col]
        ok, det = _type_ok(s, c["type"])
        rows.append(_row(table, file, col, f"type_{c['type']}", "warning", ok, 0 if ok else s.notna().sum(), n, det))
        if c.get("spec_type") and c["type"] == "bool" and not pd.api.types.is_numeric_dtype(s.dropna()):
            rows.append(_row(table, file, col, "encoding", "warning", False, s.notna().sum(), n,
                             f"spec says {c['spec_type']} 1.0/0.0; file stores TRUE/FALSE text (parsed as boolean)"))
        nulls = int(s.isna().sum())
        if not c.get("nullable", False):
            rows.append(_row(table, file, col, "not_null", "warning", nulls == 0, nulls, n, f"{nulls} nulls"))
        v = s.dropna()
        if "range" in c and len(v) and pd.api.types.is_numeric_dtype(v):
            lo, hi = c["range"]
            bad = int(((v < lo) | (v > hi)).sum())
            rows.append(_row(table, file, col, f"range[{lo},{hi}]", "warning", bad == 0, bad, n,
                             f"observed [{v.min():.4g}, {v.max():.4g}]"))
        if "values" in c and len(v):
            allowed = c["values"]
            if all(isinstance(a, (int, float)) for a in allowed) and pd.api.types.is_numeric_dtype(v):
                bad_mask = ~np.isclose(v.to_numpy()[:, None], np.asarray(allowed, float)[None, :]).any(axis=1)
                bad_vals = [x.item() if hasattr(x, "item") else x for x in sorted(pd.unique(v[bad_mask]))[:8]]
            else:
                bad_mask = ~v.astype(str).isin([str(a) for a in allowed])
                bad_vals = sorted(pd.unique(v[bad_mask].astype(str)))[:8]
            bad = int(bad_mask.sum())
            rows.append(_row(table, file, col, "allowed_values", "warning", bad == 0, bad, n,
                             f"unexpected {bad_vals}" if bad else f"within {allowed}"))
        if "pattern" in c and len(v):
            rx = re.compile(c["pattern"])
            bad = int((~v.astype(str).map(lambda x: bool(rx.match(x)))).sum())
            rows.append(_row(table, file, col, "pattern", "warning", bad == 0, bad, n, c["pattern"]))
    key = spec.get("key")
    if key and all(k in df.columns for k in key):
        dups = int(df.duplicated(key).sum())
        rows.append(_row(table, file, ",".join(key), "unique_key", "error", dups == 0, dups, n, f"{dups} duplicate keys"))
    return rows


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, na_values=["NA", ""], keep_default_na=True, low_memory=False)


def validate_all(cfg: Config, contract: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Validate every contract table that is present; returns (report, summary)."""
    contract = contract or load_contract()
    raw = cfg.path("raw")
    rows, frames, summary = [], {}, {"files": {}, "superseded_files": {}, "missing_files": [], "coverage": {}}
    order = ["players", "combine", "careers", "games", "player_play", "tracking", "combine_tracking"]
    for table in order:
        spec = contract["files"][table]
        paths, superseded = discover(cfg, table, spec)
        if superseded:
            summary["superseded_files"][table] = [p.name for p in superseded]
            rows.append(_row(table, "+".join(p.name for p in superseded), "*", "superseded_part", "warning", False,
                             len(superseded), len(superseded) + len(paths), "split part ignored: the full file is present"))
        if not paths:
            sev = "error" if spec.get("required") else "warning"
            rows.append(_row(table, spec["spec_name"], "*", "file_present", sev, False, 1, 1,
                             f"no file matching {spec['patterns']}"))
            summary["missing_files"].append({"table": table, "spec_name": spec["spec_name"], "required": spec.get("required", False)})
            continue
        summary["files"][table] = [p.name for p in paths]
        parts = []
        for p in paths:
            df = _read(p)
            rows += validate_frame(table, p.name, spec, df)
            if len(df) == SPREADSHEET_ROW_LIMIT:
                rows.append(_row(table, p.name, "*", "spreadsheet_row_limit", "warning", False, 1, 1,
                                 f"exactly {SPREADSHEET_ROW_LIMIT:,} data rows: the file was probably cut by a spreadsheet"))
            keep = [c for c in set(sum([list(map(str.strip, k.split(","))) for k in spec.get("soft_foreign_keys", {})], []))
                    | set(spec.get("key", [])) | set(spec.get("foreign_keys", {})) | {"nfl_id", "game_id", "play_id", "event_id", "season", "season_type"}
                    if c in df.columns]
            parts.append(df[keep].copy())
            del df
        tab = pd.concat(parts, ignore_index=True)
        frames[table] = tab
        n = len(tab)
        if len(paths) > 1 and spec.get("key") and all(k in tab.columns for k in spec["key"]):
            dups = int(tab.duplicated(spec["key"]).sum())
            rows.append(_row(table, "+".join(p.name for p in paths), ",".join(spec["key"]), "unique_key_across_parts",
                             "error", dups == 0, dups, n, f"{dups} duplicate keys across file parts"))
        if "expected_rows" in spec:
            exp = spec["expected_rows"]
            rows.append(_row(table, "+".join(p.name for p in paths), "*", "row_count", "warning", n == exp, abs(n - exp), exp,
                             f"{n:,} rows vs {exp:,} in the spec ({(n - exp) / exp:+.2%})"))
        if "expected_attempts" in spec and "event_id" in tab:
            got = tab["event_id"].nunique()
            rows.append(_row(table, "+".join(p.name for p in paths), "event_id", "attempt_count", "warning",
                             got == spec["expected_attempts"], abs(got - spec["expected_attempts"]), spec["expected_attempts"],
                             f"{got:,} drill attempts vs {spec['expected_attempts']:,} in the spec"))
        # foreign keys
        for col, ref in spec.get("foreign_keys", {}).items():
            rt, rc = ref.split(".")
            if rt in frames and col in tab and rc in frames[rt]:
                vals = tab[col].dropna()
                bad = int((~vals.isin(frames[rt][rc])).sum())
                rows.append(_row(table, "+".join(p.name for p in paths), col, f"foreign_key->{ref}", "error", bad == 0, bad, len(vals),
                                 f"{bad} rows reference unknown {ref}"))
        for cols, ref in spec.get("soft_foreign_keys", {}).items():
            kc = [c.strip() for c in cols.split(",")]
            if ref in frames and all(c in tab for c in kc) and all(c in frames[ref] for c in kc):
                left = tab[kc].drop_duplicates()
                m = left.merge(frames[ref][kc].drop_duplicates(), on=kc, how="left", indicator=True)
                bad = int((m["_merge"] == "left_only").sum())
                rows.append(_row(table, "+".join(p.name for p in paths), cols, f"soft_foreign_key->{ref}", "warning", bad == 0, bad, len(left),
                                 f"{bad} of {len(left):,} {cols} combinations not in {ref}"))
        # coverage vs the spec (tracking: seasons and players)
        if table == "tracking" and "games" in frames:
            g = pd.concat([pd.read_csv(p) for p in discover(cfg, "games", contract["files"]["games"])[0]])
            seasons = sorted(g[g.game_id.isin(tab.game_id.unique())].season.unique().tolist())
            types = g[g.game_id.isin(tab.game_id.unique())].season_type.value_counts().to_dict()
            exp_s = spec.get("expected_seasons", [])
            miss_s = [s for s in exp_s if s not in seasons]
            players = int(tab.nfl_id.nunique())
            games = int(tab.game_id.nunique())
            summary["coverage"]["tracking"] = {"seasons": seasons, "missing_seasons": miss_s, "games": games,
                                               "games_by_type": types, "players": players,
                                               "expected_players": spec.get("expected_players")}
            rows.append(_row(table, "+".join(p.name for p in paths), "season", "season_coverage", "warning", not miss_s,
                             len(miss_s), len(exp_s), f"seasons {seasons}; missing {miss_s}; {games} games {types}"))
            rows.append(_row(table, "+".join(p.name for p in paths), "nfl_id", "player_coverage", "warning",
                             players >= spec.get("expected_players", 0), spec.get("expected_players", 0) - players,
                             spec.get("expected_players", 0), f"{players} of {spec.get('expected_players')} prospects tracked"))
    report = pd.DataFrame(rows)
    handling = contract.get("handling", {}) or {}

    def _handle(r) -> str:
        chk = r["check"].split("[")[0].split("->")[0]
        if r["result"] == "pass":
            return ""
        return handling.get(f"{r['table']}.{r['column']}.{chk}", handling.get(f"{r['table']}.*.{chk}", ""))

    report["handling"] = report.apply(_handle, axis=1)
    summary["n_checks"] = len(report)
    summary["n_errors"] = int(((report.severity == "error") & (report.result == "fail")).sum())
    summary["n_warnings"] = int(((report.severity == "warning") & (report.result == "fail")).sum())
    summary["status"] = "fail" if summary["n_errors"] else ("pass_with_warnings" if summary["n_warnings"] else "pass")
    return report, summary


def write_report(cfg: Config, report: pd.DataFrame, summary: dict) -> None:
    m = cfg.path("metrics")
    report.to_csv(m / "data_contract_report.csv", index=False)
    with open(m / "data_contract_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
