"""Preparación de datos: contrato, ingesta, limpieza y métricas de movimiento y rendimiento.

Proyecto NFL (CGMT, Combine-to-Game Movement Translation): ¿se traduce el movimiento de un
prospecto en las pruebas del NFL Combine a su movimiento y rendimiento en partidos de temporada
regular? Este archivo reúne todas las funciones de limpieza y preparación; `train.py` las usa
para entrenar, evaluar y registrar los modelos en MLflow.

Etapas que ejecuta este archivo (`python fuentes/datos_prep.py`)
  0 contrato   valida los CSV originales (datos/datos_ini) contra el contrato de datos
               (configuracion/contrato_datos.yaml, copia legible por máquina de la especificación
               del proveedor, documentos/Data_structure_CSV_files.docx)
  1 ingesta    lee y tipa cada tabla, comprueba claves e integridad referencial y escribe el
               manifiesto con el hash SHA-256 de cada archivo (versionado de datos)
  2 limpieza   tabla de prospectos con las reglas de negocio R1-R7
  3 movimiento tracking de partido -> métricas por jugada -> perfiles por jugador e índice SBI (R8-R9)
  3b campo     player_play -> jugadas sobre lo esperado -> métricas por jugador (R10-R15)
  3c Combine   tracking de las pruebas del Combine -> variables fiables por prospecto (R16)
  Los resultados limpios se guardan en datos/datos_limp (parquet) y las auditorías en
  reportes/metricas.

Uso
  python fuentes/datos_prep.py                      contrato + ingesta + limpieza (etapa "datos")
  python fuentes/datos_prep.py --etapa contrato     solo el contrato de datos
  python fuentes/datos_prep.py --etapa verificar    comprueba los hashes del manifiesto
  python fuentes/datos_prep.py --config configuracion/rapida.yaml   configuración rápida (CI)

Los datos originales se guardan comprimidos (`.csv.gz`) para respetar el límite de 100 MB por
archivo de GitHub; pandas los lee directamente y el manifiesto guarda el hash del contenido
descomprimido, de modo que la versión de datos (`data_version`) no depende de la compresión.
Los nombres de funciones, columnas y métricas se mantienen en inglés porque los comparten el
código, los archivos de métricas y el registro de MLflow.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats

__version__ = "3.1.0"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("nfl")


# =============================================================================
# Configuración y reproducibilidad
# =============================================================================
@dataclass(frozen=True)
class Config:
    """Configuración del proyecto (configuracion/config.yaml), accesible como diccionario."""

    raw: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    def path(self, key: str) -> Path:
        """Ruta absoluta de una entrada de `paths`; la carpeta se crea si no existe."""
        p = PROJECT_ROOT / self.raw["paths"][key]
        (p if p.suffix == "" else p.parent).mkdir(parents=True, exist_ok=True)
        return p


def _project_path(path: str | os.PathLike) -> Path:
    """Ruta relativa al directorio actual si existe; si no, relativa a la raíz del proyecto."""
    p = Path(path)
    return p if p.is_absolute() or p.exists() else PROJECT_ROOT / p


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Lee la configuración (por defecto configuracion/config.yaml)."""
    path = _project_path(path) if path else PROJECT_ROOT / "configuracion" / "config.yaml"
    with open(path, encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh))


def set_global_seed(seed: int) -> None:
    """Fija la semilla de todos los generadores aleatorios; los modelos reciben además `random_state`."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def _save(df: pd.DataFrame, cfg: Config, name: str, key: str = "processed") -> None:
    """Guarda una tabla en la carpeta `paths[key]` (parquet o CSV según la extensión)."""
    path = cfg.path(key) / name
    if name.endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _dump(obj, cfg: Config, name: str) -> None:
    """Guarda un diccionario como JSON en reportes/metricas."""
    with open(cfg.path("metrics") / name, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))


# =============================================================================
# Ingesta: esquemas, tipado, manifiesto (versionado de datos)
# =============================================================================
# El manifiesto (datos/datos_ini/manifiesto.json) es el contrato de versionado de datos: cada
# ejecución registra el SHA-256 de cada archivo original, así cualquier resultado se puede
# rastrear hasta los bytes exactos con que se entrenó (un equivalente ligero de DVC).
#
# Los nombres de archivo se resuelven con patrones (`raw_files` en la configuración), de modo que
# se encuentran tanto los nombres de la especificación (game_tracking_2024.csv, player_play.csv,
# combine_tracking.csv) como las partes entregadas (game_tracking_2024_2.csv, player_play_3.csv).
# Una parte `<raíz>_<n>.csv` se ignora cuando también existe el archivo completo `<raíz>.csv`.
# El tracking se lee archivo por archivo (`iter_tracking`) para que temporadas completas quepan en memoria.

# Columnas y tipos esperados por tabla (tipos con nulos de pandas donde la especificación los admite).
SCHEMAS: dict[str, dict[str, str]] = {
    "combine": {
        "draft_year": "int64", "nfl_id": "int64", "combine_position": "string",
        "combine_height": "float64", "combine_weight": "float64", "hand_size": "float64",
        "arm_length": "float64", "wing_span": "float64", "ten_yd_split": "float64",
        "forty": "float64", "vertical": "float64", "broad_jump": "float64",
        "three_cone": "float64", "short_shuttle": "float64", "bench_reps": "float64",
        "ngs_athleticism_score": "float64", "ngs_college_production_score": "float64",
        "ngs_final_score": "float64",
    },
    "players": {
        "nfl_id": "int64", "display_name": "string", "draft_year": "int64",
        "nfl_position": "string", "birth_date": "string", "college_name": "string",
        "college_conference": "string", "draft_round": "Int64",
        "draft_pick_within_round": "Int64", "draft_overall_pick": "Int64",
    },
    "careers": {
        "nfl_id": "int64", "career_offensive_snaps": "int64",
        "career_defensive_snaps": "int64", "career_special_teams_snaps": "int64",
        "career_games_active": "int64", "career_games_started": "int64",
        "ap_all_pro_1st_team": "int64", "ap_all_pro_2nd_team": "int64",
        "pro_bowl_original_ballot": "int64",
    },
    "games": {
        "game_id": "int64", "game_key": "int64", "season": "int64",
        "season_type": "string", "week": "int64", "home_team_abbr": "string",
        "visitor_team_abbr": "string",
    },
    "tracking": {
        "game_id": "int64", "play_id": "int64", "nfl_id": "int64", "time": "string",
        "x": "float64", "y": "float64", "s": "float64", "a": "float64",
        "dis": "float64", "o": "float64", "dir": "float64", "event": "string",
    },
    # Archivo 3 de la especificación (opcional): tracking a 10 Hz de los intentos en las pruebas del
    # Combine. Las filas del balón llevan el nfl_id del jugador de ese intento.
    "combine_tracking": {
        "draft_year": "int64", "event_id": "string", "nfl_id": "Int64", "entity_type": "string",
        "time": "string", "drill_type": "string", "drill_name": "string", "attempt": "int64",
        "x": "float64", "y": "float64", "s": "float64", "a": "float64", "dis": "float64", "dir": "float64",
    },
    # Las banderas booleanas llegan como TRUE/FALSE/NA, por eso se leen como texto y se
    # convierten explícitamente en clean_player_play.
    "player_play": {
        "game_id": "int64", "play_id": "int64", "team_abbr": "string", "nfl_id": "int64",
        "lined_up_position": "string", "play_description": "string", "quarter": "int64", "down": "int64",
        "yards_to_go": "int64", "possession_team": "string", "yardline_side": "string",
        "yardline_number": "int64", "game_clock": "string", "pre_snap_home_score": "int64",
        "pre_snap_visitor_score": "int64", "play_direction": "string",
        "pre_snap_home_team_win_probability": "float64", "pre_snap_visitor_team_win_probability": "float64",
        "home_team_win_probability_added": "float64", "visitor_team_win_probility_added": "float64",
        "expected_points_added": "float64", "expected_points": "float64", "offense_formation": "string",
        "receiver_alignment": "string", "pass_result": "string", "pass_length": "float64",
        "play_nullified_by_penalty": "string", "penalty_yards": "float64", "pre_penalty_yards_gained": "int64",
        "yards_gained": "int64", "team_coverage_man_zone": "string", "team_coverage_type": "string",
        "in_motion_at_ball_snap": "string", "target": "string", "rec_yards": "float64", "rush_yards": "float64",
        "yards_after_catch": "float64", "fumble": "float64", "fumble_lost": "float64", "route_ran": "string",
        "separation_at_pass_forward": "float64", "cushion": "float64", "expected_yards_after_catch": "float64",
        "dropback_duration": "float64", "extended_sack_allowed": "float64", "pass_rushers_encountered": "float64",
        "peak_pressure_probability_allowed": "float64", "pressure_allowed": "string", "sack_allowed": "float64",
        "time_to_pressure_allowed": "float64", "quick_pressure": "string", "unblocked_pressure": "string",
        "blitzing": "string", "player_get_off": "float64", "sack": "float64", "tackle": "float64",
        "assist": "float64", "caused_forced_fumble": "float64", "recovered_fumble": "string",
        "tackle_for_loss": "string", "time_to_pressure": "float64", "time_to_qb_hurry": "float64",
        "coverage_assignment": "string", "coverage_assignment_at_snap": "string",
    },
}

# Columnas de player_play que usa el análisis (el encabezado completo de 64 columnas se valida igual).
PLAYER_PLAY_USECOLS = [
    "game_id", "play_id", "nfl_id", "team_abbr", "lined_up_position", "down", "yards_to_go",
    "pass_result", "play_nullified_by_penalty", "target", "route_ran", "separation_at_pass_forward",
    "cushion", "dropback_duration", "pass_rushers_encountered", "peak_pressure_probability_allowed",
    "pressure_allowed", "sack_allowed", "player_get_off", "sack", "time_to_pressure", "blitzing",
    "unblocked_pressure", "coverage_assignment", "coverage_assignment_at_snap",
]

PRIMARY_KEYS = {
    "combine": ["nfl_id"],
    "players": ["nfl_id"],
    "careers": ["nfl_id"],
    "games": ["game_id"],
    "tracking": ["game_id", "play_id", "nfl_id", "time"],
    "player_play": ["game_id", "play_id", "nfl_id"],
    "combine_tracking": ["event_id", "entity_type", "time"],
}
OPTIONAL_TABLES = ("player_play", "combine_tracking")
# Filas de datos de un CSV guardado en el límite de una hoja de cálculo (1,048,576 con el encabezado):
# el archivo probablemente se cortó.
SPREADSHEET_ROW_LIMIT = 1_048_575
_PART = re.compile(r"^(?P<stem>.+)_\d+$")
_CSV_EXT = re.compile(r"\.csv(\.gz)?$")


class SchemaError(ValueError):
    """Se lanza cuando una tabla original no cumple su contrato."""


def _stem(p: Path) -> str:
    """Raíz del nombre sin extensión ni compresión (`x_2.csv.gz` -> `x_2`)."""
    return _CSV_EXT.sub("", p.name)


def logical_name(p: Path) -> str:
    """Nombre lógico del archivo, sin la compresión (`x.csv.gz` -> `x.csv`)."""
    return _stem(p) + ".csv"


def _open_raw(p: Path, mode: str = "rb"):
    """Abre un CSV original, comprimido (.gz) o no."""
    kw = {"encoding": "utf-8"} if "t" in mode else {}
    return gzip.open(p, mode, **kw) if p.name.endswith(".gz") else open(p, mode, **kw)


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 del contenido (descomprimido) de un archivo."""
    h = hashlib.sha256()
    with _open_raw(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def resolve_files(raw_dir: Path, spec: str | list[str]) -> tuple[list[Path], list[Path]]:
    """Archivos de una tabla: `spec` es un nombre, un patrón glob o una lista de ellos.

    Devuelve (usados, reemplazados), ordenados por nombre. Una parte `<raíz>_<n>.csv` se
    reemplaza (no se lee ni se incluye en el hash) cuando también existe el archivo completo
    `<raíz>.csv`. `x.csv` y `x.csv.gz` son el mismo archivo lógico: se usa el primero encontrado.
    """
    specs = [spec] if isinstance(spec, str) else list(spec)
    found: dict[str, Path] = {}
    for s_ in specs:
        for p in sorted(Path(q) for q in glob.glob(str(Path(raw_dir) / s_))):
            if _CSV_EXT.search(p.name):
                found.setdefault(_stem(p), p)
    files = sorted(found.values(), key=_stem)
    names = set(found)
    used, superseded = [], []
    for p in files:
        m = _PART.match(_stem(p))
        (superseded if m and m.group("stem") in names else used).append(p)
    return used, superseded


def table_files(cfg: Config, table: str) -> list[Path]:
    """Archivos usados para `table` (lista vacía si una tabla opcional no está)."""
    spec = cfg["raw_files"].get(table)
    if spec is None:
        return []
    used, superseded = resolve_files(cfg.path("raw"), spec)
    for p in superseded:
        log.warning("%s: %s queda reemplazado por el archivo completo y no se usa", table, p.name)
    if not used and table not in OPTIONAL_TABLES and table != "tracking":
        raise SchemaError(f"tabla obligatoria {table}: ningún archivo coincide con {spec}")
    return used


def _read(path: Path, table: str, usecols: list[str] | None = None) -> pd.DataFrame:
    """Valida el encabezado completo y carga (un subconjunto de) las columnas con su tipo declarado.

    Una columna faltante es un error. Una columna extra se registra y se ignora (el reporte del
    contrato la lista), así un cambio aditivo del proveedor no detiene el flujo.
    """
    expected = SCHEMAS[table]
    header = list(pd.read_csv(path, nrows=0).columns)
    missing = set(expected) - set(header)
    extra = set(header) - set(expected)
    if missing:
        raise SchemaError(f"{path.name}: faltan columnas {sorted(missing)}")
    if extra:
        log.warning("%s: se ignoran columnas extra %s", path.name, sorted(extra))
    cols = usecols or list(expected)
    # "NA" y las cadenas vacías son los marcadores de nulo de los archivos originales.
    df = pd.read_csv(path, na_values=["NA", ""], keep_default_na=True, usecols=cols, low_memory=False)
    try:
        df = df.astype({c: expected[c] for c in cols})
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{path.name}: falló la conversión de tipos: {exc}") from exc
    dups = df.duplicated(PRIMARY_KEYS[table]).sum()
    if dups:
        raise SchemaError(f"{path.name}: {dups} filas duplicadas en la clave {PRIMARY_KEYS[table]}")
    return df


def _read_parts(paths: list[Path], table: str, usecols: list[str] | None = None) -> pd.DataFrame:
    """Lee y concatena las partes de una tabla; las partes no pueden repetir claves."""
    df = pd.concat([_read(p, table, usecols) for p in paths], ignore_index=True)
    if len(paths) > 1 and df.duplicated(PRIMARY_KEYS[table]).any():
        raise SchemaError(f"{table}: las partes del archivo se solapan en {PRIMARY_KEYS[table]}")
    return df


def iter_tracking(cfg: Config):
    """Entrega (nombre del archivo, cuadros validados) un archivo de tracking a la vez."""
    for p in table_files(cfg, "tracking"):
        yield p.name, _read(p, "tracking").assign(source_file=p.name)


def load_raw(cfg: Config, tracking: bool = True) -> dict[str, pd.DataFrame]:
    """Carga y valida todas las tablas originales.

    tracking=False omite los cuadros (el flujo los procesa archivo por archivo con `iter_tracking`);
    las claves del tracking se siguen comprobando contra jugadores y partidos, archivo por archivo.
    """
    tables = {name: _read_parts(table_files(cfg, name), name) for name in ("combine", "players", "careers", "games")}
    if tracking:
        parts = [df for _, df in iter_tracking(cfg)]
        tables["tracking"] = pd.concat(parts, ignore_index=True)
        if tables["tracking"].duplicated(PRIMARY_KEYS["tracking"]).any():
            raise SchemaError("los archivos de tracking se solapan en (partido, jugada, jugador, tiempo)")
    else:
        keys = [pd.read_csv(p, usecols=["game_id", "nfl_id"]).drop_duplicates() for p in table_files(cfg, "tracking")]
        tables["tracking_keys"] = pd.concat(keys, ignore_index=True).drop_duplicates() if keys else \
            pd.DataFrame({"game_id": pd.Series(dtype="int64"), "nfl_id": pd.Series(dtype="int64")})
    if pp := table_files(cfg, "player_play"):
        tables["player_play"] = _read_parts(pp, "player_play", PLAYER_PLAY_USECOLS)
    if ct := table_files(cfg, "combine_tracking"):
        tables["combine_tracking"] = _read_parts(ct, "combine_tracking")
    _check_referential_integrity(tables)
    return tables


def _check_referential_integrity(t: dict[str, pd.DataFrame]) -> None:
    """Cada jugador y partido referenciado debe existir en su tabla maestra."""
    ids = set(t["combine"].nfl_id)
    for name in ("players", "careers"):
        if set(t[name].nfl_id) != ids:
            raise SchemaError(f"{name}.nfl_id no coincide con combine.nfl_id")
    tr = t["tracking"] if "tracking" in t else t["tracking_keys"]
    orphan_players = set(tr.nfl_id) - ids
    if orphan_players:
        raise SchemaError(f"el tracking tiene {len(orphan_players)} jugadores sin fila en el Combine")
    orphan_games = set(tr.game_id) - set(t["games"].game_id)
    if orphan_games:
        raise SchemaError(f"el tracking tiene {len(orphan_games)} partidos que no están en games.csv")
    if "player_play" in t:
        pp = t["player_play"]
        if set(pp.nfl_id) - ids:
            raise SchemaError(f"player_play tiene {len(set(pp.nfl_id) - ids)} jugadores sin fila en el Combine")
        if set(pp.game_id) - set(t["games"].game_id):
            raise SchemaError("player_play tiene partidos que no están en games.csv")
    if "combine_tracking" in t and set(t["combine_tracking"].nfl_id.dropna()) - ids:
        raise SchemaError("combine_tracking tiene jugadores sin fila en el Combine")


MANIFEST_TABLES = ("combine", "players", "careers", "games", "tracking", "player_play", "combine_tracking")


def write_manifest(cfg: Config) -> dict:
    """Calcula el hash de cada archivo original y registra su tamaño; devuelve el manifiesto.

    `data_version` = SHA-256 de los hashes de contenido en este orden: tablas base, partes del
    tracking, player_play, combine_tracking. El hash es del contenido descomprimido, así que
    comprimir los CSV no cambia la versión de datos.
    """
    paths = [p for k in MANIFEST_TABLES for p in table_files(cfg, k)]
    entries = {}
    for p in paths:
        with _open_raw(p, "rb") as fh:
            n_bytes = 0
            while block := fh.read(1 << 20):
                n_bytes += len(block)
        # Líneas de texto menos el encabezado (con saltos de línea universales, como en versiones anteriores;
        # los textos entre comillas con saltos de línea cuentan más de una línea).
        with _open_raw(p, "rt") as fh:
            n_lines = sum(1 for _ in fh)
        entries[logical_name(p)] = {"sha256": sha256(p), "bytes": n_bytes, "rows": n_lines - 1,
                                    "stored_as": p.name, "stored_bytes": p.stat().st_size}
    combined = hashlib.sha256("".join(e["sha256"] for e in entries.values()).encode()).hexdigest()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_version": combined[:12],
        "files": entries,
    }
    with open(cfg.path("manifest"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def verify_manifest(cfg: Config) -> list[str]:
    """Compara el hash de cada archivo original con el manifiesto; devuelve los que no coinciden."""
    m = json.loads(cfg.path("manifest").read_text(encoding="utf-8"))
    raw = cfg.path("raw")
    bad = []
    for name, e in m["files"].items():
        p = raw / e.get("stored_as", name)
        if not p.exists() or sha256(p) != e["sha256"]:
            bad.append(name)
    return bad


# =============================================================================
# Contrato de datos (especificación del proveedor)
# =============================================================================
# El contrato (configuracion/contrato_datos.yaml) es una copia legible por máquina de
# documentos/Data_structure_CSV_files.docx. Cada ejecución valida los archivos originales y escribe:
#   reportes/metricas/data_contract_report.csv    una fila por comprobación (tabla, columna, comprobación,
#                                                 severidad, resultado, tratamiento)
#   reportes/metricas/data_contract_summary.json  conteos, archivos faltantes, cobertura frente a la especificación
#
# Severidad: un archivo o columna obligatoria faltante, una clave duplicada o una clave foránea rota es
# un `error` (la ejecución se detiene si `contract.fail_on_error` es true). Las desviaciones de tipo, nulos,
# rango, valores permitidos, patrón y número de filas son `warning`: las reglas de negocio (R1-R16)
# deciden cómo se tratan esas filas y el reporte documenta cuántas son.
TRUE_FALSE = {"TRUE", "FALSE", "T", "F", "1", "0", "1.0", "0.0"}


def load_contract(path: str | Path | None = None) -> dict:
    """Lee el contrato de datos (por defecto configuracion/contrato_datos.yaml)."""
    path = _project_path(path) if path else PROJECT_ROOT / "configuracion" / "contrato_datos.yaml"
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def discover(cfg: Config, table: str, spec: dict) -> tuple[list[Path], list[Path]]:
    """(usados, reemplazados) de una tabla del contrato: la entrada `raw_files` del flujo cuando existe
    (así el validador comprueba exactamente lo que lee el flujo); si no, los patrones del contrato."""
    return resolve_files(cfg.path("raw"), cfg["raw_files"].get(table, spec["patterns"]))


def _row(table, file, column, check, severity, ok, n_bad=0, n=0, detail=""):
    return {"table": table, "file": file, "column": column, "check": check, "severity": severity,
            "result": "pass" if ok else "fail", "n_bad": int(n_bad), "share_bad": (n_bad / n) if n else 0.0,
            "detail": detail}


def _type_ok(s: pd.Series, kind: str) -> tuple[bool, str]:
    """¿La columna tiene el tipo que declara el contrato? Devuelve (cumple, detalle)."""
    v = s.dropna()
    if v.empty:
        return True, "todo nulo"
    if kind == "int":
        if pd.api.types.is_integer_dtype(v):
            return True, str(s.dtype)
        if pd.api.types.is_float_dtype(v):
            ok = bool(np.all(np.isclose(v, np.round(v))))
            return ok, "columna float con valores enteros (entero con nulos)" if ok else "valores no enteros"
        return False, str(s.dtype)
    if kind == "float":
        return bool(pd.api.types.is_numeric_dtype(v) and not pd.api.types.is_bool_dtype(v)), str(s.dtype)
    if kind == "bool":
        if pd.api.types.is_bool_dtype(v):
            return True, "bool"
        vals = set(v.astype(str).str.upper().str.strip())
        return vals <= TRUE_FALSE, f"códigos {sorted(vals)[:6]}"
    if kind == "datetime":
        parsed = pd.to_datetime(v.head(5000), errors="coerce", format="ISO8601")
        return bool(parsed.notna().all()), "ISO 8601"
    return bool(not pd.api.types.is_numeric_dtype(v) or pd.api.types.is_object_dtype(v)), str(s.dtype)


def validate_frame(table: str, file: str, spec: dict, df: pd.DataFrame) -> list[dict]:
    """Comprobaciones por columna de un archivo (o parte) contra su entrada del contrato."""
    rows = []
    cols = spec["columns"]
    n = len(df)
    missing = [c for c in cols if c not in df.columns]
    extra = [c for c in df.columns if c not in cols]
    rows.append(_row(table, file, "*", "columns_present", "error", not missing, len(missing), len(cols),
                     f"faltan {missing}" if missing else f"{len(cols)} columnas según la especificación"))
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
                             f"la especificación dice {c['spec_type']} 1.0/0.0; el archivo guarda texto TRUE/FALSE (se lee como booleano)"))
        nulls = int(s.isna().sum())
        if not c.get("nullable", False):
            rows.append(_row(table, file, col, "not_null", "warning", nulls == 0, nulls, n, f"{nulls} nulos"))
        v = s.dropna()
        if "range" in c and len(v) and pd.api.types.is_numeric_dtype(v):
            lo, hi = c["range"]
            bad = int(((v < lo) | (v > hi)).sum())
            rows.append(_row(table, file, col, f"range[{lo},{hi}]", "warning", bad == 0, bad, n,
                             f"observado [{v.min():.4g}, {v.max():.4g}]"))
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
                             f"inesperados {bad_vals}" if bad else f"dentro de {allowed}"))
        if "pattern" in c and len(v):
            rx = re.compile(c["pattern"])
            bad = int((~v.astype(str).map(lambda x: bool(rx.match(x)))).sum())
            rows.append(_row(table, file, col, "pattern", "warning", bad == 0, bad, n, c["pattern"]))
    key = spec.get("key")
    if key and all(k in df.columns for k in key):
        dups = int(df.duplicated(key).sum())
        rows.append(_row(table, file, ",".join(key), "unique_key", "error", dups == 0, dups, n, f"{dups} claves duplicadas"))
    return rows


def _read_any(path: Path) -> pd.DataFrame:
    """Lectura sin tipos impuestos, para que el contrato vea los valores tal como llegan."""
    return pd.read_csv(path, na_values=["NA", ""], keep_default_na=True, low_memory=False)


def validate_all(cfg: Config, contract: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Valida cada tabla del contrato que esté presente; devuelve (reporte, resumen)."""
    contract = contract or load_contract()
    rows, frames, summary = [], {}, {"files": {}, "superseded_files": {}, "missing_files": [], "coverage": {}}
    order = ["players", "combine", "careers", "games", "player_play", "tracking", "combine_tracking"]
    for table in order:
        spec = contract["files"][table]
        paths, superseded = discover(cfg, table, spec)
        if superseded:
            summary["superseded_files"][table] = [p.name for p in superseded]
            rows.append(_row(table, "+".join(p.name for p in superseded), "*", "superseded_part", "warning", False,
                             len(superseded), len(superseded) + len(paths), "parte ignorada: existe el archivo completo"))
        if not paths:
            sev = "error" if spec.get("required") else "warning"
            rows.append(_row(table, spec["spec_name"], "*", "file_present", sev, False, 1, 1,
                             f"ningún archivo coincide con {spec['patterns']}"))
            summary["missing_files"].append({"table": table, "spec_name": spec["spec_name"], "required": spec.get("required", False)})
            continue
        summary["files"][table] = [p.name for p in paths]
        parts = []
        for p in paths:
            df = _read_any(p)
            rows += validate_frame(table, p.name, spec, df)
            if len(df) == SPREADSHEET_ROW_LIMIT:
                rows.append(_row(table, p.name, "*", "spreadsheet_row_limit", "warning", False, 1, 1,
                                 f"exactamente {SPREADSHEET_ROW_LIMIT:,} filas de datos: el archivo probablemente se cortó en una hoja de cálculo"))
            keep = [c for c in set(sum([list(map(str.strip, k.split(","))) for k in spec.get("soft_foreign_keys", {})], []))
                    | set(spec.get("key", [])) | set(spec.get("foreign_keys", {})) | {"nfl_id", "game_id", "play_id", "event_id", "season", "season_type"}
                    if c in df.columns]
            parts.append(df[keep].copy())
            del df
        tab = pd.concat(parts, ignore_index=True)
        frames[table] = tab
        n = len(tab)
        files = "+".join(p.name for p in paths)
        if len(paths) > 1 and spec.get("key") and all(k in tab.columns for k in spec["key"]):
            dups = int(tab.duplicated(spec["key"]).sum())
            rows.append(_row(table, files, ",".join(spec["key"]), "unique_key_across_parts",
                             "error", dups == 0, dups, n, f"{dups} claves duplicadas entre partes"))
        if "expected_rows" in spec:
            exp = spec["expected_rows"]
            rows.append(_row(table, files, "*", "row_count", "warning", n == exp, abs(n - exp), exp,
                             f"{n:,} filas frente a {exp:,} en la especificación ({(n - exp) / exp:+.2%})"))
        if "expected_attempts" in spec and "event_id" in tab:
            got = tab["event_id"].nunique()
            rows.append(_row(table, files, "event_id", "attempt_count", "warning",
                             got == spec["expected_attempts"], abs(got - spec["expected_attempts"]), spec["expected_attempts"],
                             f"{got:,} intentos frente a {spec['expected_attempts']:,} en la especificación"))
        # claves foráneas
        for col, ref in spec.get("foreign_keys", {}).items():
            rt, rc = ref.split(".")
            if rt in frames and col in tab and rc in frames[rt]:
                vals = tab[col].dropna()
                bad = int((~vals.isin(frames[rt][rc])).sum())
                rows.append(_row(table, files, col, f"foreign_key->{ref}", "error", bad == 0, bad, len(vals),
                                 f"{bad} filas referencian {ref} inexistentes"))
        for cols, ref in spec.get("soft_foreign_keys", {}).items():
            kc = [c.strip() for c in cols.split(",")]
            if ref in frames and all(c in tab for c in kc) and all(c in frames[ref] for c in kc):
                left = tab[kc].drop_duplicates()
                m = left.merge(frames[ref][kc].drop_duplicates(), on=kc, how="left", indicator=True)
                bad = int((m["_merge"] == "left_only").sum())
                rows.append(_row(table, files, cols, f"soft_foreign_key->{ref}", "warning", bad == 0, bad, len(left),
                                 f"{bad} de {len(left):,} combinaciones {cols} no están en {ref}"))
        # cobertura frente a la especificación (tracking: temporadas y jugadores)
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
            rows.append(_row(table, files, "season", "season_coverage", "warning", not miss_s,
                             len(miss_s), len(exp_s), f"temporadas {seasons}; faltan {miss_s}; {games} partidos {types}"))
            rows.append(_row(table, files, "nfl_id", "player_coverage", "warning",
                             players >= spec.get("expected_players", 0), spec.get("expected_players", 0) - players,
                             spec.get("expected_players", 0), f"{players} de {spec.get('expected_players')} prospectos con tracking"))
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
    """Escribe el reporte y el resumen del contrato en reportes/metricas."""
    m = cfg.path("metrics")
    report.to_csv(m / "data_contract_report.csv", index=False)
    with open(m / "data_contract_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)


# =============================================================================
# Limpieza de prospectos: reglas de negocio R1-R7
# =============================================================================
# Reglas aplicadas (todos los umbrales están en configuracion/config.yaml):
#   R1  Valores de pruebas fuera de rangos físicamente plausibles -> NaN (error de medición).
#   R2  Las pruebas faltantes NO se imputan aquí: se conserva una bandera `<prueba>_missing` y la
#       imputación (mediana del grupo de posición) ocurre dentro de cada pliegue de entrenamiento.
#   R3  Los jugadores no seleccionados reciben draft_overall_pick = 260 e is_undrafted = 1.
#   R4  Temporadas disponibles = última temporada completa - año de draft + 1, para comparar
#       carreras de distinta duración por temporada.
#   R5  start_rate = partidos como titular / (17 x temporadas disponibles); titular = start_rate >= 0.5.
#   R6  La edad al draft se calcula respecto de una fecha aproximada del draft (27 de abril).
#   R7  La conferencia se reduce a una bandera binaria de conferencia fuerte (niveles raros agrupados).
DRILLS = ["ten_yd_split", "forty", "vertical", "broad_jump", "three_cone", "short_shuttle", "bench_reps"]
MEASURABLES = ["combine_height", "combine_weight", "hand_size", "arm_length", "wing_span"]
NGS_SCORES = ["ngs_athleticism_score", "ngs_college_production_score", "ngs_final_score"]
# Pruebas en las que un valor MENOR es mejor (tiempos); sirve para orientar las correlaciones.
LOWER_IS_BETTER = {"ten_yd_split", "forty", "three_cone", "short_shuttle"}
POSITION_GROUPS = ["DB", "DL", "OL", "TE", "WR"]


def apply_range_rules(df: pd.DataFrame, ranges: dict[str, list[float]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """R1: borra valores imposibles. Devuelve (tabla limpia, bitácora de cambios)."""
    df = df.copy()
    changes = []
    for col, (lo, hi) in ranges.items():
        bad = df[col].notna() & ~df[col].between(lo, hi)
        for idx in df.index[bad]:
            changes.append({"nfl_id": df.at[idx, "nfl_id"], "column": col, "value": df.at[idx, col], "rule": f"fuera de [{lo}, {hi}]"})
        df.loc[bad, col] = np.nan
    return df, pd.DataFrame(changes, columns=["nfl_id", "column", "value", "rule"])


def build_prospects(tables: dict[str, pd.DataFrame], cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Une las tablas de Combine, jugadores y carreras en una sola tabla limpia de prospectos."""
    br = cfg["business_rules"]
    combine, audit = apply_range_rules(tables["combine"], br["drill_valid_ranges"])
    players = tables["players"].copy()
    careers = tables["careers"].copy()

    df = combine.merge(players, on=["nfl_id", "draft_year"], how="inner", validate="1:1")
    df = df.merge(careers, on="nfl_id", how="inner", validate="1:1")
    if len(df) != len(combine):
        raise ValueError("draft_year no coincide entre las tablas de Combine y de jugadores")

    # Tipos
    df["combine_position"] = pd.Categorical(df["combine_position"].astype(str), categories=POSITION_GROUPS)
    df["nfl_position"] = df["nfl_position"].astype("category")
    df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce")

    # R2: banderas de faltante (informativas: los prospectos eligen qué pruebas hacer)
    for col in DRILLS:
        df[f"{col}_missing"] = df[col].isna().astype(int)
    df["n_drills_run"] = df[DRILLS].notna().sum(axis=1)

    # R3: capital de draft
    df["is_undrafted"] = df["draft_overall_pick"].isna().astype(int)
    df["draft_pick_filled"] = df["draft_overall_pick"].astype("float").fillna(br["undrafted_pick"])
    df["log_draft_pick"] = np.log(df["draft_pick_filled"])
    df["draft_round_filled"] = df["draft_round"].astype("float").fillna(8)  # 8 = no seleccionado

    # R6: edad al draft
    draft_date = pd.to_datetime(df["draft_year"].astype(str) + "-" + br["draft_reference_month_day"])
    df["age_at_draft"] = (draft_date - df["birth_date"]).dt.days / 365.25

    # R7: conferencia
    df["power_conference"] = df["college_conference"].isin(br["power_conferences"]).astype(int)

    # Variables atléticas derivadas (documentadas, estándar en analítica de scouting)
    # Speed Score (Barnwell): tiempo de 40 yardas ajustado por peso.
    df["speed_score"] = df["combine_weight"] * 200 / df["forty"] ** 4
    # Proporción de aceleración: fracción del 40 que se va en las primeras 10 yardas (menor = salida más rápida).
    df["ten_split_share"] = df["ten_yd_split"] / df["forty"]
    # Momento en la salida: peso desplazado por segundo en las primeras 10 yardas (lb/s).
    df["ten_split_momentum"] = df["combine_weight"] / df["ten_yd_split"]

    # R4 + R5: resultado de carrera normalizado por oportunidad
    df["seasons_available"] = br["last_completed_season"] - df["draft_year"] + 1
    possible = br["games_per_season"] * df["seasons_available"]
    df["start_rate"] = (df["career_games_started"] / possible).clip(0, 1)
    df["is_starter"] = (df["start_rate"] >= br["starter_threshold"]).astype(int)
    df["snaps_per_season"] = (df["career_offensive_snaps"] + df["career_defensive_snaps"]) / df["seasons_available"]

    return df.sort_values("nfl_id").reset_index(drop=True), audit


def missingness_by_position(df: pd.DataFrame) -> pd.DataFrame:
    """Proporción de prospectos sin cada prueba, por grupo de posición del Combine."""
    return df.groupby("combine_position", observed=True)[DRILLS].apply(lambda g: g.isna().mean()).round(3)


# =============================================================================
# Movimiento en partido: tracking a 10 Hz (reglas R8-R9)
# =============================================================================
# Cada métrica se ancla al snap, así es comparable entre jugadas e independiente de la dirección
# de la jugada (no hace falta normalizar izquierda/derecha: todo son velocidades, distancias o
# cambios de ángulo).
#
# Métricas por jugador-jugada
#   burst_1s / burst_2s  distancia en línea recta desde el punto del snap en t = 1.0 / 2.0 s (yd).
#                        burst_2s es el análogo en partido del parcial de 10 yardas del Combine, que
#                        los prospectos recorren en ~1.5-1.9 s.
#   get_off_1s           velocidad en t = 1.0 s tras el snap (yd/s).
#   peak_accel_2s        aceleración máxima en los primeros 2.0 s (yd/s^2).
#   top_speed            velocidad máxima tras el snap (yd/s); análogo de las 40 yardas.
#   max_decel            mayor tasa de frenado, -d(velocidad)/dt sobre la velocidad suavizada en 3
#                        cuadros mientras se mueve a >= 1 yd/s (yd/s^2); análogo del giro del shuttle / 3 conos.
#   max_cut_deg          mayor cambio de dirección de desplazamiento en 0.5 s, moviéndose a >= 2 yd/s
#                        en ambos extremos (grados).
#
# Nivel jugador
#   nivel      = media de las jugadas del jugador. La media es insesgada sea cual sea el número de
#                jugadas; primero se probó una "capacidad" de percentil 80 (v0.3.0) y se descartó: lee
#                ~0.4 yd de menos con 3 jugadas y ~0.2 yd con 5, y como los titulares suman más
#                jugadas ese sesgo filtraba la titularidad a la métrica. El percentil 80 se conserva
#                como `<métrica>_q80` solo para el análisis de sensibilidad.
#   contracción = Bayes empírico hacia la media del grupo de posición, peso n/(n+k), para que un
#                jugador visto en 3 jugadas no se ordene con la misma confianza que uno visto en 30.
#   Snap Burst Index (SBI) = 50 + 10 * z del burst_2s contraído dentro del grupo de posición.
PLAY_KEY = ["game_id", "play_id", "nfl_id"]
METRICS = ["burst_1s", "burst_2s", "get_off_1s", "peak_accel_2s", "top_speed", "max_decel", "max_cut_deg"]


def clean_tracking(tr: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """Tipa, ordena y revisa los cuadros de tracking. Devuelve (cuadros, conteos de auditoría)."""
    rules = cfg["tracking_rules"]
    df = tr.copy()
    df["time"] = pd.to_datetime(df["time"], format="ISO8601")
    df = df.sort_values(PLAY_KEY + ["time"]).reset_index(drop=True)
    audit: dict[str, int] = {"frames_in": len(df)}

    # El espaciado entre cuadros debe ser 1 / frecuencia; las jugadas con huecos se descartan (no se esperan).
    step = 1.0 / rules["frame_rate_hz"]
    gaps = df.groupby(PLAY_KEY)["time"].diff().dt.total_seconds()
    bad_plays = df.loc[gaps.notna() & ~np.isclose(gaps, step), PLAY_KEY].drop_duplicates()
    audit["plays_with_frame_gaps"] = len(bad_plays)
    if len(bad_plays):
        df = df.merge(bad_plays.assign(_bad=1), on=PLAY_KEY, how="left")
        df = df[df["_bad"].isna()].drop(columns="_bad")

    # Velocidad / aceleración físicamente imposibles -> NaN y luego interpolación dentro de la jugada.
    for col, cap in (("s", rules["speed_cap"]), ("a", rules["accel_cap"])):
        bad = df[col] > cap
        audit[f"frames_{col}_above_cap"] = int(bad.sum())
        df.loc[bad, col] = np.nan
        df[col] = df.groupby(PLAY_KEY)[col].transform(lambda s: s.interpolate(limit_direction="both"))

    # Posiciones muy fuera del campo se cuentan (se conservan: los jugadores sí salen del campo).
    audit["frames_out_of_field"] = int(((df.x < -10) | (df.x > 130) | (df.y < -10) | (df.y > 63.3)).sum())
    audit["frames_out"] = len(df)
    return df.reset_index(drop=True), audit


def _heading_change(dir_deg: np.ndarray, lag: int) -> np.ndarray:
    """Cambio absoluto (envuelto) en la dirección de desplazamiento a lo largo de `lag` cuadros (grados)."""
    h = np.deg2rad(dir_deg)
    return np.abs(np.rad2deg(np.angle(np.exp(1j * (h[lag:] - h[:-lag])))))


def play_metrics(frames: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """Una fila por jugador-jugada con las métricas de movimiento ancladas al snap."""
    rules = cfg["tracking_rules"]
    hz = rules["frame_rate_hz"]
    i1, i2 = int(rules["get_off_horizon_s"] * hz), int(rules["burst_horizon_s"] * hz)
    lag = int(rules["cut_window_s"] * hz)
    snap_events = set(rules["snap_events"])
    out, skipped_no_snap, skipped_short = [], 0, 0

    for key, d in frames.groupby(PLAY_KEY, sort=False):
        ev = d["event"].to_numpy(dtype=object)
        snap_idx = next((i for i, e in enumerate(ev) if e in snap_events), None)
        if snap_idx is None:
            skipped_no_snap += 1
            continue
        post = d.iloc[snap_idx:]
        if len(post) < rules["min_post_snap_frames"] + 1:
            skipped_short += 1
            continue
        x, y = post["x"].to_numpy(), post["y"].to_numpy()
        s, a = post["s"].to_numpy(), post["a"].to_numpy()
        disp = np.hypot(x - x[0], y - y[0])

        s_smooth = pd.Series(s).rolling(3, center=True, min_periods=1).mean().to_numpy()
        dsdt = np.gradient(s_smooth, 1.0 / hz)
        moving = s_smooth >= 1.0
        max_decel = float(-dsdt[moving].min()) if moving.any() else np.nan

        dh = _heading_change(post["dir"].to_numpy(), lag)
        fast = (s[lag:] >= rules["cut_min_speed"]) & (s[:-lag] >= rules["cut_min_speed"])
        max_cut = float(dh[fast].max()) if fast.any() else np.nan

        out.append({
            "game_id": key[0], "play_id": key[1], "nfl_id": key[2],
            "snap_event": ev[snap_idx], "post_snap_frames": len(post),
            "burst_1s": disp[i1], "burst_2s": disp[i2], "get_off_1s": s[i1],
            "peak_accel_2s": a[: i2 + 1].max(), "top_speed": s.max(),
            "max_decel": max_decel, "max_cut_deg": max_cut,
        })

    audit = {"plays_no_snap": skipped_no_snap, "plays_too_short": skipped_short, "plays_kept": len(out)}
    return pd.DataFrame(out), audit


def _shrink(group: pd.DataFrame, col: str) -> pd.Series:
    """Contracción por Bayes empírico de las medias de cada jugador hacia la media del grupo de posición.

    k = sigma2_dentro / tau2_entre, peso = n / (n + k). sigma2_dentro es la varianza por jugada
    agrupada dentro de los jugadores; tau2 es la varianza entre medias de jugadores neta del ruido de
    muestreo (método de momentos, con piso en el 5% de la varianza observada para que ningún grupo
    se contraiga a un único punto).
    """
    mean = group[f"{col}_mean"]
    w = shrink_weight(group, col)
    grand = np.average(mean, weights=group["n_plays"])
    return grand + w * (mean - grand)


def shrink_weight(group: pd.DataFrame, col: str) -> pd.Series:
    """Peso de Bayes empírico n / (n + k) sobre la media propia del jugador (= su fiabilidad)."""
    n = group["n_plays"]
    s2w = group[f"{col}_var"].fillna(group[f"{col}_var"].median())
    sigma2 = np.average(s2w, weights=n - 1)
    total = group[f"{col}_mean"].var(ddof=1)
    tau2 = max(total - np.mean(sigma2 / n), 0.05 * total)
    return n / (n + sigma2 / tau2)


def player_profiles(plays: pd.DataFrame, prospects: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Agrega las métricas por jugada a jugadores, aplica la contracción y calcula el Snap Burst Index."""
    rules = cfg["tracking_rules"]
    q = rules["capacity_quantile"]
    g = plays.groupby("nfl_id")
    prof = g[METRICS].mean().add_suffix("_mean")
    prof = prof.join(g[METRICS].quantile(q).add_suffix("_q80"))  # solo para el análisis de sensibilidad
    prof = prof.join(g[METRICS].var(ddof=1).add_suffix("_var"))
    prof = prof.join(g.size().rename("n_plays"))
    prof = prof.join(g["game_id"].nunique().rename("n_games"))
    prof = prof.reset_index().merge(prospects[["nfl_id", "combine_position"]], on="nfl_id", how="left")
    prof = prof[prof["n_plays"] >= rules["min_plays_per_player"]].copy()

    for col in METRICS:
        prof[f"{col}_eb"] = prof.groupby("combine_position", observed=True, group_keys=False).apply(
            lambda grp, c=col: _shrink(grp, c)
        )
    # Fiabilidad del SBI de cada jugador (proporción de su propia media que conserva tras la contracción).
    prof["sbi_reliability"] = prof.groupby("combine_position", observed=True, group_keys=False).apply(
        lambda grp: shrink_weight(grp, "burst_2s")
    )
    # Snap Burst Index: normalizado por posición, en escala 50 +/- 10 como otras notas de scouting.
    grp = prof.groupby("combine_position", observed=True)["burst_2s_eb"]
    z = (prof["burst_2s_eb"] - grp.transform("mean")) / grp.transform("std")
    prof["sbi"] = rules["sbi_center"] + rules["sbi_scale"] * z
    # La misma normalización para las demás métricas contraídas (variables de los clasificadores).
    for col in METRICS:
        gcol = prof.groupby("combine_position", observed=True)[f"{col}_eb"]
        prof[f"{col}_z"] = (prof[f"{col}_eb"] - gcol.transform("mean")) / gcol.transform("std")
    return prof.drop(columns=[c for c in prof.columns if c.endswith("_var")]).reset_index(drop=True)


def metric_reliability(plays: pd.DataFrame, profiles: pd.DataFrame, group: str = "combine_position") -> pd.DataFrame:
    """¿Es una métrica de movimiento un rasgo estable del jugador? Una fila por (grupo de posición, métrica).

    icc            proporción de la varianza entre medias de jugadores neta del ruido de muestreo
                   (método de momentos; <= 0 significa que no se detectan diferencias entre jugadores)
    mean_weight    peso medio de Bayes empírico n/(n+k) = fiabilidad de la estimación de un jugador
    anova_p        valor p del ANOVA de una vía para "todos los jugadores del grupo se mueven igual"
    """
    keep = plays[plays.nfl_id.isin(profiles.nfl_id)]
    rows = []
    for pos, g in keep.groupby(group, observed=True):
        for m in METRICS:
            gg = g.groupby("nfl_id")[m]
            n, mu, var = gg.size(), gg.mean(), gg.var(ddof=1)
            sigma2 = np.average(var.fillna(var.median()), weights=n - 1)
            total = mu.var(ddof=1)
            tau2 = total - np.mean(sigma2 / n)
            groups = [v.dropna().to_numpy() for _, v in gg]
            groups = [v for v in groups if len(v) > 0]
            f, p = stats.f_oneway(*groups) if len(groups) > 1 else (np.nan, np.nan)
            w = n / (n + sigma2 / tau2) if tau2 > 0 else pd.Series(0.0, index=n.index)
            rows.append({group: pos, "metric": m, "players": len(n), "plays": int(n.sum()),
                         "icc": tau2 / total if total > 0 else np.nan, "mean_weight": float(w.mean()),
                         "anova_f": float(f), "anova_p": float(p)})
    return pd.DataFrame(rows)


def estimator_bias(plays: pd.DataFrame, metric: str, ns: list[int], n_sims: int, seed: int, min_plays: int = 20) -> pd.DataFrame:
    """Sesgo de la media / mediana / percentil 80 cuando un jugador se ve en solo n jugadas.

    Usa como verdad a los jugadores con >= min_plays jugadas y toma submuestras de n de ellas.
    """
    rng = np.random.default_rng(seed)
    big = plays.groupby("nfl_id").filter(lambda g: len(g) >= min_plays)
    est = {"mean": np.mean, "median": np.median, "q80": lambda v: np.quantile(v, 0.8)}
    rows = []
    for _, g in big.groupby("nfl_id"):
        v = g[metric].to_numpy()
        for name, fn in est.items():
            truth = fn(v)
            for n in ns:
                sims = [fn(rng.choice(v, n, replace=False)) for _ in range(n_sims)]
                rows.append({"estimator": name, "n_plays": n, "bias": float(np.mean(sims) - truth)})
    out = pd.DataFrame(rows).groupby(["estimator", "n_plays"], as_index=False)["bias"].mean()
    out["players_used"] = big.nfl_id.nunique()
    return out


def speed_curves(frames: pd.DataFrame, plays: pd.DataFrame, nfl_ids: list[int], horizon_s: float = 4.0) -> pd.DataFrame:
    """Trazas de velocidad tras el snap (t, velocidad) de los jugadores indicados, para graficar."""
    hz = 10
    sub = frames[frames.nfl_id.isin(nfl_ids)]
    rows = []
    for key, d in sub.groupby(PLAY_KEY, sort=False):
        ev = d["event"].to_numpy(dtype=object)
        idx = next((i for i, e in enumerate(ev) if e in {"ball_snap", "snap_direct"}), None)
        if idx is None:
            continue
        post = d.iloc[idx: idx + int(horizon_s * hz) + 1]
        rows.append(pd.DataFrame({"nfl_id": key[2], "game_id": key[0], "play_id": key[1],
                                  "t": np.arange(len(post)) / hz, "speed": post["s"].to_numpy()}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# =============================================================================
# Rendimiento en el campo: player_play (reglas R10-R15)
# =============================================================================
# Cada métrica se mide "sobre lo esperado": el resultado de la jugada menos el resultado medio de
# todas las jugadas de prospectos en el mismo contexto, para no premiar a un jugador por su rol o
# por sus rutas. Positivo siempre es mejor.
#
# Familia  Jugadas                            Métrica (por jugada)                                Contexto de lo "esperado"
# DL       snaps de presión al pasador (no    goe  = arranque esperado - arranque (s; + = más rápido) alineación (EDGE / INTERIOR_LINE / otra)
#          retrocede a cobertura)             proe = presión - presión esperada                   alineación
# OL       snaps de protección de pase        ppoe = esperado - probabilidad máxima de presión    alineación (T / G / C) x quintil de dropback
#                                             paoe = esperado - presión permitida (0/1)           igual
# WR/TE    rutas con lectura de separación    soe  = separación - separación esperada (yd)        ruta x alineación (solo ruta si < 30 jugadas)
#
# Nivel jugador: media de jugadas válidas, contracción por Bayes empírico hacia la media del grupo de
# posición del Combine (`<métrica>_eb`), su peso de fiabilidad (`<métrica>_rel`) y un z dentro de la
# familia (`<métrica>_z`).
#
# Reglas de negocio (configuracion/config.yaml -> performance_rules)
#   R10 solo temporada regular; R11 se quitan las jugadas anuladas por castigo;
#   R12 el arranque NGS fuera de [0.30, 2.00) s se trata como faltante;
#   R13 mínimo de jugadas válidas por familia (DL 20 snaps de presión, OL 50 de protección, WR/TE 30 rutas);
#   R14 los DB no reciben métrica en el campo: el archivo trae su alineación y asignación, pero no el
#       resultado de la cobertura;
#   R15 se quitan los downs fuera del 1-4 de la especificación (down 0: conversiones de dos puntos).
# Las banderas booleanas se leen tanto si el archivo trae texto TRUE/FALSE (como se entregó) como
# flotantes 1.0/0.0 (como documenta la especificación); las celdas de alineación salen de
# `performance_rules.alignment_codes`, que también mapea los códigos DE / DT / NT de la especificación.
ONFIELD_METRICS = ["goe", "proe", "ppoe", "paoe", "soe"]
METRIC_FAMILY = {"goe": "DL", "proe": "DL", "ppoe": "OL", "paoe": "OL", "soe": "WRTE"}
MIN_PLAYS_KEY = {"DL": "dl_rush", "OL": "ol_pass_block", "WRTE": "wrte_route"}
BOOL_COLS = ["target", "pressure_allowed", "blitzing", "unblocked_pressure"]
KEY = ["game_id", "play_id", "nfl_id"]

_BOOL_CODES = {"TRUE": 1.0, "T": 1.0, "1": 1.0, "1.0": 1.0, "FALSE": 0.0, "F": 0.0, "0": 0.0, "0.0": 0.0}


def _parse_bool(s: pd.Series) -> pd.Series:
    """TRUE/FALSE, True/False, 1/0 y 1.0/0.0 -> 1.0/0.0; cualquier otra cosa (incluido NA) -> NaN."""
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
        v = pd.to_numeric(s, errors="coerce").astype("float64")
        return v.where(v.isin([0.0, 1.0]))
    return s.astype("string").str.strip().str.upper().map(_BOOL_CODES).astype("float64")


def _alignment(codes: pd.Series, mapping: dict[str, str]) -> pd.Series:
    """lined_up_position -> celda de alineación del modelo de lo "esperado"; códigos sin mapeo -> OTHER."""
    return codes.map(mapping).fillna("OTHER").astype(codes.dtype)


def clean_player_play(pp: pd.DataFrame, games: pd.DataFrame, prospects: pd.DataFrame,
                      cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Devuelve (jugadas de temporada regular, contexto de jugada de todo tipo de temporada, auditoría)."""
    rules = cfg["performance_rules"]
    df = pp.merge(games[["game_id", "season", "season_type", "week"]], on="game_id", how="left", validate="m:1")
    df = df.merge(prospects[["nfl_id", "combine_position", "nfl_position", "draft_year"]], on="nfl_id",
                  how="left", validate="m:1")
    for c in BOOL_COLS:
        df[c] = _parse_bool(df[c])
    df["is_pass"] = df["pass_result"].notna().astype(int)
    df["combine_position"] = df["combine_position"].astype(str)
    df["family"] = df["combine_position"].map(rules["families"])
    df["rookie_season"] = (df["season"] == df["draft_year"]).astype(int)
    audit = {"pp_rows_in": len(df), "pp_players": int(df.nfl_id.nunique())}

    # Tabla de contexto para unir con el tracking (el tracking también cubre los playoffs).
    context = df[KEY + ["season_type", "lined_up_position", "route_ran", "is_pass",
                        "coverage_assignment_at_snap", "player_get_off"]].copy()

    keep = df["season_type"].isin(rules["season_types"])  # R10
    audit["pp_rows_not_regular_season"] = int((~keep).sum())
    if rules["exclude_nullified"]:  # R11
        nullified = df["play_nullified_by_penalty"].eq("Y")
        audit["pp_rows_nullified"] = int((keep & nullified).sum())
        keep &= ~nullified
    valid_downs = rules.get("valid_downs")  # R15
    if valid_downs:
        bad_down = ~df["down"].isin(valid_downs)
        audit["pp_rows_invalid_down"] = int((keep & bad_down).sum())
        keep &= ~bad_down
    lo, hi = rules["get_off_valid_s"]  # R12
    bad_go = df["player_get_off"].notna() & ~df["player_get_off"].between(lo, hi, inclusive="left")
    audit["pp_get_off_out_of_range"] = int((keep & bad_go).sum())
    df.loc[bad_go, "player_get_off"] = np.nan
    out = df[keep].reset_index(drop=True)
    audit["pp_rows_regular_season"] = len(out)
    return out, context, audit


def _expected(df: pd.DataFrame, value: str, keys: list[str], fallback: list[str] | None = None,
              min_n: int = 1) -> pd.Series:
    """Media de `value` dentro del contexto `keys`; las celdas con < min_n jugadas usan `fallback`."""
    grp = df.groupby(keys, dropna=False)[value]
    exp = grp.transform("mean")
    if fallback is not None:
        small = grp.transform("count") < min_n
        exp = exp.where(~small, df.groupby(fallback, dropna=False)[value].transform("mean"))
    return exp


def family_plays(plays: pd.DataFrame, cfg: Config) -> dict[str, pd.DataFrame]:
    """Tablas de jugadas por familia con las métricas sobre lo esperado."""
    rules = cfg["performance_rules"]
    codes = rules.get("alignment_codes", {
        "DL": {"EDGE": "EDGE", "INTERIOR_LINE": "INTERIOR_LINE"},
        "OL": {"T": "T", "G": "G", "C": "C"},
        "WRTE": {"WR": "WR", "SLOT_WR": "SLOT_WR", "TE": "TE"}})
    out = {}

    # DL, snaps de presión: jugada de pase en la que el liniero fue al pasador (sin asignación de cobertura).
    dl = plays[(plays.family == "DL") & plays["sack"].notna() & (plays.is_pass == 1)
               & (plays.coverage_assignment.isna() | plays.coverage_assignment.eq("NO_ASSIGNMENT"))].copy()
    dl["alignment"] = _alignment(dl.lined_up_position, codes["DL"])
    dl["pressure"] = dl["time_to_pressure"].notna().astype(float)
    dl["goe"] = _expected(dl, "player_get_off", ["alignment"]) - dl["player_get_off"]
    dl["proe"] = dl["pressure"] - _expected(dl, "pressure", ["alignment"])
    out["DL"] = dl

    # OL, snaps de protección de pase.
    ol = plays[(plays.family == "OL") & plays.peak_pressure_probability_allowed.notna()].copy()
    ol["alignment"] = _alignment(ol.lined_up_position, codes["OL"])
    q = rules["dropback_quantiles"]
    ol["dropback_bin"] = pd.qcut(ol.dropback_duration, q, labels=False, duplicates="drop").fillna(-1).astype(int)
    ol["ppoe"] = _expected(ol, "peak_pressure_probability_allowed", ["alignment", "dropback_bin"]) - ol.peak_pressure_probability_allowed
    # paoe solo donde pressure_allowed está registrado (antes: faltante = sin presión; aquí afecta a 0 filas).
    ol["paoe"] = _expected(ol, "pressure_allowed", ["alignment", "dropback_bin"]) - ol["pressure_allowed"]
    out["OL"] = ol

    # WR/TE, rutas con lectura de separación.
    wr = plays[(plays.family == "WRTE") & plays.separation_at_pass_forward.notna() & plays.route_ran.notna()].copy()
    wr["alignment"] = _alignment(wr.lined_up_position, codes["WRTE"])
    wr["soe"] = wr.separation_at_pass_forward - _expected(wr, "separation_at_pass_forward", ["route_ran", "alignment"],
                                                          fallback=["route_ran"], min_n=30)
    out["WRTE"] = wr
    return out


def shrink_frame(table: pd.DataFrame, metric: str, group: str = "combine_position") -> pd.DataFrame:
    """Contracción por Bayes empírico de las medias de jugadores hacia la media del grupo (mismo estimador que el SBI)."""
    res = []
    for _, g in table.groupby(group, observed=True):
        n, mean = g[f"n_{metric}"], g[f"{metric}_mean"]
        var = g[f"{metric}_var"].fillna(g[f"{metric}_var"].median())
        sigma2 = np.average(var, weights=np.maximum(n - 1, 1))
        total = mean.var(ddof=1) if len(g) > 1 else 0.0
        tau2 = max(total - np.mean(sigma2 / n), 0.05 * total) if total > 0 else np.nan
        w = n / (n + sigma2 / tau2) if np.isfinite(tau2) and tau2 > 0 else pd.Series(0.0, index=g.index)
        grand = np.average(mean, weights=n)
        res.append(pd.DataFrame({f"{metric}_eb": grand + w * (mean - grand), f"{metric}_rel": w}, index=g.index))
    return pd.concat(res) if res else pd.DataFrame(columns=[f"{metric}_eb", f"{metric}_rel"])


def player_onfield(fam_plays: dict[str, pd.DataFrame], prospects: pd.DataFrame, cfg: Config,
                   rookie_only: bool = False) -> pd.DataFrame:
    """Una fila por prospecto con cada métrica de campo válida (NaN donde no se mide)."""
    rules = cfg["performance_rules"]
    base = prospects[["nfl_id", "combine_position"]].copy()
    base["combine_position"] = base["combine_position"].astype(str)
    base["family"] = base["combine_position"].map(rules["families"])
    for m in ONFIELD_METRICS:
        fam = METRIC_FAMILY[m]
        d = fam_plays[fam]
        if rookie_only:
            d = d[d.rookie_season == 1]
        d = d.dropna(subset=[m])
        g = d.groupby("nfl_id")[m]
        st = pd.DataFrame({f"n_{m}": g.size(), f"{m}_mean": g.mean(), f"{m}_var": g.var(ddof=1)}).reset_index()
        st = st[st[f"n_{m}"] >= rules["min_plays"][MIN_PLAYS_KEY[fam]]]  # R13
        st = st.merge(base[["nfl_id", "combine_position"]], on="nfl_id", how="left")
        st = st.join(shrink_frame(st, m))
        base = base.merge(st.drop(columns=["combine_position", f"{m}_var"]), on="nfl_id", how="left")
        for c in (f"n_{m}", f"{m}_mean", f"{m}_eb", f"{m}_rel"):
            base[c] = pd.to_numeric(base[c], errors="coerce").astype("float64")
        # z dentro de la familia del valor contraído (positivo = mejor), para los clasificadores.
        base[f"{m}_z"] = np.nan
        fam_rows = base.family == fam
        v = base.loc[fam_rows, f"{m}_eb"]
        base.loc[fam_rows, f"{m}_z"] = (v - v.mean()) / v.std(ddof=1)
    prim = rules["primary_metric"]
    base["perf_index"] = np.nan
    for fam, m in prim.items():
        rows = base.family == fam
        base.loc[rows, "perf_index"] = base.loc[rows, f"{m}_z"]
    return base


def reliability_table(fam_plays: dict[str, pd.DataFrame], profiles: pd.DataFrame) -> pd.DataFrame:
    """ICC (método de momentos), peso medio de Bayes empírico y ANOVA de una vía por métrica de campo x posición."""
    rows = []
    for m in ONFIELD_METRICS:
        fam = METRIC_FAMILY[m]
        qualified = profiles.loc[profiles[f"n_{m}"].notna(), "nfl_id"]
        d = fam_plays[fam].dropna(subset=[m])
        d = d[d.nfl_id.isin(qualified)]
        for pos, g in d.groupby("combine_position"):
            gg = g.groupby("nfl_id")[m]
            n, mu, var = gg.size(), gg.mean(), gg.var(ddof=1)
            if len(n) < 3:
                continue
            sigma2 = np.average(var.fillna(var.median()), weights=np.maximum(n - 1, 1))
            total = mu.var(ddof=1)
            tau2 = total - np.mean(sigma2 / n)
            f, p = stats.f_oneway(*[v.to_numpy() for _, v in gg])
            w = n / (n + sigma2 / tau2) if tau2 > 0 else pd.Series(0.0, index=n.index)
            rows.append({"metric": m, "family": fam, "combine_position": pos, "players": len(n),
                         "median_plays": float(n.median()), "icc": tau2 / total if total > 0 else np.nan,
                         "mean_weight": float(w.mean()), "anova_f": float(f), "anova_p": float(p)})
    return pd.DataFrame(rows)


def rookie_stability(fam_plays: dict[str, pd.DataFrame], cfg: Config) -> pd.DataFrame:
    """¿Predice la temporada de novato las siguientes? Spearman de la media de novato frente a la posterior."""
    rules = cfg["performance_rules"]
    rows = []
    for m in ONFIELD_METRICS:
        fam = METRIC_FAMILY[m]
        d = fam_plays[fam].dropna(subset=[m])
        d = d[d.draft_year < rules.get("stability_last_class", 2025)]
        mn = rules["min_plays"][MIN_PLAYS_KEY[fam]]
        a = d[d.rookie_season == 1].groupby("nfl_id")[m].agg(["mean", "size"])
        b = d[d.rookie_season == 0].groupby("nfl_id")[m].agg(["mean", "size"])
        j = a[a["size"] >= mn].join(b[b["size"] >= mn], lsuffix="_rookie", rsuffix="_later", how="inner")
        if len(j) >= 8:
            r, p = stats.spearmanr(j["mean_rookie"], j["mean_later"])
            rows.append({"metric": m, "family": fam, "players": len(j), "rho": r, "p_value": p})
    return pd.DataFrame(rows)


def context_adjusted_burst(track_plays: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    """Snap burst neto del contexto de la jugada, para volver a comprobar la fiabilidad.

    WR/TE: ruta corrida; DB: asignación de cobertura en el snap y pase/carrera; OL/DL: alineación y pase/carrera.
    """
    d = track_plays.merge(context, on=KEY, how="left")
    pos = d["combine_position"].astype(str)
    d["context"] = np.where(pos.isin(["WR", "TE"]), d["route_ran"].fillna("NONE"),
                   np.where(pos.eq("DB"), d["coverage_assignment_at_snap"].fillna("NONE") + "|" + d["is_pass"].astype(str),
                            d["lined_up_position"].fillna("NONE") + "|" + d["is_pass"].astype(str)))
    grp = d.groupby([pos, "context"])["burst_2s"]
    small = grp.transform("count") < 5
    exp = grp.transform("mean").where(~small, d.groupby(pos)["burst_2s"].transform("mean"))
    d["burst_2s_ctx"] = d["burst_2s"] - exp
    return d


# =============================================================================
# Tracking de las pruebas del Combine (archivo 3 de la especificación, regla R16)
# =============================================================================
# Los resultados oficiales del Combine son tiempos de cronómetro; el tracking de las pruebas muestra
# cómo se produjo cada tiempo. Las métricas usan las mismas definiciones que el tracking de partido.
#
# Hallazgo de calibración (v2.2.0): **el inicio de un intento registrado no es repetible.** Un 40
# registrado empieza en un momento variable de la salida (velocidad del primer cuadro 0.2-3.2 yd/s),
# así que todo lo cronometrado desde el primer cuadro (distancia a 1 s / 2 s, parcial de 10 yd desde
# la salida, tiempo al 90% de la velocidad máxima) apenas se repite entre los dos intentos de un
# jugador (parcial 0-5 yd: r = 0.16). Los parciales entre marcas de distancia no dependen de cuándo
# empezó el reloj y se repiten bien (5-10 yd 0.87, 10-20 yd 0.98, velocidad máxima 0.99). Por eso las
# variables pasan por un filtro de fiabilidad, la misma regla que siguen las métricas de partido (ICC
# antes de usar): una métrica solo puede ser variable del modelo si se repite con r >= 0.5, y cada
# constructo recibe una sola variable (su métrica más fiable). `attempt_reliability` escribe cada
# métrica, usada o no y por qué, en reportes/metricas/ct_reliability.csv.
#
# Por intento (filas PLAYER; el intento empieza en el cuadro anterior al primero con velocidad >= start_speed)
#   top_speed, peak_accel, peak_decel (frenado suavizado en 3 cuadros moviéndose a >= 1 yd/s), max_cut_deg,
#   burst_1s / burst_2s, t90_top_s, duration_s
#   40 yardas: split_0_5_s, split_5_10_s, split_10_20_s (tiempo entre marcas, interpolado),
#              split_10yd_s (desde la salida, como el parcial oficial)
#   3 conos y shuttle: n_plants y plant_speed_ratio (mínimos de velocidad por debajo del 50% del pico
#              anterior; razón media entre la velocidad en el giro y ese pico = velocidad conservada en el giro)
#
# Variables por jugador (CT_FEATURES; mejor intento por métrica, como el Combine oficial conserva el mejor)
#   constructo               variable                       fiabilidad en el archivo entregado
#   aceleración de sprint    ct_forty_split_5_10_s          0.87 (5 -> 10 yd; menor = mejor)
#   velocidad de sprint      ct_forty_top_speed             0.99 (mismo constructo que el parcial 10-20 yd, 0.98)
#   cambio de dirección      ct_shuttle_plant_speed_ratio   0.54 (velocidad conservada en los giros del shuttle)
#   velocidad en pruebas     ct_skill_speed_z               velocidad máxima en z dentro de cada prueba
#   de posición                                             (grupo de posición x nombre de prueba), media del jugador
#   aceleración en pruebas   ct_skill_accel_z               aceleración máxima, misma escala
#   de posición
#
# Regla de negocio (configuracion/config.yaml -> combine_tracking_rules)
#   R16 se quitan las filas del balón y las filas sin nfl_id; los cuadros deben estar a 0.1 s dentro de
#       un intento; velocidad / aceleración por encima de los topes de partido -> NaN e interpolación;
#       se descartan los intentos más cortos que `min_frames` o que nunca alcanzan `start_speed`.
ATTEMPT_KEY = ["event_id"]
FORTY, THREE_CONE, SHUTTLE = "FORTY_YARD_DASH", "THREE_CONE_DRILL", "SHORT_SHUTTLE"
SKILL_PREFIX = "SKILL_DRILLS_"

CT_FEATURES = [
    "ct_forty_split_5_10_s", "ct_forty_top_speed", "ct_shuttle_plant_speed_ratio",
    "ct_skill_speed_z", "ct_skill_accel_z",
]
# Para estas, menor es mejor (tiempos); para el resto, mayor es mejor.
CT_LOWER_IS_BETTER = {"ct_forty_split_5_10_s"}

_BEST = {  # grupo de prueba -> {métrica del intento: "max" | "min"} para las variables del jugador
    "forty": {"split_5_10_s": "min", "top_speed": "max"},
    "shuttle": {"plant_speed_ratio": "max"},
}
_SKILL_Z = {"top_speed": "speed", "peak_accel": "accel"}

# Métricas cuya fiabilidad entre intentos se reporta (sean variable o no).
RELIABILITY_METRICS = {
    "forty": ["split_0_5_s", "split_5_10_s", "split_10_20_s", "split_10yd_s", "top_speed", "peak_accel",
              "burst_1s", "burst_2s", "t90_top_s"],
    "3cone": ["duration_s", "top_speed", "peak_accel", "peak_decel", "plant_speed_ratio", "max_cut_deg"],
    "shuttle": ["duration_s", "top_speed", "peak_accel", "peak_decel", "plant_speed_ratio", "max_cut_deg"],
    "skill": ["top_speed", "peak_accel", "peak_decel", "max_cut_deg", "burst_1s"],
}
FEATURE_SOURCE = {("forty", "split_5_10_s"): "ct_forty_split_5_10_s", ("forty", "top_speed"): "ct_forty_top_speed",
                  ("shuttle", "plant_speed_ratio"): "ct_shuttle_plant_speed_ratio",
                  ("skill", "top_speed"): "ct_skill_speed_z", ("skill", "peak_accel"): "ct_skill_accel_z"}
# Por qué una métrica que pasa el filtro de fiabilidad no es variable (una variable por constructo).
NOT_USED = {("forty", "split_10_20_s"): "mismo constructo que la velocidad máxima (r = -0.95), algo menos fiable",
            ("forty", "peak_accel"): "mismo constructo que el parcial 5-10 yd, menos fiable",
            ("3cone", "duration_s"): "duplica el tiempo oficial de 3 conos",
            ("3cone", "top_speed"): "velocidad de sprint, ya medida en el 40"}

DEFAULT_RULES = {"frame_rate_hz": 10, "start_speed": 0.5, "min_frames": 15, "speed_cap": 12.5,
                 "accel_cap": 15.0, "plant_drop": 0.5, "cut_window_s": 0.5, "cut_min_speed": 2.0,
                 "min_reliability": 0.5}
DRILL_KEY = ["drill_type", "drill_name"]  # una prueba tal como la corre un grupo de posición (GAUNTLET_DRILL: WR, TE, DB)


def _rules(cfg: Config) -> dict:
    return {**DEFAULT_RULES, **(cfg.raw.get("combine_tracking_rules") or {})}


def drill_group(drill_type: pd.Series) -> pd.Series:
    """Tipo de prueba -> grupo: forty, 3cone, shuttle, skill (pruebas de posición) u other."""
    d = drill_type.astype(str)
    return pd.Series(np.select([d.eq(FORTY), d.eq(THREE_CONE), d.eq(SHUTTLE), d.str.startswith(SKILL_PREFIX)],
                               ["forty", "3cone", "shuttle", "skill"], "other"), index=drill_type.index)


def clean_combine_tracking(ct: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """R16: solo cuadros de jugador, espaciado regular a 10 Hz, velocidad / aceleración topadas e interpoladas."""
    r = _rules(cfg)
    audit = {"ct_frames_in": len(ct), "ct_attempts_in": int(ct["event_id"].nunique())}
    df = ct[(ct["entity_type"].astype(str) == "PLAYER") & ct["nfl_id"].notna()].copy()
    audit["ct_frames_not_player"] = audit["ct_frames_in"] - len(df)
    df["nfl_id"] = df["nfl_id"].astype("int64")
    df["time"] = pd.to_datetime(df["time"], format="ISO8601")
    df = df.sort_values(ATTEMPT_KEY + ["time"]).reset_index(drop=True)
    gaps = df.groupby(ATTEMPT_KEY)["time"].diff().dt.total_seconds()
    bad = df.loc[gaps.notna() & ~np.isclose(gaps, 1.0 / r["frame_rate_hz"]), "event_id"].unique()
    audit["ct_attempts_frame_gaps"] = len(bad)
    df = df[~df["event_id"].isin(bad)]
    for col, cap in (("s", r["speed_cap"]), ("a", r["accel_cap"])):
        over = df[col] > cap
        audit[f"ct_frames_{col}_above_cap"] = int(over.sum())
        df.loc[over, col] = np.nan
        df[col] = df.groupby(ATTEMPT_KEY)[col].transform(lambda s: s.interpolate(limit_direction="both"))
    df["drill_group"] = drill_group(df["drill_type"])
    audit["ct_frames_out"] = len(df)
    return df.reset_index(drop=True), audit


def _time_to_distance(disp: np.ndarray, hz: int, target: float) -> float:
    """Primer instante (s) en que el desplazamiento alcanza `target`, interpolado linealmente entre cuadros."""
    idx = np.flatnonzero(disp >= target)
    if not len(idx):
        return np.nan
    i = idx[0]
    if i == 0:
        return 0.0
    frac = (target - disp[i - 1]) / (disp[i] - disp[i - 1]) if disp[i] > disp[i - 1] else 0.0
    return float((i - 1 + frac) / hz)


def _plants(s_smooth: np.ndarray, drop: float) -> tuple[int, float]:
    """Mínimos de velocidad por debajo de `drop` x el pico anterior (los giros del 3 conos / shuttle)."""
    n, ratios, peak = 0, [], 0.0
    for i in range(1, len(s_smooth) - 1):
        peak = max(peak, s_smooth[i])
        if s_smooth[i] <= s_smooth[i - 1] and s_smooth[i] < s_smooth[i + 1] and peak > 0 and s_smooth[i] < drop * peak:
            n += 1
            ratios.append(s_smooth[i] / peak)
            peak = 0.0
    return n, float(np.mean(ratios)) if ratios else np.nan


def attempt_metrics(frames: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """Una fila por intento con las métricas de movimiento (mismas definiciones que en partido)."""
    r = _rules(cfg)
    hz = int(r["frame_rate_hz"])
    lag = int(r["cut_window_s"] * hz)
    out, no_start, short = [], 0, 0
    for eid, d in frames.groupby("event_id", sort=False):
        s_all = d["s"].to_numpy()
        start = np.flatnonzero(s_all >= r["start_speed"])
        if not len(start):
            no_start += 1
            continue
        d = d.iloc[max(start[0] - 1, 0):]  # el cuadro anterior al primero en movimiento = punto de salida
        if len(d) < r["min_frames"]:
            short += 1
            continue
        x, y, s, a = (d[c].to_numpy() for c in ("x", "y", "s", "a"))
        disp = np.hypot(x - x[0], y - y[0])
        s_sm = pd.Series(s).rolling(3, center=True, min_periods=1).mean().to_numpy()
        dsdt = np.gradient(s_sm, 1.0 / hz)
        moving = s_sm >= 1.0
        top = float(s.max())
        h = np.deg2rad(d["dir"].to_numpy())
        dh = np.abs(np.rad2deg(np.angle(np.exp(1j * (h[lag:] - h[:-lag]))))) if len(h) > lag else np.array([])
        fast = (s[lag:] >= r["cut_min_speed"]) & (s[:-lag] >= r["cut_min_speed"]) if len(h) > lag else np.array([], bool)
        grp = d["drill_group"].iloc[0]
        row = {"event_id": eid, "nfl_id": int(d["nfl_id"].iloc[0]), "draft_year": int(d["draft_year"].iloc[0]),
               "drill_type": d["drill_type"].iloc[0], "drill_name": d["drill_name"].iloc[0], "drill_group": grp,
               "attempt": int(d["attempt"].iloc[0]), "n_frames": len(d), "duration_s": (len(d) - 1) / hz,
               "burst_1s": float(disp[hz]) if len(disp) > hz else np.nan,
               "burst_2s": float(disp[2 * hz]) if len(disp) > 2 * hz else np.nan,
               "top_speed": top,
               "t90_top_s": float(np.flatnonzero(s >= 0.9 * top)[0] / hz) if top > 0 else np.nan,
               "peak_accel": float(np.nanmax(a)),
               "peak_decel": float(-dsdt[moving].min()) if moving.any() else np.nan,
               "max_cut_deg": float(dh[fast].max()) if fast.any() else np.nan,
               "split_10yd_s": np.nan, "split_0_5_s": np.nan, "split_5_10_s": np.nan, "split_10_20_s": np.nan,
               "n_plants": np.nan, "plant_speed_ratio": np.nan}
        if grp == "forty":
            t5, t10, t20 = (_time_to_distance(disp, hz, m) for m in (5.0, 10.0, 20.0))
            row.update({"split_10yd_s": t10, "split_0_5_s": t5, "split_5_10_s": t10 - t5, "split_10_20_s": t20 - t10})
        if grp in ("3cone", "shuttle"):
            row["n_plants"], row["plant_speed_ratio"] = _plants(s_sm, r["plant_drop"])
        out.append(row)
    audit = {"ct_attempts_no_start": no_start, "ct_attempts_too_short": short, "ct_attempts_kept": len(out)}
    return pd.DataFrame(out), audit


def _skill_z(attempts: pd.DataFrame, metrics) -> pd.DataFrame:
    """Intentos de pruebas de posición con cada métrica en z dentro de su prueba (grupo de posición x nombre).

    Las pruebas difieren por diseño, y el mismo nombre de prueba lo pueden correr varios grupos de
    posición (el gauntlet lo corren WR, TE y DB), así que el grupo de referencia es la prueba tal como
    la corre un grupo de posición.
    """
    sk = attempts[attempts.drill_group == "skill"].copy()
    for m in metrics:
        g = sk.groupby(DRILL_KEY)[m]
        sk[f"{m}_z"] = (sk[m] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    return sk


def attempt_reliability(attempts: pd.DataFrame, min_reliability: float = 0.5) -> pd.DataFrame:
    """Qué tan bien se repite cada métrica dentro de un jugador (sin etiquetas; decide qué métricas son variables).

    40 / 3 conos / shuttle: r de Pearson entre los dos primeros intentos del jugador (test-retest).
    Pruebas de posición: r de mitades del z medio dentro de la prueba (intentos alternos), corregido con
    Spearman-Brown al conjunto completo de intentos, que es lo que promedia la variable.
    """
    rows = []
    for grp in ("forty", "3cone", "shuttle"):
        a = attempts[attempts.drill_group == grp].sort_values(["nfl_id", "attempt"]).copy()
        a["k"] = a.groupby("nfl_id").cumcount()
        for m in RELIABILITY_METRICS[grp]:
            if a.empty or m not in a.columns:
                rows.append({"drill_group": grp, "metric": m, "method": "test-retest (intentos 1 y 2)",
                             "n_players": 0, "reliability": np.nan})
                continue
            w = a[a.k < 2].pivot(index="nfl_id", columns="k", values=m).dropna()
            r = float(stats.pearsonr(w[0], w[1])[0]) if len(w) >= 10 and w.shape[1] == 2 else np.nan
            rows.append({"drill_group": grp, "metric": m, "method": "test-retest (intentos 1 y 2)",
                         "n_players": len(w), "reliability": r})
    skill_metrics = [m for m in RELIABILITY_METRICS["skill"] if m in attempts.columns]
    sk = _skill_z(attempts, skill_metrics).sort_values(["nfl_id", *DRILL_KEY, "attempt"])
    sk["half"] = sk.groupby("nfl_id").cumcount() % 2
    for m in RELIABILITY_METRICS["skill"]:
        if sk.empty or m not in skill_metrics:
            rows.append({"drill_group": "skill", "metric": m, "method": "mitades, Spearman-Brown",
                         "n_players": 0, "reliability": np.nan})
            continue
        h = sk.groupby(["nfl_id", "half"])[f"{m}_z"].mean().unstack().dropna()
        r = float(stats.pearsonr(h[0], h[1])[0]) if len(h) >= 10 and h.shape[1] == 2 else np.nan
        rows.append({"drill_group": "skill", "metric": m, "method": "mitades, Spearman-Brown",
                     "n_players": len(h), "reliability": 2 * r / (1 + r) if np.isfinite(r) else np.nan})
    out = pd.DataFrame(rows)
    out["feature"] = [FEATURE_SOURCE.get((g, m), "") for g, m in zip(out.drill_group, out.metric)]
    out["passes_gate"] = out["reliability"] >= min_reliability
    out["note"] = [NOT_USED.get((g, m), "") if p and not f else ("" if p else "por debajo del filtro de fiabilidad")
                   for g, m, p, f in zip(out.drill_group, out.metric, out.passes_gate, out.feature)]
    return out


def player_features(attempts: pd.DataFrame, nfl_ids: pd.Series | None = None) -> pd.DataFrame:
    """Mejor intento por métrica (`ct_<grupo>_<métrica>`) más el z medio dentro de la prueba en pruebas de posición."""
    ids = pd.Index(sorted(attempts["nfl_id"].unique()) if nfl_ids is None else nfl_ids, name="nfl_id")
    feats = pd.DataFrame(index=ids)
    for grp, metrics in _BEST.items():
        sub = attempts[attempts.drill_group == grp]
        for m, how in metrics.items():
            feats[f"ct_{grp}_{m}"] = sub.groupby("nfl_id")[m].agg(how).reindex(ids)
    sk = _skill_z(attempts, list(_SKILL_Z))
    for m, name in _SKILL_Z.items():
        feats[f"ct_skill_{name}_z"] = sk.groupby("nfl_id")[f"{m}_z"].mean().reindex(ids)
    feats["ct_attempts"] = attempts.groupby("nfl_id").size().reindex(ids).fillna(0).astype(int)
    feats["ct_missing"] = feats[CT_FEATURES].isna().all(axis=1).astype(int)
    return feats[CT_FEATURES + ["ct_attempts", "ct_missing"]].reset_index()


def official_vs_tracking(features: pd.DataFrame, prospects: pd.DataFrame, group: str = "combine_position") -> pd.DataFrame:
    """Validez convergente: ¿reproduce el tracking el cronómetro oficial?

    `spearman` / `pearson` juntan a todos los prospectos; `within_position_spearman` quita antes las
    medias del grupo de posición (como todas las tablas de traducción), que es la comparación que
    importa para un modelo dentro de la posición.
    """
    pairs = [("ct_forty_split_5_10_s", "ten_yd_split"), ("ct_forty_split_5_10_s", "forty"),
             ("ct_forty_top_speed", "forty"), ("ct_forty_top_speed", "ten_yd_split"),
             ("ct_shuttle_plant_speed_ratio", "short_shuttle"), ("ct_skill_speed_z", "forty"),
             ("ct_skill_accel_z", "ten_yd_split")]
    official = prospects[["nfl_id"] + [c for c in prospects.columns if c not in features.columns]]
    df = features.merge(official, on="nfl_id", how="inner")
    rows = []
    for ct_col, off in pairs:
        if ct_col not in df or off not in df:
            continue
        t = df[[ct_col, off, group]].dropna()
        if len(t) < 10:
            continue
        rho, p = stats.spearmanr(t[ct_col], t[off])
        r, _ = stats.pearsonr(t[ct_col], t[off])
        g = t.groupby(t[group].astype(str))
        w = t[[ct_col, off]] - g[[ct_col, off]].transform("mean")
        rho_w, p_w = stats.spearmanr(w[ct_col], w[off])
        rows.append({"tracking_feature": ct_col, "official_result": off, "n": len(t), "spearman": rho,
                     "pearson": r, "p_value": p, "within_position_spearman": rho_w, "within_position_p": p_w})
    return pd.DataFrame(rows)


def build_combine_tracking(ct_raw: pd.DataFrame, prospects: pd.DataFrame, cfg: Config
                           ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Tracking crudo del Combine -> (tabla de intentos, variables para cada prospecto, fiabilidad, auditoría)."""
    frames, a1 = clean_combine_tracking(ct_raw, cfg)
    attempts, a2 = attempt_metrics(frames, cfg)
    if len(attempts):
        feats = player_features(attempts, prospects["nfl_id"])
        rel = attempt_reliability(attempts, _rules(cfg)["min_reliability"])
    else:
        feats = pd.DataFrame({"nfl_id": prospects["nfl_id"], **{c: np.nan for c in CT_FEATURES},
                              "ct_attempts": 0, "ct_missing": 1})
        rel = pd.DataFrame()
    audit = {**a1, **a2, "ct_players_with_features": int((feats["ct_missing"] == 0).sum()),
             **{f"ct_players_with_{c.removeprefix('ct_')}": int(feats[c].notna().sum()) for c in CT_FEATURES}}
    return attempts, feats, rel, audit


# =============================================================================
# Conjuntos de variables y tablas de modelado
# =============================================================================
# Los conjuntos de variables están anidados para que cada paso responda una sola pregunta:
#   combine             lo que tiene un scout antes del draft (pruebas atléticas + perfil)
#   combine_draft       + capital de draft (lo que decidió la liga)
#   combine_draft_mov   + movimiento en partido (tracking posterior al draft; diagnóstico, no pre-draft)
#   combine_draft_proj  + rendimiento en el campo proyectado desde el Combine. Los valores observados
#                         entran solo como etiquetas de entrenamiento de un proyector apilado dentro del
#                         pipeline (train.OnFieldProjector); al predecir solo se usan datos del Combine,
#                         así que el conjunto se puede usar el día del draft.
#   combine_draft_perf  + rendimiento observado en el campo (posterior al draft; diagnóstico)
#   combine_draft_ct    + variables del tracking de pruebas del Combine (pre-draft)
#   ATHLETIC_CT         variables atléticas de regresión + tracking de pruebas (con frente a sin tracking)
DERIVED = ["speed_score", "ten_split_share", "ten_split_momentum"]
MISSING_FLAGS = [f"{d}_missing" for d in DRILLS]
CATEGORICAL = ["combine_position"]

# Regresión: traducción atlética pura (sin información de draft ni de producción).
ATHLETIC = MEASURABLES + DRILLS + DERIVED + MISSING_FLAGS + ["n_drills_run", "ngs_athleticism_score", "age_at_draft"]

FEATURE_SETS: dict[str, list[str]] = {
    "combine": ATHLETIC + ["ngs_college_production_score", "power_conference"],
}
FEATURE_SETS["combine_draft"] = FEATURE_SETS["combine"] + ["log_draft_pick", "is_undrafted"]
MOVEMENT_FEATURES = ["sbi"] + [f"{m}_z" for m in METRICS if m != "burst_2s"]
FEATURE_SETS["combine_draft_mov"] = FEATURE_SETS["combine_draft"] + MOVEMENT_FEATURES

# Conjuntos de rendimiento en el campo. paoe queda fuera de ambos: es el gemelo binario de ppoe (misma familia).
PROJECTED = ["goe", "proe", "ppoe", "soe"]
PROJECTOR_TARGETS = [f"onfield_target_{m}" for m in PROJECTED]
PERF_FEATURES = [f"{m}_z" for m in PROJECTED]
FEATURE_SETS["combine_draft_proj"] = FEATURE_SETS["combine_draft"] + PROJECTOR_TARGETS
FEATURE_SETS["combine_draft_perf"] = FEATURE_SETS["combine_draft"] + PERF_FEATURES

# Tracking de pruebas del Combine (solo se usa si el archivo está presente).
CT_SET = list(CT_FEATURES)
ATHLETIC_CT = ATHLETIC + CT_SET
FEATURE_SETS["combine_draft_ct"] = FEATURE_SETS["combine_draft"] + CT_SET


def has_combine_tracking(prospects: pd.DataFrame) -> bool:
    """True si la tabla de prospectos trae variables de tracking de pruebas para al menos un jugador."""
    return all(c in prospects.columns for c in CT_SET + ["ct_missing"]) and bool((prospects["ct_missing"] == 0).any())


def regression_table(prospects: pd.DataFrame, profiles: pd.DataFrame, target: str) -> pd.DataFrame:
    """Prospectos con perfil de movimiento y el objetivo de regresión (SBI)."""
    keep = list(dict.fromkeys(["nfl_id", target, "sbi", "burst_2s_eb", "sbi_reliability", "n_plays"]))
    return profiles[keep].merge(prospects, on="nfl_id", how="inner", validate="1:1")


def classification_table(prospects: pd.DataFrame, profiles: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tabla del clasificador de titulares; con `profiles` agrega las variables de movimiento."""
    if profiles is None:
        return prospects.copy()
    cols = ["nfl_id", "n_plays", "burst_2s_eb"] + MOVEMENT_FEATURES
    return profiles[cols].merge(prospects, on="nfl_id", how="inner", validate="1:1")


def onfield_regression_table(prospects: pd.DataFrame, onfield: pd.DataFrame, target: str,
                             positions: list[str]) -> pd.DataFrame:
    """Jugadores de las posiciones del Combine indicadas con un valor válido de `target`."""
    metric = target.split("_")[0]
    cols = ["nfl_id", target, f"{metric}_rel", f"n_{metric}"]
    df = onfield[cols].dropna(subset=[target]).merge(prospects, on="nfl_id", how="inner", validate="1:1")
    return df[df.combine_position.astype(str).isin(positions)]


def projector_table(prospects: pd.DataFrame, onfield: pd.DataFrame) -> pd.DataFrame:
    """Todos los prospectos + sus valores de campo contraídos como etiquetas del proyector (NaN si no se miden)."""
    lab = onfield[["nfl_id"] + [f"{m}_eb" for m in PROJECTED]].rename(
        columns={f"{m}_eb": f"onfield_target_{m}" for m in PROJECTED})
    return prospects.merge(lab, on="nfl_id", how="left", validate="1:1")


def onfield_classification_table(prospects: pd.DataFrame, onfield: pd.DataFrame) -> pd.DataFrame:
    """Jugadores con índice de campo; las métricas fuera de la familia del jugador valen 0 (= media de la familia)."""
    df = onfield[["nfl_id", "perf_index"] + PERF_FEATURES].dropna(subset=["perf_index"])
    df = df.merge(prospects, on="nfl_id", how="inner", validate="1:1")
    df[PERF_FEATURES] = df[PERF_FEATURES].fillna(0.0)
    return df


def xy(df: pd.DataFrame, feature_set: list[str], target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Matriz X (posición + variables del conjunto) y objetivo y."""
    cols = list(dict.fromkeys(CATEGORICAL + feature_set))
    return df[cols].copy(), df[target].copy()


def _role(nfl_position: pd.Series) -> pd.Series:
    """Rol aproximado en la NFL; solo equilibra los pliegues y los controles por rol (nunca es variable)."""
    return nfl_position.astype(str).replace({"DE": "edge", "OLB": "edge", "DT": "interior", "NT": "interior"})


def _strata(roles: pd.Series, min_size: int) -> pd.Series:
    """Estratos de la validación cruzada: los roles con menos de `min_size` jugadores se agrupan en "other"."""
    counts = roles.value_counts()
    return roles.where(roles.map(counts) >= min_size, "other")


# =============================================================================
# Etapas de preparación
# =============================================================================
def stage_contract(cfg: Config) -> dict:
    """Valida los archivos originales contra configuracion/contrato_datos.yaml (la especificación)."""
    ccfg = cfg.raw.get("contract", {})
    contract = load_contract(ccfg.get("path"))
    report, summary = validate_all(cfg, contract)
    write_report(cfg, report, summary)
    log.info("contrato de datos: %s (%d comprobaciones, %d errores, %d advertencias, faltan %s)", summary["status"],
             summary["n_checks"], summary["n_errors"], summary["n_warnings"], [m["spec_name"] for m in summary["missing_files"]])
    if summary["n_errors"] and ccfg.get("fail_on_error", True):
        bad = report[(report.severity == "error") & (report.result == "fail")]
        raise RuntimeError("errores en el contrato de datos:\n" + bad[["table", "file", "column", "check", "detail"]].to_string())
    return summary


def _process_tracking(cfg: Config, games: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Limpia el tracking y calcula las métricas por jugada, un archivo a la vez (temporadas completas).

    R8b conserva los tipos de temporada de `tracking_rules.season_types`. R8c: un archivo con exactamente
    1,048,575 filas se cortó en el límite de una hoja de cálculo, así que se descarta su última
    jugador-jugada. Una jugada cuyos cuadros quedan repartidos entre partes se aparta y se procesa al
    final. Los cuadros limpios se agregan a datos/datos_limp/intermedios/tracking_frames.parquet. El
    orden de salida es el de la clave de jugada, idéntico a procesar los archivos concatenados.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    rules = cfg["tracking_rules"]
    keep_games = set(games.loc[games.season_type.isin(rules.get("season_types", ["REG", "POST"])), "game_id"])
    key = ["game_id", "play_id", "nfl_id"]
    paths = table_files(cfg, "tracking")
    keysets = [pd.read_csv(p, usecols=key).drop_duplicates() for p in paths]
    spanning = (pd.concat(keysets, ignore_index=True).value_counts().reset_index(name="n").query("n > 1")[key]
                if len(keysets) > 1 else pd.DataFrame(columns=key))
    audit: dict = {"tracking_files": len(paths), "frames_excluded_season_type": 0, "plays_split_across_files": len(spanning)}
    out_path = cfg.path("interim") / "tracking_frames.parquet"
    writer, plays, held = None, [], []

    def run(frames_raw: pd.DataFrame) -> None:
        nonlocal writer
        frames, fa = clean_tracking(frames_raw, cfg)
        pm, pa_ = play_metrics(frames, cfg)
        for k, v in {**fa, **pa_}.items():
            audit[k] = audit.get(k, 0) + v
        plays.append(pm)
        table = pa.Table.from_pandas(frames, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table.cast(writer.schema))

    audit["plays_dropped_cut_file"] = 0
    for name, df in iter_tracking(cfg):
        if len(df) == SPREADSHEET_ROW_LIMIT:  # R8c: archivo cortado en el límite; su última jugada puede estar incompleta
            last = df.iloc[-1][key]
            cut = (df[key] == last.values).all(axis=1)
            audit["plays_dropped_cut_file"] += 1
            log.warning("%s tiene %d filas (límite de hoja de cálculo): se descarta su última jugador-jugada %s",
                        name, len(df), tuple(last))
            df = df[~cut]
        season_ok = df.game_id.isin(keep_games)
        audit["frames_excluded_season_type"] += int((~season_ok).sum())
        df = df[season_ok]
        if len(spanning):
            m = df.merge(spanning.assign(_span=1), on=key, how="left")["_span"].notna().to_numpy()
            held.append(df[m])
            df = df[~m]
        log.info("tracking %s: %d cuadros", name, len(df))
        run(df.reset_index(drop=True))
    if held:
        run(pd.concat(held, ignore_index=True))
    if writer is not None:
        writer.close()
    plays_df = pd.concat(plays, ignore_index=True) if plays else pd.DataFrame(columns=key)
    if plays_df.duplicated(key).any():
        raise RuntimeError("tracking: una jugada se procesó dos veces entre archivos")
    return plays_df.sort_values(key, kind="mergesort").reset_index(drop=True), audit


def stage_data(cfg: Config) -> dict:
    """Contrato + ingesta + limpieza + movimiento + rendimiento en el campo + manifiesto."""
    t0 = time.time()
    contract_summary = stage_contract(cfg)
    tables = load_raw(cfg, tracking=False)
    manifest = write_manifest(cfg)
    log.info("ingesta correcta: data_version=%s", manifest["data_version"])

    prospects, audit = build_prospects(tables, cfg)
    ct_audit = {}
    if "combine_tracking" in tables:  # archivo 3 de la especificación (opcional)
        attempts, ct_feats, ct_rel, ct_audit = build_combine_tracking(tables["combine_tracking"], prospects, cfg)
        _save(attempts, cfg, "combine_tracking_attempts.parquet")
        _save(ct_feats, cfg, "combine_tracking_features.parquet")
        _save(ct_rel, cfg, "ct_reliability.csv", key="metrics")
        prospects = prospects.merge(ct_feats, on="nfl_id", how="left", validate="1:1")
        log.info("tracking del Combine: %s", ct_audit)
    _save(prospects, cfg, "prospects.parquet")
    _save(audit, cfg, "cleaning_audit.csv", key="metrics")
    miss = missingness_by_position(prospects)
    miss.to_csv(cfg.path("metrics") / "drill_missingness_by_position.csv")

    plays, tracking_audit = _process_tracking(cfg, tables["games"])
    plays = plays.merge(prospects[["nfl_id", "combine_position"]], on="nfl_id", how="left")
    _save(plays, cfg, "play_metrics.parquet")

    profiles = player_profiles(plays, prospects, cfg)
    _save(profiles, cfg, "player_movement_profiles.parquet")

    # Búsqueda de anomalías en las métricas por jugada: z robusto (mediana/MAD) dentro del grupo de posición.
    flags = []
    for m in ["burst_2s", "top_speed", "max_decel", "peak_accel_2s"]:
        g = plays.groupby("combine_position", observed=True)[m]
        med = g.transform("median")
        mad = g.transform(lambda s: (s - s.median()).abs().median()) * 1.4826
        rz = (plays[m] - med) / mad
        flags.append(pd.DataFrame({"game_id": plays.game_id, "play_id": plays.play_id, "nfl_id": plays.nfl_id,
                                   "metric": m, "value": plays[m], "robust_z": rz})[rz.abs() > 3.5])
    anomalies = pd.concat(flags, ignore_index=True)
    _save(anomalies, cfg, "play_metric_anomalies.csv", key="metrics")

    # 3b rendimiento en el campo
    onf_audit, onfield, onfield_rookie, fam = {}, None, None, None
    if "player_play" in tables:
        pp_plays, context, onf_audit = clean_player_play(tables["player_play"], tables["games"], prospects, cfg)
        fam = family_plays(pp_plays, cfg)
        for k, v in fam.items():
            _save(v, cfg, f"onfield_plays_{k}.parquet")
        context.to_parquet(cfg.path("interim") / "play_context.parquet", index=False)
        onfield = player_onfield(fam, prospects, cfg)
        onfield_rookie = player_onfield(fam, prospects, cfg, rookie_only=True)
        _save(onfield, cfg, "player_onfield_profiles.parquet")
        _save(onfield_rookie, cfg, "player_onfield_rookie.parquet")
        onf_audit.update({f"onfield_plays_{k}": len(v) for k, v in fam.items()})
        onf_audit.update({f"players_with_{m}": int(onfield[f"{m}_eb"].notna().sum()) for m in ONFIELD_METRICS})
        onf_audit["players_with_perf_index"] = int(onfield.perf_index.notna().sum())

    audit_all = {**tracking_audit, **ct_audit, "cleaning_rule_changes": len(audit),
                 "players_tracked": int(plays.nfl_id.nunique()), "players_profiled": len(profiles),
                 "play_metric_anomalies": len(anomalies), **onf_audit,
                 "contract_status": contract_summary["status"], "contract_warnings": contract_summary["n_warnings"],
                 "seconds": round(time.time() - t0, 1)}
    _dump(audit_all, cfg, "data_audit.json")
    log.info("etapa de datos: %s", audit_all)
    return {"prospects": prospects, "plays": plays, "profiles": profiles, "frames": None,
            "onfield": onfield, "onfield_rookie": onfield_rookie, "fam": fam}


def load_processed(cfg: Config) -> dict:
    """Carga los datos limpios de datos/datos_limp (sin repetir la preparación)."""
    p = cfg.path("processed")
    d = {
        "prospects": pd.read_parquet(p / "prospects.parquet"),
        "plays": pd.read_parquet(p / "play_metrics.parquet"),
        "profiles": pd.read_parquet(p / "player_movement_profiles.parquet"),
        "onfield": None, "onfield_rookie": None, "fam": None,
    }
    if (p / "player_onfield_profiles.parquet").exists():
        d["onfield"] = pd.read_parquet(p / "player_onfield_profiles.parquet")
        d["onfield_rookie"] = pd.read_parquet(p / "player_onfield_rookie.parquet")
        d["fam"] = {k: pd.read_parquet(p / f"onfield_plays_{k}.parquet") for k in ("DL", "OL", "WRTE")}
    return d


# =============================================================================
# Línea de comandos
# =============================================================================
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Preparación de datos del proyecto NFL (contrato, ingesta y limpieza)")
    ap.add_argument("--config", default=None, help="archivo de configuración (por defecto configuracion/config.yaml)")
    ap.add_argument("--etapa", default="datos", choices=["datos", "contrato", "verificar"],
                    help="datos = contrato + ingesta + limpieza; contrato = solo validación; verificar = hashes del manifiesto")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    set_global_seed(cfg.seed)
    if args.etapa == "contrato":
        stage_contract(cfg)
    elif args.etapa == "verificar":
        bad = verify_manifest(cfg)
        if bad:
            raise SystemExit(f"hash distinto al del manifiesto: {bad}")
        log.info("manifiesto verificado: todos los archivos originales coinciden")
    else:
        stage_data(cfg)


if __name__ == "__main__":
    main()
