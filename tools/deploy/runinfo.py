# tools/deploy/runinfo.py
"""Run metadata stored INSIDE a GeoPackage, as key/value rows.

The surveyed area, checkpoint and thresholds that produced a set of bubbles
belong with the bubbles. A sidecar file is the thing that gets left behind the
first time someone copies the .gpkg somewhere, and the area is exactly the
value that must never get separated from the count it goes with.

Written with plain sqlite3 and registered in `gpkg_contents` as an attributes
table, so it is a legal GeoPackage table: QGIS lists it, GDAL ignores it when
reading the spatial layer, and `read_run_info` gets it back without either.
Values are JSON-encoded so nested entries (the tile grid) survive the round
trip. Absent table -> empty dict, so an externally produced gpkg still loads.
"""
from __future__ import annotations

import json
import os
import sqlite3

TABLE = "deploy_run_info"


def write_run_info(gpkg: str, info: dict, table: str = TABLE) -> str:
    con = sqlite3.connect(gpkg)
    try:
        con.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ('
                    'key TEXT PRIMARY KEY, value TEXT)')
        con.executemany(
            f'INSERT OR REPLACE INTO "{table}" (key, value) VALUES (?, ?)',
            [(k, json.dumps(v, default=str)) for k, v in info.items()])
        # Register as an attributes table. Best-effort: a file that is not a
        # GeoPackage still gets a usable metadata table, it just is not listed.
        try:
            con.execute(
                "INSERT OR REPLACE INTO gpkg_contents "
                "(table_name, data_type, identifier, description, last_change) "
                "VALUES (?, 'attributes', ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'))",
                (table, table, "tools.deploy run metadata"))
        except sqlite3.Error:
            pass
        con.commit()
    finally:
        con.close()
    return gpkg


def read_run_info(gpkg: str, table: str = TABLE) -> dict:
    if not gpkg or not os.path.exists(gpkg):
        return {}
    con = sqlite3.connect(gpkg)
    try:
        rows = con.execute(f'SELECT key, value FROM "{table}"').fetchall()
    except sqlite3.OperationalError:
        return {}
    finally:
        con.close()
    out = {}
    for k, v in rows:
        try:
            out[k] = json.loads(v)
        except (TypeError, ValueError):
            out[k] = v
    return out


def surveyed_area(gpkg: str):
    """Valid area behind a bubble set in m2, or None if it was not recorded."""
    v = read_run_info(gpkg).get("surveyed_area_m2")
    return None if v is None else float(v)