"""
Provenance ledger: hashes + trajectory + profiles for forensic replay.

Records: data hash, knot/config snapshot, seeds, hyperparams, per-iteration
trajectory (objective, residual, prime-support), final params, validation
activation profiles, env spec. Stored in reasoning.db:provenance_ledger.
Tolerances: heterogeneous 1e-6 obj / 1e-4 params / <1% active mismatch;
forced-determinism single-thread mode for bit-exact digests (opt-in).
"""
import hashlib
import json
import platform
import sys
import time
import config
from core import db
import logging
logger = logging.getLogger(__name__)


def _sha256_canonical(obj):
    try:
        s = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        s = str(obj)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _env_spec():
    try:
        import numpy as _np
        import scipy
        _scipy_v = scipy.__version__
    except Exception:
        _scipy_v = "unknown"
    try:
        import numpy as _np2
        _np_v = _np2.__version__
    except Exception:
        _np_v = "unknown"
    return {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": _np_v,
        "scipy": _scipy_v,
        "cpu": platform.machine(),
    }


def init_ledger():
    conn = db.db_connect("reasoning")
    try:
        cur = conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS provenance_ledger (
            run_id TEXT PRIMARY KEY,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            data_hash TEXT, config_snapshot TEXT, seeds_json TEXT,
            hyperparams_json TEXT, trajectory_json TEXT,
            params_hash TEXT, profiles_json TEXT, env_json TEXT,
            root_digest TEXT
        )""")
        conn.commit()
    finally:
        try:
            conn.close()
        except Exception:
            pass


def record_run(run_id=None, data_ref=None, config_snapshot=None, seeds=None,
               hyperparams=None, trajectory=None, params=None, profiles=None):
    """Record a run bundle, return (run_id, root_digest). Generic, no doc-specific content."""
    import uuid as _uuid
    init_ledger()
    run_id = run_id or f"run-{int(time.time())}-{_uuid.uuid4().hex[:8]}"
    data_hash = _sha256_canonical(data_ref) if data_ref is not None else ""
    cfg_snap = config_snapshot or {k: getattr(config, k, None) for k in (
        "GATE_LAM1", "GATE_LAM2", "GATE_LAM3", "GATE_LAM4", "GATE_LR",
        "MIN_FACT_CONFIDENCE", "NOVELTY_SIM_THRESHOLD") if hasattr(config, k)}
    cfg_json = json.dumps(cfg_snap, sort_keys=True, default=str)
    seeds_json = json.dumps(seeds or {}, sort_keys=True, default=str)
    hyper_json = json.dumps(hyperparams or {}, sort_keys=True, default=str)
    traj_json = json.dumps(trajectory or [], default=str)[:100000]
    params_hash = _sha256_canonical(params) if params is not None else ""
    profiles_json = json.dumps(profiles or {}, sort_keys=True, default=str)[:1000000]
    env_json = json.dumps(_env_spec(), sort_keys=True)
    root = _sha256_canonical([data_hash, cfg_json, seeds_json, hyper_json, traj_json, params_hash, profiles_json, env_json])
    conn = db.db_connect("reasoning")
    try:
        conn.execute("""INSERT OR REPLACE INTO provenance_ledger
            (run_id, data_hash, config_snapshot, seeds_json, hyperparams_json,
             trajectory_json, params_hash, profiles_json, env_json, root_digest)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (run_id, data_hash, cfg_json, seeds_json, hyper_json, traj_json, params_hash, profiles_json, env_json, root))
        conn.commit()
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return run_id, root
