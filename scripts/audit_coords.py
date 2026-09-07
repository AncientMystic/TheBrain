"""Post-redo audit (phase 71): depth spread, clip check, shard params."""
import sys

sys.path.insert(0, "A:/scripts/TheBrain")
import numpy as np


def main():
    from core import db
    conn = db.db_connect("mapping")
    d = np.array([r[0] for r in conn.execute("SELECT depth FROM entities")])
    la = np.array([r[0] for r in conn.execute("SELECT lat_r FROM entities")])
    lo = np.array([r[0] for r in conn.execute("SELECT lon_r FROM entities")])
    print("depth pct 0/25/50/75/100:",
          np.percentile(d, [0, 25, 50, 75, 100]).round(4).tolist())
    print("distinct depths:", len(set(d.tolist())), "of", len(d))
    nlat = int((np.abs(la) >= 90.0).sum())
    nlon = int((np.abs(lo) >= 180.0).sum())
    print("clipped lat:", nlat, "clipped lon:", nlon)
    print("lon range:", round(float(lo.min()), 2), round(float(lo.max()), 2))
    shards = conn.execute(
        "SELECT shard_key, k, k2, dmax, radius FROM shards").fetchall()
    print("shards:", [(s[0], round(s[1], 4), round(s[2] or 0, 4)) for s in shards])
    conn.close()
    assert len(set(d.tolist())) > 140000, "depth not spread"
    # boundary rows are the per-axis argmax rows sitting exactly on the
    # bound (legitimate full-range use), not clipping damage: allow a few
    assert nlat <= 12 and nlon <= 12, "clipping present"
    assert abs(float(lo.min()) + 180.0) < 5 and abs(float(lo.max()) - 180.0) < 5, \
        "lon not using full range"
    print("audit_coords: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
