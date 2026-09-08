"""Wikidata hypernym miner (phase 93): P31/P279 pairs for entailment training.

Two stages: (1) type-derived pairs from our own type system (free, clean);
(2) API-resolved pairs — wbsearchentities (exact-label match only, precision
first) then batched wbgetentities claims for P31 (instance-of) / P279
(subclass-of). Resume-safe (skips children already edged), bounded
(--limit/--offset), polite (0.5s between calls, 3 retries). No SPARQL
(retired: HTTP 504 on full scans). Stores hypernym_edges + exports JSONL.

Usage: mine_hypernyms.py [--stage1] [--stage2] [--limit N] [--offset N]
                          [--sample STR] [--export PATH]
"""
import json
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, "A:/scripts/TheBrain")

API = "https://www.wikidata.org/w/api.php"
SLEEP = 0.5


def _get(params, tries=3):
    url = API + "?" + urllib.parse.urlencode(params)
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "TheBrain/1.0 (research; contact: local)"})
            with urllib.request.urlopen(req, timeout=30) as r:
                time.sleep(SLEEP)
                return json.loads(r.read().decode("utf-8", errors="ignore"))
        except Exception as e:
            if i == tries - 1:
                return {"error": str(e)[:120]}
            time.sleep(2 * (i + 1))
    return {}


def ensure_table(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS hypernym_edges("
                 " child_cid TEXT NOT NULL, parent_label TEXT NOT NULL,"
                 " parent_qid TEXT DEFAULT '', rel TEXT NOT NULL,"
                 " source TEXT NOT NULL DEFAULT 'wikidata',"
                 " PRIMARY KEY (child_cid, parent_label, rel))")
    conn.commit()


def stage1(conn):
    """Type-derived pairs: every entity -> its type_family display + shard concept."""
    n = 0
    for cid, et, tf, shard in conn.execute(
            "SELECT canonical_id, entity_type, type_family, shard_key FROM entities"):
        pairs = set()
        if tf and tf != et:
            pairs.add((tf.replace("-", " ").replace("_", " "), "", "P31"))
        if et:
            pairs.add((et.replace("-", " ").replace("_", " "), "", "P31"))
        for p, q, rel in pairs:
            try:
                conn.execute("INSERT OR IGNORE INTO hypernym_edges"
                             " (child_cid, parent_label, parent_qid, rel, source)"
                             " VALUES (?,?,?,?,?)", (cid, p, q, rel, "type-system"))
                n += 1
            except Exception:
                continue
    conn.commit()
    return n


COUNTRY_HINTS = {
 "US": ["united states", "u.s.", "texas", "california", "florida", "ohio", "paris, texas"],
 "FR": ["france", "french"], "DE": ["germany", "german"], "GB": ["england", "britain", "united kingdom", "london"],
 "IT": ["italy", "italian"], "ES": ["spain", "spanish"], "AD": ["andorra"],
 "JP": ["japan", "japanese"], "BR": ["brazil"], "CA": ["canada"], "AU": ["australia"],
 "IN": ["india"], "CN": ["china", "chinese"], "RU": ["russia"], "MX": ["mexico"],
 "NL": ["netherlands", "dutch"], "BE": ["belgium"], "CH": ["switzerland", "swiss"],
 "AT": ["austria"], "GR": ["greece"], "PT": ["portugal"], "IE": ["ireland"],
 "NP": ["nepal"], "AR": ["argentina"], "JM": ["jamaica"], "HK": ["hong kong"],
 "AE": ["united arab emirates", "dubai"], "DE": ["germany"], "NP": ["nepal"],
}


def search_qid(name, region=""):
    d = _get({"action": "wbsearchentities", "search": name, "language": "en",
              "format": "json", "limit": 5})
    hints = []
    if region:
        hints = [region.lower()] + COUNTRY_HINTS.get(str(region).upper(), [])
    for m in d.get("search", []) or []:
        if (m.get("label") or "").strip().lower() != name.strip().lower():
            continue
        if hints:
            desc = (m.get("description") or "").lower()
            if not any(h in desc for h in hints):
                continue
        return m.get("id", ""), m.get("description", "")
    return "", ""


def fetch_claims(qids):
    """{qid: [(parent_qid, rel)]} for P31/P279."""
    out = {}
    for i in range(0, len(qids), 50):
        batch = [q for q in qids[i:i + 50] if q]
        if not batch:
            continue
        d = _get({"action": "wbgetentities", "ids": "|".join(batch),
                  "props": "claims|labels", "languages": "en", "format": "json"})
        for qid, ent in (d.get("entities") or {}).items():
            pairs = []
            for rel in ("P31", "P279"):
                for cl in ((ent.get("claims") or {}).get(rel) or []):
                    try:
                        pid = cl["mainsnak"]["datavalue"]["value"]["id"]
                        pairs.append((pid, rel))
                    except Exception:
                        continue
            out[qid] = pairs
    return out


def label_qids(qids):
    """{qid: english label}."""
    out = {}
    for i in range(0, len(qids), 50):
        batch = [q for q in qids[i:i + 50] if q]
        if not batch:
            continue
        d = _get({"action": "wbgetentities", "ids": "|".join(batch),
                  "props": "labels", "languages": "en", "format": "json"})
        for qid, ent in (d.get("entities") or {}).items():
            try:
                out[qid] = ent["labels"]["en"]["value"]
            except Exception:
                continue
    return out


def stage_known(conn):
    """Exact QIDs from ingest_provenance (source=wikidata): zero search risk."""
    rows = conn.execute("SELECT e.canonical_id, p.source_id FROM entities e"
                        " JOIN ingest_provenance p ON p.canonical_id=e.canonical_id"
                        " WHERE p.source='wikidata' AND p.source_id LIKE 'Q%'").fetchall()
    pairs = {}
    for cid, qid in rows:
        qid = (qid or "").strip()
        if qid.startswith("Q"):
            pairs[cid] = qid
    n_edge = 0
    qids = list(set(pairs.values()))
    claims = fetch_claims(qids)
    need = set()
    for plist in claims.values():
        need.update(p for p, _ in plist)
    labels = label_qids(sorted(need))
    for cid, qid in pairs.items():
        for pid, rel in claims.get(qid, []):
            pl = labels.get(pid, "")
            if not pl:
                continue
            try:
                conn.execute("INSERT OR IGNORE INTO hypernym_edges"
                             " (child_cid, parent_label, parent_qid, rel, source)"
                             " VALUES (?,?,?,?,?)", (cid, pl, pid, rel, "wikidata"))
                n_edge += 1
            except Exception:
                continue
    conn.commit()
    return len(pairs), n_edge


def stage2(conn, limit=0, offset=0, sample_where=""):
    """Resolve QIDs (exact match) + fetch hypernyms for sampled entities."""
    q = ("SELECT canonical_id, display_name, region_code FROM entities WHERE emb IS NOT NULL"
         + (f" AND {sample_where}" if sample_where else "")
         + " ORDER BY canonical_id LIMIT ? OFFSET ?")
    lim = limit if limit and limit > 0 else 10 ** 9
    rows = conn.execute(q, (lim, offset)).fetchall()
    n_res = n_edge = 0
    for cid, name, region in rows:
        try:
            has = conn.execute("SELECT COUNT(*) FROM hypernym_edges WHERE child_cid=?"
                               " AND source='wikidata'").fetchone()[0]
            if has:
                continue
            qid, _ = search_qid(name or "", region or "")
            if not qid:
                continue
            n_res += 1
            claims = fetch_claims([qid]).get(qid, [])
            if not claims:
                continue
            pids = list({p for p, _ in claims})
            labels = label_qids(pids)
            for pid, rel in claims:
                pl = labels.get(pid, "")
                if not pl:
                    continue
                try:
                    conn.execute("INSERT OR IGNORE INTO hypernym_edges"
                                 " (child_cid, parent_label, parent_qid, rel, source)"
                                 " VALUES (?,?,?,?,?)", (cid, pl, pid, rel, "wikidata"))
                    n_edge += 1
                except Exception:
                    continue
            conn.commit()
        except Exception:
            continue
    conn.commit()
    return n_res, n_edge


def export_jsonl(conn, path):
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in conn.execute("SELECT child_cid, parent_label, parent_qid, rel, source"
                              " FROM hypernym_edges"):
            fh.write(json.dumps({"child": r[0], "parent": r[1], "parent_qid": r[2],
                                 "rel": r[3], "source": r[4]}, ensure_ascii=True) + "\n")
            n += 1
    return n


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", action="store_true")
    ap.add_argument("--stage2", action="store_true")
    ap.add_argument("--known", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--sample", default="")
    ap.add_argument("--export", default="")
    ap.add_argument("--log", default="")
    args = ap.parse_args()
    logfh = open(args.log, "a", encoding="utf-8") if args.log else None

    def say(m):
        line = f"[mine_hyper] {m}"
        print(line, flush=True)
        if logfh:
            logfh.write(line + "\n")
            logfh.flush()

    from core import db as _db
    conn = _db.db_connect("mapping")
    ensure_table(conn)
    if args.stage1 or not (args.stage2 or args.known):
        n = stage1(conn)
        say(f"stage1 type-derived pairs: {n}")
    if args.known:
        nk, ne = stage_known(conn)
        say(f"known-qid entities={nk} wikidata edges={ne}")
    if args.stage2:
        nr, ne = stage2(conn, limit=args.limit, offset=args.offset,
                        sample_where=args.sample)
        say(f"stage2 resolved={nr} edges={ne}")
    if args.export:
        n = export_jsonl(conn, args.export)
        say(f"exported {n} pairs -> {args.export}")
    tot = conn.execute("SELECT COUNT(*) FROM hypernym_edges").fetchone()[0]
    say(f"hypernym_edges total: {tot}")
    conn.close()
    if logfh:
        logfh.close()
