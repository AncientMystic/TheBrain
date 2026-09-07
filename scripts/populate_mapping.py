"""
Mapping-database population framework + sources (P3).

Order per plan: public APIs first (correctness baseline), then local
names.db kickoff, then Library-folder mining. Every write is idempotent
(upsert by canonical_id / external_ids, provenance rows keyed); reruns
never duplicate. Embeddings stay NULL until the embed pass (schema
allows lazy emb) — do NOT block population on the LLM backend.
"""
import json
import time
import sqlite3

WD_SPARQL = "https://query.wikidata.org/sparql"


def _now():
    return int(time.time())


def upsert_entity(conn, canonical_id, entity_type, type_family, display_name,
                  shard_key, region_code=None, description="", popularity=0,
                  external_ids=None, source="", emb=None):
    conn.execute(
        """INSERT INTO entities
           (canonical_id, entity_type, type_family, display_name, shard_key,
            region_code, description, popularity, external_ids, updated_at, source, emb)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(canonical_id) DO UPDATE SET
             display_name=excluded.display_name,
             description=CASE WHEN excluded.description != '' THEN excluded.description ELSE entities.description END,
             popularity=max(entities.popularity, excluded.popularity),
             external_ids=excluded.external_ids,
             updated_at=excluded.updated_at,
             source=excluded.source,
             emb=COALESCE(excluded.emb, entities.emb)""",
        (canonical_id, entity_type, type_family, display_name, shard_key,
         region_code, description or "", float(popularity or 0),
         json.dumps(external_ids or {}), _now(), source,
         sqlite3.Binary(bytes(emb)) if emb is not None else None),
    )


def add_alias(conn, alias_norm, canonical_id, lang="en", alias_type="aka"):
    norm = str(alias_norm or "").strip().lower()
    if not norm or not canonical_id:
        return
    conn.execute("INSERT OR IGNORE INTO aliases (alias_norm, canonical_id, lang, alias_type) VALUES (?,?,?,?)",
                 (norm, canonical_id, lang, alias_type))
    try:
        conn.execute("INSERT OR IGNORE INTO aliases_fts (alias_norm, canonical_id) VALUES (?,?)",
                     (norm, canonical_id))
    except Exception:
        pass


def add_relation(conn, src_id, dst_id, rel):
    if not src_id or not dst_id or not rel:
        return
    conn.execute("INSERT OR IGNORE INTO entity_relations (src_id, dst_id, rel) VALUES (?,?,?)",
                 (src_id, dst_id, rel))


def record_provenance(conn, canonical_id, source, source_id):
    conn.execute("INSERT OR IGNORE INTO ingest_provenance (canonical_id, source, source_id, fetched_at) VALUES (?,?,?,?)",
                 (canonical_id, source, source_id, _now()))


def fetch_wikidata_sparql(query, timeout=120):
    """Run a SPARQL SELECT against Wikidata, return bindings list.

    NOTE: instance-wide human scans time out on WDQS (504s observed) — prefer
    fetch_wikidata_entity() below with curated QIDs for population work.
    """
    import requests
    r = requests.get(WD_SPARQL, params={"query": query, "format": "json"},
                     headers={"User-Agent": "TheBrain-mapping/1.0 (local-first research)"},
                     timeout=timeout)
    r.raise_for_status()
    return r.json().get("results", {}).get("bindings", [])


def fetch_wikidata_entity(qid, timeout=60):
    """Fetch one entity via Special:EntityData (fast, cached). Returns dict or None."""
    import requests
    qid = str(qid).strip().upper()
    r = requests.get(f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json",
                     headers={"User-Agent": "TheBrain-mapping/1.0 (local-first research)"},
                     timeout=timeout)
    r.raise_for_status()
    return (r.json().get("entities") or {}).get(qid)


# Curated seed: (expected surname fragment, QID, entity_type). The gate below
# verifies the fetched English label contains the fragment — a wrong QID from
# memory can never poison the database, it only gets skipped with a warning.
HISTORICAL_SEED = [
    ("einstein", "Q937", "scientist"), ("newton", "Q935", "scientist"),
    ("curie", "Q7186", "scientist"), ("darwin", "Q1035", "scientist"),
    ("galileo", "Q307", "scientist"), ("tesla", "Q9036", "scientist"),
    ("edison", "Q8743", "scientist"), ("turing", "Q7251", "scientist"),
    ("lovelace", "Q7259", "scientist"), ("washington", "Q23", "historical_figure"),
    ("lincoln", "Q91", "historical_figure"), ("shakespeare", "Q692", "celebrity"),
    ("mozart", "Q254", "celebrity"), ("vinci", "Q762", "celebrity"),
    ("vangogh", "Q5582", "celebrity"), ("gogh", "Q5582", "celebrity"),
    ("aristotle", "Q868", "historical_figure"), ("plato", "Q859", "historical_figure"),
    ("socrates", "Q913", "historical_figure"),
    # Batch 2: leaders, artists, composers, scientists (gate verifies each).
    ("churchill", "Q8016", "historical_figure"), ("gandhi", "Q1001", "historical_figure"),
    ("napoleon", "Q517", "historical_figure"), ("caesar", "Q44", "historical_figure"),
    ("alexander", "Q8409", "historical_figure"), ("picasso", "Q5593", "celebrity"),
    ("michelangelo", "Q5592", "celebrity"), ("rembrandt", "Q5593", "celebrity"),
    ("bach", "Q1339", "celebrity"), ("beethoven", "Q255", "celebrity"),
    ("mendel", "Q37970", "scientist"), ("pasteur", "Q82122", "scientist"),
    ("faraday", "Q8758", "scientist"), ("maxwell", "Q9095", "scientist"),
    ("hawking", "Q496", "scientist"), ("confucius", "Q4604", "historical_figure"),
    ("jesus", "Q302", "historical_figure"), ("joan", "Q23182", "historical_figure"),
    ("cleopatra", "Q635", "historical_figure"), ("columbus", "Q7327", "historical_figure"),
    ("magellan", "Q1496", "historical_figure"), ("cook", "Q7327", "historical_figure"),
    ("marco polo", "Q6108", "historical_figure"), ("nightingale", "Q35610", "scientist"),
    ("salk", "Q188845", "scientist"), ("fleming", "Q35244", "scientist"),
    ("bohr", "Q5292", "scientist"), ("heisenberg", "Q40916", "scientist"),
    ("schrodinger", "Q42450", "scientist"), ("dirac", "Q47480", "scientist"),
]


def populate_historical_seed(conn, seeds=None):
    """Fetch curated QIDs with label-gate verification. Returns (kept, skipped)."""
    kept, skipped = 0, []
    for frag, qid, etype in (seeds if seeds is not None else HISTORICAL_SEED):
        try:
            ent = fetch_wikidata_entity(qid)
            if not ent:
                skipped.append((qid, "fetch-empty"))
                continue
            _labels = ent.get("labels") or {}
            label = (_labels.get("en") or _labels.get("mul") or {}).get("value", "")
            desc = ((ent.get("descriptions") or {}).get("en") or {}).get("value", "")
            _aliases = [a.get("value", "") for a in ((ent.get("aliases") or {}).get("en") or [])]
            _pool = " | ".join([label, desc] + _aliases).lower().replace(" ", "")
            if frag.lower().replace(" ", "") not in _pool:
                skipped.append((qid, f"label-mismatch:{label!r}"))
                continue
            if not label:
                label = _aliases[0] if _aliases else qid
            claims = ent.get("claims") or {}
            birth = death = ""
            try:
                birth = claims["P569"][0]["mainsnak"]["datavalue"]["value"]["time"][1:5]
                death = claims["P570"][0]["mainsnak"]["datavalue"]["value"]["time"][1:5]
            except Exception:
                pass
            when = birth if birth.isdigit() else ""
            if when and death.isdigit():
                when += "-" + death
            full_desc = (desc + (f" ({when})" if when and when not in desc else "")).strip()
            links = int(ent.get("sitelinks", {}).__len__()) if isinstance(ent.get("sitelinks"), dict) else 0
            cid = _canon_person(qid, label)
            upsert_entity(conn, cid, etype, "person", label, "person:hist",
                          description=full_desc, popularity=float(links),
                          external_ids={"wikidata": qid.upper()}, source="wikidata")
            add_alias(conn, label, cid, alias_type="primary")
            record_provenance(conn, cid, "wikidata", qid.upper())
            kept += 1
        except Exception as e:
            skipped.append((qid, f"error:{type(e).__name__}"))
            continue
    conn.commit()
    return kept, skipped


def _canon_person(qid, label):
    import re as _re
    slug = _re.sub(r"[^a-z0-9]+", "-", str(label or qid).lower()).strip("-") or qid.lower()
    return f"per:{slug}-{qid.lower()}"


HISTORICAL_FIGURES_SPARQL = """
SELECT ?person ?personLabel ?desc ?birth ?death ?n WHERE {
  ?person wdt:P31 wd:Q5;
          wikibase:sitelinks ?n.
  FILTER(?n >= %(min_links)d)
  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "en".
    ?person rdfs:label ?personLabel.
    ?person schema:description ?desc.
  }
  OPTIONAL { ?person wdt:P569 ?birth. }
  OPTIONAL { ?person wdt:P570 ?death. }
}
ORDER BY DESC(?n)
LIMIT %(limit)d
"""


def populate_historical_figures(conn, limit=500, min_links=80):
    """Wikidata humans by sitelink fame: scientists, figures, celebrities.

    Bounded + ordered: most-referenced first, so small limits still yield
    the highest-value rows. Description carries who/when for AI context.
    """
    rows = fetch_wikidata_sparql(HISTORICAL_FIGURES_SPARQL % {"min_links": int(min_links), "limit": int(limit)})
    n = 0
    for b in rows:
        try:
            uri = b["person"]["value"]
            qid = uri.rsplit("/", 1)[-1]
            label = b.get("personLabel", {}).get("value", qid)
            desc = b.get("desc", {}).get("value", "")
            links = int(b.get("n", {}).get("value", 0))
            birth = (b.get("birth", {}).get("value", "") or "")[:4]
            death = (b.get("death", {}).get("value", "") or "")[:4]
            when = ""
            if birth.isdigit():
                when = birth + ("-" + death if death.isdigit() else "-")
            full_desc = (desc + (f" ({when})" if when and when not in desc else "")).strip()
            cid = _canon_person(qid, label)
            upsert_entity(conn, cid, "historical_figure", "person", label, "person:hist",
                          description=full_desc, popularity=float(links),
                          external_ids={"wikidata": qid}, source="wikidata")
            add_alias(conn, label, cid, alias_type="primary")
            record_provenance(conn, cid, "wikidata", qid)
            n += 1
        except Exception:
            continue
    conn.commit()
    return n


if __name__ == "__main__":
    from scripts.init_mapping_db import init_mapping_db
    _conn = init_mapping_db()
    print("populating curated historical seed (label-gated)...")
    _kept, _skipped = populate_historical_seed(_conn)
    print(f"done: {_kept} kept, {len(_skipped)} skipped: {_skipped}")
    _conn.close()
