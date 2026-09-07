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


def wbsearch_qid(name, timeout=30):
    """Resolve a name to a human-instance QID via wbsearchentities.

    Returns (qid, label, description) or None. Prefers results whose
    description signals a person; caller-side label gate still applies.
    """
    import requests
    r = requests.get(
        "https://www.wikidata.org/w/api.php",
        params={"action": "wbsearchentities", "search": name, "language": "en",
                "format": "json", "limit": 10},
        headers={"User-Agent": "TheBrain-mapping/1.0 (local-first research)"},
        timeout=timeout)
    r.raise_for_status()
    for hit in (r.json().get("search") or []):
        qid = hit.get("id", "")
        if not qid.startswith("Q"):
            continue
        yield qid, hit.get("label", ""), hit.get("description", "")


BATCH3_NAMES = [
    ("rutherford", "scientist"), ("bohr", "scientist"),
    ("heisenberg", "scientist"), ("schrodinger", "scientist"),
    ("dirac", "scientist"), ("fermi", "scientist"),
    ("planck", "scientist"), ("hubble", "scientist"),
    ("mendel", "scientist"), ("koch", "scientist"),
    ("lister", "scientist"), ("fleming", "scientist"),
    ("nightingale", "scientist"), ("goodall", "scientist"),
    ("carson", "scientist"), ("mandela", "historical_figure"),
    ("joan of arc", "historical_figure"), ("cleopatra", "historical_figure"),
    ("columbus", "historical_figure"), ("magellan", "historical_figure"),
]


def populate_name_batch(conn, names=None):
    """Resolve names via search API, then run the gated seed importer.

    No memory-sourced QIDs: every ID comes from wbsearchentities and still
    passes the label gate inside populate_historical_seed.
    """
    seeds = []
    for frag, etype in (names if names is not None else BATCH3_NAMES):
        try:
            got = False
            for qid, label, desc in wbsearch_qid(frag):
                seeds.append((frag, qid, etype))
                got = True
                break
            if not got:
                print(f"  no search hit for {frag!r}")
        except Exception as e:
            print(f"  search failed for {frag!r}: {type(e).__name__}")
    return populate_historical_seed(conn, seeds=seeds)


GEONAMES_CITIES1000_URL = "https://download.geonames.org/export/dump/cities1000.zip"
# GeoNames readme columns for cities dumps.
_GEONAMES_COLS = ("geonameid name asciiname alternatenames latitude longitude "
                  "feature_class feature_code country_code cc2 admin1 admin2 admin3 "
                  "admin4 population elevation dem timezone modification").split()


def _geonames_region(country, admin1):
    cc = (country or "").upper()
    continent = {"US": "na", "CA": "na", "MX": "na", "GL": "na"}.get(cc)
    if continent is None:
        continent = {"GB": "eu", "FR": "eu", "DE": "eu", "IT": "eu", "ES": "eu",
                     "NL": "eu", "BE": "eu", "CH": "eu", "AT": "eu", "IE": "eu",
                     "PT": "eu", "GR": "eu", "PL": "eu", "CZ": "eu", "SK": "eu",
                     "HU": "eu", "RO": "eu", "BG": "eu", "HR": "eu", "SI": "eu",
                     "SE": "eu", "NO": "eu", "DK": "eu", "FI": "eu", "IS": "eu",
                     "EE": "eu", "LV": "eu", "LT": "eu", "UA": "eu", "BY": "eu",
                     "MD": "eu", "RS": "eu", "BA": "eu", "AL": "eu", "MK": "eu",
                     "ME": "eu", "LU": "eu", "MC": "eu", "AD": "eu", "SM": "eu",
                     "VA": "eu", "MT": "eu", "CY": "eu", "TR": "eu", "RU": "eu"}.get(cc, "other")
    return continent, cc


def populate_geonames_cities(conn, cache_dir=None, min_population=5000, limit=None):
    """Bulk-load GeoNames cities1000 dump: city/town entities + aliases.

    Static open-data dump (no API, no rate limits, fully local after first
    download). Idempotent upserts keyed on geonames IDs.
    """
    import os as _os
    import zipfile as _zf
    import requests as _rq
    from pathlib import Path as _P
    cache = _P(cache_dir or _P(__file__).resolve().parent.parent / "data" / "geonames")
    cache.mkdir(parents=True, exist_ok=True)
    zpath = cache / "cities1000.zip"
    if not zpath.exists():
        print(f"downloading {GEONAMES_CITIES1000_URL} ...")
        with _rq.get(GEONAMES_CITIES1000_URL, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(zpath, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
        print("download complete.")
    n = 0
    with _zf.ZipFile(zpath) as z:
        name = [i.filename for i in z.infolist() if i.filename.endswith(".txt")][0]
        with z.open(name) as f:
            for line in f:
                try:
                    parts = line.decode("utf-8").rstrip("\n").split("\t")
                    if len(parts) < 19:
                        continue
                    rec = dict(zip(_GEONAMES_COLS, parts))
                    pop = int(rec["population"] or 0)
                    if pop < int(min_population):
                        continue
                    gid = rec["geonameid"]
                    cname = rec["name"] or rec["asciiname"]
                    if not cname:
                        continue
                    continent, cc = _geonames_region(rec["country_code"], rec["admin1"])
                    fclass = (rec["feature_class"] or "").upper()
                    etype = "city" if fclass == "P" else "town" if fclass == "T" else "location"
                    cid = f"geo:{cname.lower().replace(' ', '-')}-{gid}"
                    desc = f"{etype.title()} in {cc}"
                    if rec["admin1"]:
                        desc += f" ({rec['admin1']})"
                    desc += f"; population {pop}."
                    upsert_entity(conn, cid, etype, "geo", cname, f"geo:{continent}",
                                  region_code=cc, description=desc,
                                  popularity=float(pop),
                                  external_ids={"geonames": gid}, source="geonames")
                    add_alias(conn, cname, cid, alias_type="primary")
                    if rec["asciiname"] and rec["asciiname"] != cname:
                        add_alias(conn, rec["asciiname"], cid, alias_type="aka")
                    record_provenance(conn, cid, "geonames", gid)
                    n += 1
                    if limit and n >= int(limit):
                        break
                    if n % 20000 == 0:
                        conn.commit()
                        print(f"  ...{n} cities")
                except Exception:
                    continue
    conn.commit()
    return n


NAMESDB_PATH = r"A:\scripts\abs_auto_match\names.db"
# Exact-norm matches that are format categories, not real series.
NON_SERIES = {"ebooks", "audiobooks", "audio books", "audio books short stories",
              "short stories", "comics", "manga", "magazines", "newspapers",
              "books", "novels", "anthologies", "collections", "misc", "unknown"}


def _slug(s):
    import re as _re
    return _re.sub(r"[^a-z0-9]+", "-", str(s or "").lower()).strip("-")


def populate_namesdb(conn, path=None):
    """Local kickoff import: authors/series/titles from names.db.

    Read-only against the source. Authors -> person/author rows, series ->
    work/series rows, titles -> work/book rows; norm doubles as alias.
    """
    import sqlite3 as _sq3
    src = _sq3.connect(path or NAMESDB_PATH)
    src.row_factory = _sq3.Row
    counts = {"author": 0, "series": 0, "book": 0}
    try:
        for r in src.execute("SELECT name, norm, source, count FROM authors"):
            name = (r["name"] or "").strip()
            if not name:
                continue
            cid = f"per:{_slug(name) or 'unknown'}"
            upsert_entity(conn, cid, "author", "person", name, "person:global",
                          description=f"Author ({r['source'] or 'local'}).",
                          popularity=float(r["count"] or 1),
                          external_ids={"namesdb": r["norm"]}, source="namesdb")
            add_alias(conn, name, cid, alias_type="primary")
            if r["norm"] and r["norm"] != name.lower():
                add_alias(conn, r["norm"], cid, alias_type="aka")
            record_provenance(conn, cid, "namesdb", f"authors:{r['norm']}")
            counts["author"] += 1
        for r in src.execute("SELECT name, norm, source, count FROM series"):
            name = (r["name"] or "").strip()
            norm = (r["norm"] or "").strip().lower()
            if not name or norm in NON_SERIES:
                continue
            cid = f"work:series-{_slug(name)}"
            upsert_entity(conn, cid, "series", "work", name, "work:global",
                          description=f"Book series ({r['source'] or 'local'}).",
                          popularity=float(r["count"] or 1),
                          external_ids={"namesdb": norm}, source="namesdb")
            add_alias(conn, name, cid, alias_type="primary")
            record_provenance(conn, cid, "namesdb", f"series:{norm}")
            counts["series"] += 1
        for r in src.execute("SELECT name, norm, source, count FROM titles"):
            name = (r["name"] or "").strip()
            if not name:
                continue
            cid = f"work:book-{_slug(name)}"
            upsert_entity(conn, cid, "book", "work", name, "work:global",
                          description=f"Book title ({r['source'] or 'local'}).",
                          popularity=float(r["count"] or 1),
                          external_ids={"namesdb": r["norm"]}, source="namesdb")
            add_alias(conn, name, cid, alias_type="primary")
            record_provenance(conn, cid, "namesdb", f"titles:{r['norm']}")
            counts["book"] += 1
    finally:
        src.close()
    conn.commit()
    return counts


if __name__ == "__main__":
    import sys as _sys
    from scripts.init_mapping_db import init_mapping_db
    _conn = init_mapping_db()
    if len(_sys.argv) > 1 and _sys.argv[1] == "geonames":
        _lim = int(_sys.argv[2]) if len(_sys.argv) > 2 else 0
        print("populating GeoNames cities (static dump, idempotent)...")
        _n = populate_geonames_cities(_conn, limit=_lim or None)
        print(f"done: {_n} cities.")
    if len(_sys.argv) > 1 and _sys.argv[1] == "namesdb":
        print("importing names.db kickoff (read-only source)...")
        _c = populate_namesdb(_conn)
        print(f"done: {_c}")
    elif len(_sys.argv) > 1 and _sys.argv[1] == "batch3":
        print("populating name batch via wbsearchentities (label-gated)...")
        _kept, _skipped = populate_name_batch(_conn)
        print(f"done: {_kept} kept, {len(_skipped)} skipped.")
    else:
        print("populating curated historical seed (label-gated)...")
        _kept, _skipped = populate_historical_seed(_conn)
        print(f"done: {_kept} kept, {len(_skipped)} skipped.")
    _conn.close()
