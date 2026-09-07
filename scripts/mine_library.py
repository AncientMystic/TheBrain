"""
Library-folder mining for the mapping database (P3, local phase).

Parses book filenames (author/title/publisher/year patterns) plus PDF
document metadata — never full text extraction. Everything upserts by
the same canonical-ID schemes as populate_mapping, so library rows merge
with names.db rows instead of duplicating. Fully idempotent reruns.
"""
import re

LIBRARY_FOLDERS = [
    r"H:\Library\Optimized\50 All About History Books Collection",
    r"H:\Library\Optimized\Cambridge Histories",
    r"H:\Library\Optimized\DK publishing",
    r"H:\Library\Optimized\Project-Muse",
    r"H:\Library\Optimized\Survival",
]

_YEAR_RE = re.compile(r"\((19|20)\d{2}\)|\b(19|20)\d{2}\b")
_PUB_RE = re.compile(r"\(([^()]{2,60}?)\)")
_BY_RE = re.compile(r"\s+By\s+([A-Z][\w.'-]+(?:\s+[A-Z][\w.'-]+){0,3})\s*(?:\.pdf)?\s*$", re.IGNORECASE)
_PUB_HINT = re.compile(r"publish|press|books|media|penguin|harper|simon|schuster|random|macmillan|wiley|oxford|cambridge|\bdk\b|smithsonian|routledge|yale|princeton|harvard|mit\b|springer|elsevier|taylor|francis|sage|wiley|norton|vintage|anchor|doubleday|knopf|farrar|straus|giroux|houghton|mifflin|harcourt|little|brown|putnam|scribner|avon|bantam|dell|tor|baen|orbit|gollancz|headline|bloomsbury|fab|corgi|arrow|panel|verso|pluto|haymarket|seven stories|city lights|melville|akashic|europa|other press|grove|atlantic|harpercollins|simon & schuster", re.IGNORECASE)


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "-", str(s or "").lower()).strip("-")


def parse_book_filename(fname, default_publisher=""):
    """Return {title, author, publisher, year} from a book filename.

    Conservative: only fills fields with explicit markers (By-pattern,
    publisher-hint parentheticals, 4-digit years). Title is the remainder.
    """
    stem = re.sub(r"\.pdf\s*$", "", str(fname or ""), flags=re.IGNORECASE).strip()
    out = {"title": "", "author": "", "publisher": default_publisher, "year": ""}
    m = _BY_RE.search(stem)
    if m:
        _by = m.group(1).strip()
        if _PUB_HINT.search(_by) and not out["publisher"]:
            out["publisher"] = _by
        else:
            out["author"] = _by
        stem = (stem[:m.start()] + stem[m.end():]).strip(" -")
    for pm in _PUB_RE.finditer(stem):
        inner = pm.group(1).strip()
        if _PUB_HINT.search(inner):
            out["publisher"] = inner
            stem = (stem[:pm.start()] + stem[pm.end():]).strip(" -")
            break
    ym = _YEAR_RE.search(stem)
    if ym:
        _digits = re.sub(r"\D", "", ym.group(0))
        if len(_digits) == 4:
            out["year"] = _digits
        stem = (stem[:ym.start()] + stem[ym.end():]).strip(" -")
    # Leading "(Series)" or "Series - " crumbs and dash-prefixed junk.
    stem = re.sub(r"^\([^()]*\)\s*[-–]?\s*", "", stem).strip(" -")
    stem = re.sub(r"^-\s*", "", stem).strip()
    out["title"] = re.sub(r"\s+", " ", stem).strip(" -")
    return out


def pdf_meta_title_author(path):
    """Fast PDF metadata read only (no text extraction)."""
    try:
        from pypdf import PdfReader
        r = PdfReader(str(path))
        meta = r.metadata or {}
        return (str(meta.get("/Title") or meta.get("title") or ""),
                str(meta.get("/Author") or meta.get("author") or ""))
    except Exception:
        try:
            from PyPDF2 import PdfReader
            r = PdfReader(str(path))
            meta = r.metadata or {}
            return (str(meta.get("/Title") or ""), str(meta.get("/Author") or ""))
        except Exception:
            return "", ""


def mine_libraries(conn, folders=None, limit=None):
    """Walk library folders, upsert books/authors/publishers + relations."""
    from pathlib import Path as _P
    from scripts.populate_mapping import upsert_entity, add_alias, add_relation, record_provenance
    counts = {"book": 0, "author": 0, "publisher": 0, "relations": 0, "files": 0}
    n = 0
    for folder in (folders or LIBRARY_FOLDERS):
        root = _P(folder)
        if not root.exists():
            print(f"  skip missing: {folder}")
            continue
        for pdf in sorted(root.rglob("*.pdf")):
            if limit and n >= int(limit):
                break
            try:
                info = parse_book_filename(pdf.name)
                mt, ma = pdf_meta_title_author(pdf)
                title = info["title"] or mt or pdf.stem
                if not title or len(title) < 3:
                    continue
                author = info["author"] or ma
                tcid = f"work:book-{_slug(title)}"
                desc = "Book"
                if author:
                    desc += f" by {author}"
                if info["publisher"]:
                    desc += f" ({info['publisher']})"
                if info["year"]:
                    desc += f", {info['year']}."
                else:
                    desc += "."
                upsert_entity(conn, tcid, "book", "work", title[:200], "work:global",
                              description=desc[:500], popularity=1.0,
                              external_ids={"library": pdf.name}, source="library")
                add_alias(conn, title[:200], tcid, alias_type="primary")
                record_provenance(conn, tcid, "library", str(pdf))
                counts["book"] += 1
                if author and len(author) > 2:
                    acid = f"per:{_slug(author)}"
                    upsert_entity(conn, acid, "author", "person", author[:120], "person:global",
                                  description=f"Author ({info['publisher'] or 'library'}).",
                                  popularity=1.0, external_ids={}, source="library")
                    add_alias(conn, author[:120], acid, alias_type="primary")
                    add_relation(conn, acid, tcid, "authored")
                    record_provenance(conn, acid, "library", str(pdf))
                    counts["author"] += 1
                    counts["relations"] += 1
                if info["publisher"]:
                    pcid = f"work:publisher-{_slug(info['publisher'])}"
                    upsert_entity(conn, pcid, "publisher", "work", info["publisher"][:120],
                                  "work:global",
                                  description=f"Publisher ({folder.rsplit(chr(92), 1)[-1]}).",
                                  popularity=1.0, external_ids={}, source="library")
                    add_alias(conn, info["publisher"][:120], pcid, alias_type="primary")
                    add_relation(conn, pcid, tcid, "published_by")
                    record_provenance(conn, pcid, "library", str(pdf))
                    counts["publisher"] += 1
                    counts["relations"] += 1
                n += 1
                counts["files"] += 1
                if n % 500 == 0:
                    conn.commit()
                    print(f"  ...{n} files")
            except Exception:
                continue
            if limit and n >= int(limit):
                break
    conn.commit()
    return counts


if __name__ == "__main__":
    import sys as _sys
    from scripts.init_mapping_db import init_mapping_db
    _conn = init_mapping_db()
    _lim = int(_sys.argv[1]) if len(_sys.argv) > 1 else 0
    print("mining library folders (filename + PDF metadata only)...")
    _c = mine_libraries(_conn, limit=_lim or None)
    print(f"done: {_c}")
    _conn.close()
