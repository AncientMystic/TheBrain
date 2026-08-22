"""
Optional Recoll client wrapper.
Falls back gracefully if Recoll is not installed.
"""
import os
import config

class RecollClient:
    def __init__(self, confdir=None, extra_dbs=None):
        try:
            from recoll import recoll
        except ImportError as e:
            raise ImportError("Recoll Python API not installed. Install python3-recoll or equivalent.") from e

        confdir = confdir or config.RECOLL_CONFDIR or ""
        extra_dbs = extra_dbs or [p for p in config.RECOLL_EXTRA_DBS.split() if p]

        self.db = recoll.connect(confdir=confdir, extra_dbs=extra_dbs)

    def search(self, query_string, limit=None, sort_by=None, fetch_text=True):
        """Return (list of Recoll Doc objects, total_count)."""
        limit = limit or config.RECOLL_DEFAULT_LIMIT
        query = self.db.query()
        if sort_by:
            query.sortby(sort_by)
        count = query.execute(query_string, fetchtext=fetch_text)
        results = query.fetchmany(limit)
        return results, count

    def get_document(self, udi):
        return self.db.getDoc(udi)

    def get_abstract(self, doc, query, maxchars=200, contextwords=5):
        self.db.setAbstractParams(maxchars=maxchars, contextwords=contextwords)
        return self.db.makeDocAbstract(doc, query)

    def extract_text(self, doc):
        """Extract plain text from a Recoll result document."""
        try:
            from recoll import rclextract
            extractor = rclextract.Extractor(doc)
            result = extractor.textextract(doc.ipath)
            return result.text if result else None
        except Exception:
            return None

    def close(self):
        try:
            self.db.close()
        except Exception:
            pass
