
"""
Async validation queue for extracted items.
Uses ThreadPoolExecutor for parallel batch validation.
Reliable shutdown and progress.
"""
import concurrent.futures
import queue
import threading
import time
import traceback
from typing import List

import config
from core.llm import call_model_json
import logging
logger = logging.getLogger(__name__)


class ValidationQueue:
    """Thread-safe queue that sends items to large model for validation."""

    def __init__(self, workers=None, batch_size=None):
        self.workers = workers or config.VALIDATION_WORKERS
        self.batch_size = batch_size or config.VALIDATION_BATCH_SIZE
        self.items = []
        self.results = []
        self.lock = threading.Lock()
        self.total_items = 0
        self.processed_count = 0
        self.seen_ids = set()

    def _content_hash(self, item):
        import hashlib
        import json
        try:
            key_parts = [
                str(item.get("_category", "")),
                str(item.get("fact_text", item.get("entity_name", item.get("person_name", item.get("location_name", ""))))),
                str(item.get("canonical_value", "")),
                str(item.get("source_span", "")),
            ]
            return hashlib.sha256("\0".join(key_parts).encode("utf-8", errors="ignore")).hexdigest()
        except Exception:
            return None

    def put(self, item, timeout=1.0):
        """Add an item to the validation list (copy-on-put, content-hash dedup)."""
        if item is None or not isinstance(item, dict):
            return
        item_copy = dict(item)
        chash = self._content_hash(item_copy)
        with self.lock:
            self._next_id = getattr(self, '_next_id', 0) + 1
            item_copy['_item_id'] = self._next_id
            if chash is not None:
                if chash in self.seen_ids:
                    return
                self.seen_ids.add(chash)
            self.items.append(item_copy)
            self.total_items += 1

    def start(self):
        """No-op for API compatibility."""
        pass

    def _process_batch(self, batch):
        """Send a batch to LLM for validation and store results.
           Splits large batches into smaller sub-batches to avoid empty responses."""
        if not batch:
            return
        max_per_call = int(getattr(config, "VALIDATION_MAX_PER_CALL", 4))
        if len(batch) > max_per_call:
            # Recursively process smaller chunks
            for i in range(0, len(batch), max_per_call):
                sub_batch = batch[i:i+max_per_call]
                self._process_batch(sub_batch)
            return
        # Build prompt
        import json
        items_with_excerpt = []
        for item in batch:
            item_copy = dict(item)
            chunk_idx = item_copy.get('_chunk_idx')
            if chunk_idx is not None:
                try:
                    from core import db
                    conn = db.db_connect("index")
                    cur = conn.cursor()
                    cur.execute("SELECT chunk_text FROM document_chunks WHERE chunk_id=?", (chunk_idx,))
                    row = cur.fetchone()
                    conn.close()
                    if row:
                        item_copy['_source_excerpt'] = row['chunk_text'][:500]
                except Exception:
                    logger.warning("Unexpected exception occurred", exc_info=True)
                    pass
            items_with_excerpt.append(item_copy)

        prompt = f"""You are a meticulous validation agent.
Given the following extracted items and their source excerpts, verify each item.
Correct any inaccuracies, remove unsupported claims, and assign a final confidence (0-1).
Return JSON with key "validated_items" as a list of objects (same structure as input).
Do not include the '_source_excerpt' key in the output.

Items:
{json.dumps(items_with_excerpt, indent=2, default=str)}

Return only JSON.
"""
        resp = None
        try:
            resp = call_model_json(
                prompt,
                max_tokens=2048,
                endpoint_type=config.VALIDATION_MODEL_GROUP,
                timeout=config.VALIDATION_TIMEOUT,
            )
        except Exception as e:
            if config.DEBUG_VERBOSE:
                print(f"Validation batch error: {e}")

        if resp is None or not isinstance(resp, dict):
            # Fallback: preserve verification-first — mark unverified, don't pass as validated
            fallback = []
            for b in batch:
                fb = dict(b) if isinstance(b, dict) else {}
                fb["verification_status"] = "unverified"
                fb["confidence_final"] = 0.0
                fallback.append(fb)
            with self.lock:
                self.results.extend(fallback)
            self._update_progress(len(batch))
            return

        validated = resp.get("validated_items", [])
        if not validated:
            fallback = []
            for b in batch:
                fb = dict(b) if isinstance(b, dict) else {}
                fb["verification_status"] = "unverified"
                fb["confidence_final"] = 0.0
                fallback.append(fb)
            with self.lock:
                self.results.extend(fallback)
            self._update_progress(len(batch))
            return

        # Merge metadata back by _item_id (not positionally, LLM may reorder/drop)
        by_id = {b.get("_item_id"): b for b in batch if isinstance(b, dict)}
        merged = []
        seen = set()
        for item in validated:
            if not isinstance(item, dict):
                continue
            iid = item.get("_item_id")
            src = by_id.get(iid) if iid is not None else None
            if src is None and len(by_id) == 1:
                src = next(iter(by_id.values()))
            if src is not None:
                item.setdefault("_chunk_idx", src.get("_chunk_idx"))
                item.setdefault("_category", src.get("_category"))
                item["_item_id"] = src.get("_item_id")
                seen.add(src.get("_item_id"))
            merged.append(item)
        # Any batch items missing from LLM output → keep as unverified, don't silently drop
        for b in batch:
            if b.get("_item_id") not in seen:
                fb = dict(b)
                fb["verification_status"] = "unverified"
                fb["confidence_final"] = 0.0
                merged.append(fb)

        # Clean and store
        from extraction.cleaners import (
            _clean_facts, _clean_entities, _clean_people, _clean_locations,
            _clean_dates, _clean_events, _clean_discoveries, _clean_gems
        )
        clean_map = {
            'facts': _clean_facts,
            'entities': _clean_entities,
            'people': _clean_people,
            'locations': _clean_locations,
            'dates': _clean_dates,
            'events': _clean_events,
            'discoveries': _clean_discoveries,
            'gems': _clean_gems,
        }
        from core.schema_validation import validate_and_coerce
        final_items = []
        for item in merged:
            if not isinstance(item, dict):
                continue
            cat = item.get('_category')
            if cat in clean_map:
                cleaned_list = clean_map[cat]([item])
                if cleaned_list:
                    cleaned_item = cleaned_list[0]
                    v = validate_and_coerce(cat, cleaned_item)
                    if v is not None:
                        for key in ('_item_id', '_chunk_idx', '_category'):
                            v.pop(key, None)
                        final_items.append(v)
            else:
                for key in ('_item_id', '_chunk_idx', '_category'):
                    item.pop(key, None)
                final_items.append(item)
        with self.lock:
            self.results.extend(final_items)
        self._update_progress(len(batch))

    def _update_progress(self, count):
        with self.lock:
            self.processed_count += count
            if self.total_items > 0:
                print(f"\r    Validation queue progress: {self.processed_count}/{self.total_items} items processed", end="", flush=True)

    def wait_and_get_results(self):
        if not self.items:
            print()
            return []
        # Respect LLM endpoint capacities (don't exhaust first endpoint cap 3 with 8 workers)
        try:
            caps = getattr(config, "LLM_ENDPOINT_CAPACITIES", []) or []
            total_cap = sum(int(c) for c in caps) if caps else self.workers
        except Exception:
            total_cap = self.workers
        effective_workers = max(1, min(self.workers, total_cap, len(self.items)))
        # Process in batches using thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = []
            for i in range(0, len(self.items), self.batch_size):
                batch = self.items[i:i+self.batch_size]
                futures.append(executor.submit(self._process_batch, batch))
            # Wait for all to finish
            concurrent.futures.wait(futures)
        print()  # newline after progress
        return self.results
