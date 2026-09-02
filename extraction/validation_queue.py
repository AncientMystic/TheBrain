
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

    def put(self, item, timeout=1.0):
        """Add an item to the validation list."""
        if item is None:
            return
        self._next_id = getattr(self, '_next_id', 0) + 1
        item['_item_id'] = self._next_id
        with self.lock:
            if item['_item_id'] in self.seen_ids:
                return
            self.seen_ids.add(item['_item_id'])
            self.items.append(item)
            self.total_items += 1

    def start(self):
        """No-op for API compatibility."""
        pass

    def _process_batch(self, batch):
        """Send a batch to LLM for validation and store results."""
        if not batch:
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
            # Fallback: keep original items
            with self.lock:
                self.results.extend(batch)
            self._update_progress(len(batch))
            return

        validated = resp.get("validated_items", [])
        if not validated:
            with self.lock:
                self.results.extend(batch)
            self._update_progress(len(batch))
            return

        # Merge metadata back
        merged = []
        for i, item in enumerate(validated):
            if not isinstance(item, dict):
                item = {}
            if i < len(batch) and isinstance(batch[i], dict):
                item["_chunk_idx"] = batch[i].get("_chunk_idx")
                item["_category"] = batch[i].get("_category")
                item["_item_id"] = batch[i].get("_item_id")
            merged.append(item)
        if len(validated) < len(batch):
            merged.extend(batch[len(validated):])

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
        # Process in batches using thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = []
            for i in range(0, len(self.items), self.batch_size):
                batch = self.items[i:i+self.batch_size]
                futures.append(executor.submit(self._process_batch, batch))
            # Wait for all to finish
            concurrent.futures.wait(futures)
        print()  # newline after progress
        return self.results
