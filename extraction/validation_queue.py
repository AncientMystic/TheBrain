"""Async validation queue for small model outputs."""
import queue
import threading
import time
import traceback

import config
from core.vector_store import ExactVectorStore
from core.llm import call_model_json


class ValidationQueue:
    """Thread-safe queue that sends items to large model for validation."""

    def __init__(self, workers=None, batch_size=None):
        self.queue = queue.Queue(maxsize=config.VALIDATION_QUEUE_SIZE)
        self.batch_size = batch_size or config.VALIDATION_BATCH_SIZE
        self.workers = workers or config.VALIDATION_WORKERS
        self._workers = []
        self.results = []
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self._next_id = 0

    def put(self, item, timeout=1.0):
        if item is None:  # sentinel for shutdown
            self.queue.put(None)
            return
        self._next_id += 1
        item['_item_id'] = self._next_id
        try:
            self.queue.put(item, timeout=timeout)
        except queue.Full:
            # Fallback: process synchronously
            print("    (Validation queue full, processing item synchronously)")
            self._process_batch([item])

    def start(self):
        for _ in range(self.workers):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self._workers.append(t)

    def _worker(self):
        while not self._stop_event.is_set():
            batch = []
            try:
                # collect a batch or until timeout
                while len(batch) < self.batch_size:
                    try:
                        item = self.queue.get(timeout=0.2)
                        if item is None:  # sentinel
                            self._stop_event.set()
                            break
                        batch.append(item)
                    except queue.Empty:
                        break
                if not batch:
                    continue

                self._process_batch(batch)
            except Exception as e:
                print(f"Validation worker error: {e}")
                traceback.print_exc()
            finally:
                # Mark all items in batch as done to avoid deadlock in join()
                for _ in batch:
                    self.queue.task_done()

    def _process_batch(self, batch):
        prompt = self._build_prompt(batch)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                call_model_json,
                prompt,
                max_tokens=2048,
                endpoint_type=config.VALIDATION_MODEL_GROUP,
            )
            try:
                resp = future.result(timeout=config.VALIDATION_TIMEOUT)
            except concurrent.futures.TimeoutError:
                print("    (Validation batch timed out; skipping validation)")
                resp = None

        if resp is None:
            # fallback: keep original items
            with self.lock:
                self.results.extend(batch)
            return

        # Expect response with key "validated_items": list of dicts
        validated = resp.get("validated_items", [])
        if not validated:
            # If no structured response, keep original
            with self.lock:
                self.results.extend(batch)
            return

        # Merge original metadata (_chunk_idx, _category, _item_id) into validated items
        # because the LLM may not preserve them.
        merged = []
        for i, item in enumerate(validated):
            if not isinstance(item, dict):
                item = {}
            # Copy from corresponding original batch item
            if i < len(batch) and isinstance(batch[i], dict):
                item["_chunk_idx"] = batch[i].get("_chunk_idx")
                item["_category"] = batch[i].get("_category")
                item["_item_id"] = batch[i].get("_item_id")
            merged.append(item)

        # If LLM returned fewer items than batch, keep remaining originals unchanged
        if len(validated) < len(batch):
            merged.extend(batch[len(validated):])

        # Re-clean each item using its category-specific cleaner before storing
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
                    # Apply strict schema validation before removing metadata
                    validated_item = validate_and_coerce(cat, cleaned_item)
                    if validated_item is not None:
                        # Remove internal metadata that should not be stored
                        for key in ('_item_id', '_chunk_idx', '_category'):
                            validated_item.pop(key, None)
                        final_items.append(validated_item)
            else:
                # Unknown category: keep original but remove metadata
                for key in ('_item_id', '_chunk_idx', '_category'):
                    item.pop(key, None)
                final_items.append(item)

        with self.lock:
            self.results.extend(final_items)

    def _build_prompt(self, batch):
        import json
        # Attempt to fetch source excerpt for context
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

        items_str = json.dumps(items_with_excerpt, indent=2, default=str)
        prompt = f"""You are a meticulous validation agent.
Given the following extracted items and their source excerpts, verify each item.
Correct any inaccuracies, remove unsupported claims, and assign a final confidence (0-1).
Return JSON with key "validated_items" as a list of objects (same structure as input).
Do not include the '_source_excerpt' key in the output.

Items:
{items_str}

Return only JSON.
"""
        return prompt

    def wait_and_get_results(self):
        """Signal workers to finish after queue is empty and return aggregated results."""
        # Wait for queue to drain
        self.queue.join()
        # Send sentinel None to each worker using raw queue.put to avoid our custom put
        for _ in self._workers:
            self.queue.put(None, timeout=1.0)  # raw put with timeout
        for t in self._workers:
            t.join(timeout=30)
        return self.results
