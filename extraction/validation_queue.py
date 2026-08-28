"""Async validation queue for small model outputs."""
import queue
import threading
import time
import traceback

import config
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

    def put(self, item):
        self.queue.put(item)

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
                resp = future.result(timeout=240)
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

        with self.lock:
            self.results.extend(validated)

    def _build_prompt(self, batch):
        import json
        items_str = json.dumps(batch, indent=2, default=str)
        prompt = f"""You are a meticulous validation agent.
Given the following extracted items and their source chunks, verify each item.
Correct any inaccuracies, remove unsupported claims, and assign a final confidence (0-1).
Return JSON with key "validated_items" as a list of objects (same structure as input).

Items:
{items_str}

Return only JSON.
"""
        return prompt

    def wait_and_get_results(self):
        """Signal workers to finish after queue is empty and return aggregated results."""
        # Wait for queue to drain
        self.queue.join()
        # Send sentinel None to each worker to stop
        for _ in self._workers:
            self.queue.put(None)
        for t in self._workers:
            t.join(timeout=30)
        return self.results
