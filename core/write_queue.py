"""
Bulk write queue for TheBrain.
"""
import threading
import queue
import time
from collections import defaultdict
from core import db

_write_queue = queue.Queue()
_flush_event = threading.Event()
_writer_started = False
_lock = threading.Lock()

def _writer_loop():
    while True:
        items = []
        while len(items) < 500:
            try:
                items.append(_write_queue.get_nowait())
            except queue.Empty:
                break
        if not items:
            _flush_event.wait(0.5)
            _flush_event.clear()
            continue
        grouped = defaultdict(list)
        for db_type, sql, params in items:
            grouped[db_type].append((sql, params))
        for db_type, writes in grouped.items():
            conn = db.db_connect(db_type)
            try:
                conn.execute("BEGIN")
                for sql, params in writes:
                    conn.execute(sql, params)
                conn.commit()
            except Exception as e:
                conn.rollback()
                print(f"Bulk write error for {db_type}: {e}")
            finally:
                conn.close()

def start_writer():
    global _writer_started
    with _lock:
        if not _writer_started:
            t = threading.Thread(target=_writer_loop, daemon=True)
            t.start()
            _writer_started = True

def enqueue_write(db_type, sql, params):
    start_writer()
    _write_queue.put((db_type, sql, params))
    if _write_queue.qsize() > 1000:
        _flush_event.set()

def enqueue_many(db_type, sql, params_list):
    start_writer()
    for params in params_list:
        _write_queue.put((db_type, sql, params))
        if _write_queue.qsize() > 1000:
            _flush_event.set()

def flush_writes():
    _flush_event.set()
    while not _write_queue.empty():
        time.sleep(0.1)
