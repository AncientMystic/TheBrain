import json
import sqlite3
from pathlib import Path

import numpy as np

import config
from core import db
from core.file_utils import get_file_hash
from extractors.registry import extract_text_from_file
from ingestion.chunker import chunk_document
from core.llm import call_model_json
from core.embeddings import get_embedding


LOGIC_EXTRACTION_PROMPT = """
You are a knowledge extraction agent specialized in identifying reusable logic modules, protocols, reasoning patterns, skills, strategies, and task formats from a document.

Read the following document excerpt and extract any distinct logic modules. For each module, provide:
- name: short unique name
- category: one of [reasoning, protocol, strategy, skill, task_format, management, identification, sentiment, relationship, agent, instruction]
- summary: 2-3 sentence summary of the module
- keywords: list of important keywords
- content: the full text of the module (instructions, steps, examples, etc.)

Return JSON with key "modules" containing list of module objects.

Excerpt:
\"\"\"
{chunk_text}
\"\"\"
"""


def learn_logic_from_file(filepath: Path):
    file_hash = get_file_hash(filepath)
    # Check if already processed for logic
    conn_idx = db.db_connect("index")
    cur_idx = conn_idx.cursor()
    cur_idx.execute("SELECT 1 FROM processing_progress WHERE file_hash=? AND status='processed_logic'", (file_hash,))
    if cur_idx.fetchone():
        conn_idx.close()
        return []
    conn_idx.close()

    result = extract_text_from_file(filepath)
    text = result["text"]
    if not text:
        return []

    chunks = chunk_document(text)
    modules = []
    for chunk in chunks[:10]:  # limit to first 10 chunks for initial version
        prompt = LOGIC_EXTRACTION_PROMPT.format(chunk_text=chunk)
        data = call_model_json(prompt, max_tokens=2048)
        if data and "modules" in data:
            modules.extend(data["modules"])

    conn = db.db_connect("logic")
    cur = conn.cursor()
    stored_ids = []
    for mod in modules:
        name = mod.get("name", "")
        category = mod.get("category", "other")
        summary = mod.get("summary", "")
        keywords = json.dumps(mod.get("keywords", []))
        content = mod.get("content", "")
        emb = get_embedding(name + " " + summary)
        blob = sqlite3.Binary(np.array(emb, dtype=np.float32).tobytes()) if emb else None
        cur.execute("""
            INSERT OR REPLACE INTO logic_modules (name, category, summary, keywords, content, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (name, category, summary, keywords, content, blob))
        stored_ids.append(cur.lastrowid)
        for kw in mod.get("keywords", []):
            cur.execute("INSERT OR IGNORE INTO logic_keywords (logic_id, keyword, weight) VALUES (?, ?, 1.0)",
                        (cur.lastrowid, kw))
    conn.commit()
    conn.close()

    # Mark as processed
    conn_idx = db.db_connect("index")
    cur_idx = conn_idx.cursor()
    cur_idx.execute("INSERT OR REPLACE INTO processing_progress (file_hash, status, stage) VALUES (?, 'processed_logic', 'logic_learning')",
                    (file_hash,))
    conn_idx.commit()
    conn_idx.close()

    return stored_ids