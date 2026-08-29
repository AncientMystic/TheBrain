"""
Active learning with Thompson sampling for knowledge gap filling.
"""
import random
import math
from typing import List, Dict
from core import db
import config
from core.hyperbolic import hyperbolic_distance
from core.embeddings import get_embedding
from core.hyperbolic import exp_map


class ActiveLearner:
    """Selects knowledge gaps to fill using Thompson sampling."""
    def __init__(self, gaps: List[Dict]):
        self.gaps = gaps
        self.alpha = {g.get("id", i): 1.0 for i, g in enumerate(gaps)}
        self.beta = {g.get("id", i): 1.0 for i, g in enumerate(gaps)}

    def select_gap(self):
        """Sample from Beta distribution for each gap and choose highest."""
        sampled = {}
        for gap in self.gaps:
            gid = gap.get("id", id(gap))
            a = self.alpha.get(gid, 1.0)
            b = self.beta.get(gid, 1.0)
            sampled[gid] = random.betavariate(a, b)
        best_gid = max(sampled, key=sampled.get)
        for gap in self.gaps:
            if gap.get("id", id(gap)) == best_gid:
                return gap
        return None

    def update(self, gap_id, success: bool):
        """Update Beta parameters based on outcome."""
        if success:
            self.alpha[gap_id] += 1
        else:
            self.beta[gap_id] += 1

    def generate_recoll_query(self, gap: Dict) -> str:
        """Generate a Recoll query from gap entity."""
        entity = gap.get("entity", "")
        gap_type = gap.get("type", "")
        if gap_type == "missing_relationships":
            return f"{entity} related"
        elif gap_type == "low_confidence":
            return entity
        elif gap_type == "sparse_entity":
            return entity
        else:
            return entity

    def select_gap_hyperbolic(self, query_embedding=None):
        """Select gap using hyperbolic uncertainty (max distance from query)."""
        if query_embedding is None:
            return self.select_gap()
        distances = []
        for gap in self.gaps:
            entity = gap.get("entity", "")
            emb = get_embedding(entity)
            if emb is not None:
                h = exp_map(emb)
                d = hyperbolic_distance(query_embedding, h)
            else:
                d = float('inf')
            distances.append(d)
        best_idx = max(range(len(distances)), key=lambda i: distances[i])
        return self.gaps[best_idx]

    def run_round(self, process_file_callback, tracker, max_rounds=3):

        """Run active learning rounds using Recoll."""
        from core.recoll_client import RecollClient
        from deep_research.recoll_guided_learning import _log_query, _mark_processed
        from pathlib import Path

        recoll = RecollClient()
        for round_num in range(max_rounds):
            gap = self.select_gap()
            if not gap:
                print("No gaps to process.")
                break
            query = self.generate_recoll_query(gap)
            print(f"Active learning round {round_num+1}: gap '{gap.get('type')}' on '{gap.get('entity')}', query: {query}")
            try:
                results, _ = recoll.search(query, limit=5)
            except Exception as e:
                print(f"Recoll search failed: {e}")
                self.update(gap.get("id", id(gap)), False)
                continue

            processed = 0
            for doc in results:
                file_path = doc.get("path")
                if not file_path:
                    continue
                try:
                    success = process_file_callback(Path(file_path), tracker)
                    if success:
                        processed += 1
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
            self.update(gap.get("id", id(gap)), processed > 0)
        recoll.close()
