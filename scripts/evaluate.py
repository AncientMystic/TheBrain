#!/usr/bin/env python3
"""
Basic evaluation script for TheBrain retrieval.

Usage:
    python scripts/evaluate.py --eval-file data/eval_queries.json
"""
import sys
import json
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from retrieval.orchestrator import RetrievalOrchestrator
from chat.query_analyzer import analyze_query


def load_eval_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_retrieval(query):
    analysis = analyze_query(query)
    orchestrator = RetrievalOrchestrator()
    datapoints = orchestrator.retrieve(query, analysis, top_k=10)
    fact_ids = []
    for dp in datapoints:
        if dp.get('type') == 'fact' and dp.get('id', '').startswith('fact:'):
            try:
                fact_ids.append(int(dp['id'].split(':')[1]))
            except (ValueError, IndexError):
                continue
    return fact_ids


def compute_metrics(query_data, retrieved_ids):
    relevant = set(query_data.get('relevant_fact_ids', []))
    retrieved = retrieved_ids
    if not relevant:
        return None

    recall = len(relevant & set(retrieved)) / len(relevant)
    precision = len(relevant & set(retrieved)) / len(retrieved) if retrieved else 0.0
    mrr = 0.0
    for i, fid in enumerate(retrieved):
        if fid in relevant:
            mrr = 1.0 / (i + 1)
            break

    return {
        'recall@10': recall,
        'precision@10': precision,
        'mrr': mrr,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate retrieval quality.')
    parser.add_argument('--eval-file', required=True, help='Path to evaluation JSON file.')
    args = parser.parse_args()

    queries = load_eval_data(args.eval_file)
    metrics_sum = {'recall@10': 0.0, 'precision@10': 0.0, 'mrr': 0.0}
    count = 0

    for q in queries:
        query_text = q.get('query')
        if not query_text:
            continue
        print(f"Evaluating: {query_text}")
        retrieved_ids = run_retrieval(query_text)
        metrics = compute_metrics(q, retrieved_ids)
        if metrics:
            for key in metrics_sum:
                metrics_sum[key] += metrics[key]
            count += 1

    if count > 0:
        print("\nAverage metrics:")
        for key, value in metrics_sum.items():
            avg = value / count
            print(f"  {key}: {avg:.4f}")
    else:
        print("No valid queries with relevant facts found.")


if __name__ == "__main__":
    main()
