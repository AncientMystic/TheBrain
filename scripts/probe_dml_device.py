"""Benchmark DML device_id values to find the NVIDIA adapter.

For each candidate device_id: builds the mxbai ONNX session pinned to that
DML device, warms up, times K encodes of 32 short texts, and samples
nvidia-smi utilization during the run (nonzero NVIDIA util identifies the
RTX card empirically — no DXGI name-mapping needed).

Usage: probe_dml_device.py [--ids 0,1] [--reps 6]
Prints a table; fastest id wins. Exit 0 always (informational).
"""
import argparse
import subprocess
import sys
import time

sys.path.insert(0, "A:/scripts/TheBrain")


def nvidia_util():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        return [l.strip() for l in out.stdout.strip().splitlines()]
    except Exception:
        return []


def bench(device_id, reps):
    import numpy as _np
    import onnxruntime as _ort
    from pathlib import Path
    import config as _cfg
    from transformers import AutoTokenizer
    model_dir = Path(_cfg.LOCAL_EMBEDDER_MODEL_DIR)
    so = _ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    prov = [("DmlExecutionProvider", {"device_id": str(device_id)}),
            "CPUExecutionProvider"]
    ses = _ort.InferenceSession(str(model_dir / "model.onnx"),
                                sess_options=so, providers=prov)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    texts = ["benchmark probe sentence number %d about Paris and Curie" % i
             for i in range(32)]
    enc = tok(texts, padding=True, truncation=True, max_length=64,
              return_tensors="np")
    feeds = {"input_ids": enc["input_ids"].astype(_np.int64),
             "attention_mask": enc["attention_mask"].astype(_np.int64)}
    names = [i.name for i in ses.get_inputs()]
    feeds = {k: v for k, v in feeds.items() if k in names}
    if "token_type_ids" in names and "token_type_ids" not in feeds:
        feeds["token_type_ids"] = _np.zeros_like(feeds["input_ids"])
    ses.run(None, feeds)  # warmup (compiles shaders on DML)
    ses.run(None, feeds)
    t0 = time.time()
    for _ in range(reps):
        ses.run(None, feeds)
    dt = time.time() - t0
    return 32 * reps / max(dt, 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="0,1")
    ap.add_argument("--reps", type=int, default=6)
    args = ap.parse_args()
    print(f"{'device_id':>9} | {'enc/s':>8} | nvidia-smi util%")
    best = None
    for did in [s.strip() for s in args.ids.split(",")]:
        try:
            rate = bench(did, args.reps)
            util = nvidia_util()
            print(f"{did:>9} | {rate:8.1f} | {util}", flush=True)
            if best is None or rate > best[1]:
                best = (did, rate)
        except Exception as e:
            print(f"{did:>9} | FAILED: {str(e)[:120]}", flush=True)
    if best:
        print(f"winner: device_id={best[0]} ({best[1]:.1f} enc/s)")


if __name__ == "__main__":
    main()
