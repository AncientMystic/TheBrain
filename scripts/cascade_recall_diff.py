"""Cascade recall-diff (phase 76): GLINER-first cascade on a real document.

Runs the REAL ONNX pre-pass (FastExtractor) + REAL LLM extraction
(_process_batch against the LM Studio backend) on chunks of
deepseek-r1-report.md with GLINER_CASCADE off, then on, and diffs:
  - people-category LLM calls (saved work)
  - people/locations/dates entity sets off-vs-on (losses vs parity)

Model note: the harness uses whatever model LM Studio currently serves
(passed explicitly as --model); the enrolled extraction model may differ.
A same-model on/off diff proves MECHANICS (calls saved, no entity loss);
quality transfer to the enrolled model needs a re-run once it is loaded.
Usage: cascade_recall_diff.py [--model ID] [--chunks N] [--log PATH]
Exit 0 with printed verdict; exit 2 if the backend is unreachable.
"""
import argparse
import sys

sys.path.insert(0, "A:/scripts/TheBrain")
DOC = "A:/scripts/TheBrain/deepseek-r1-report.md"


def _names(items):
    out = set()
    for it in items or []:
        if isinstance(it, dict):
            t = str(it.get("person_name", "") or it.get("location_name", "")
                    or it.get("text", "") or it.get("name", "")).strip().lower()
            if t:
                out.add(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--log", default="")
    args = ap.parse_args()
    logfh = open(args.log, "a", encoding="utf-8") if args.log else None

    def say(msg):
        line = f"[cascade_diff] {msg}"
        print(line, flush=True)
        if logfh:
            logfh.write(line + "\n")
            logfh.flush()

    import config as C
    from extraction import llm_extractor as LX
    from fast_extractor.hybrid_extractor import FastExtractor
    # Clean experiment: disable the prompt cache so skipped categories can
    # only come from cascade logic, never cache hits. Restored at the end.
    _cache_orig = bool(getattr(C, "LLM_EXTRACTION_CACHE", True))
    C.LLM_EXTRACTION_CACHE = False

    text = open(DOC, encoding="utf-8", errors="ignore").read()
    size = max(len(text) // args.chunks, 1)
    chunks = [text[i * size:(i + 1) * size] for i in range(args.chunks)]
    say(f"doc {len(text)} chars -> {len(chunks)} chunks")
    fx = FastExtractor()
    pres = []
    for ch in chunks:
        try:
            pres.append(fx.extract(ch))
        except Exception as e:
            say(f"pre-pass failed: {e}")
            pres.append(None)
    say("pre-pass done")
    ep = dict(C.LLM_ENDPOINTS[0])
    model = args.model or ep.get("model", "")
    ep["model"] = model
    # backend reachability first (fail fast, no partial burns)
    try:
        probe = LX.call_model_json("Reply with {}.", model=model,
                                   max_tokens=8, endpoint=ep)
        say(f"backend ok (probe -> {type(probe).__name__})")
    except Exception as e:
        say(f"backend unreachable for model={model}: {str(e)[:160]}")
        return 2

    results = {}
    for flag in (False, True):
        C.GLINER_CASCADE_ENABLED = flag
        calls = []

        orig = LX.call_model_json

        def counting(prompt, **kw):
            calls.append(prompt)
            return orig(prompt, **kw)

        LX.call_model_json = counting
        try:
            out = LX._process_batch(list(chunks), model=None, logic_context="",
                                    endpoint=ep, actual_model=model,
                                    batch_pre_extractions=list(pres))
        finally:
            LX.call_model_json = orig
        marker = "extract ALL relevant people, locations, and dates"
        people_calls = sum(1 for p in calls if marker in p)
        ents = set()
        for r in out:
            if isinstance(r, dict):
                for k in ("people", "locations", "dates", "entities"):
                    ents |= _names(r.get(k))
        results[flag] = (people_calls, len(calls), ents)
        say(f"cascade={'on ' if flag else 'off'}: people_calls={people_calls}"
            f" total_calls={len(calls)} entities={len(ents)}")
        if flag:
            try:
                covered = LX._gliner_entity_complete_idx(list(chunks), list(pres))
                say(f"covered chunks: {sorted(covered)}")
            except Exception as e:
                covered = set()
                say(f"coverage check failed: {e}")
        else:
            covered = set()
        results[flag] = (people_calls, len(calls), ents, set(covered))
    C.GLINER_CASCADE_ENABLED = False
    C.LLM_EXTRACTION_CACHE = _cache_orig
    off, on = results[False], results[True]
    saved = off[0] - on[0]
    lost = sorted(off[2] - on[2])
    gained = sorted(on[2] - off[2])
    say(f"people LLM calls saved: {saved}/{off[0]}")
    say(f"entities lost by cascade: {len(lost)} {lost[:12]}")
    say(f"entities gained by cascade: {len(gained)} {gained[:12]}")
    say(f"chunks covered by pre-pass: {sorted(on[3])}")
    if not on[3]:
        say("verdict: NO COVERAGE (pre-pass below thresholds on this doc;"
            " savings, if any, are retry noise — flag stays off; tune"
            " thresholds or pre-pass recall, then re-run)")
    elif saved > 0 and not lost:
        say("verdict: MECHANICAL PASS (calls saved, zero entity loss)")
    elif saved <= 0:
        say("verdict: NO SAVINGS (flag stays off)")
    else:
        say("verdict: LOSSES PRESENT (flag stays off pending investigation)")
    if logfh:
        logfh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
