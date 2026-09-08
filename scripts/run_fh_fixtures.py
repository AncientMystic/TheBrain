"""Run all F/H fixtures through the judge (phase 79): build-then-measure.

Loads tests/fixtures_ground_truth.json, judges every faithfulness item
(answer+quotes) and hallucination item (claim+context), compares against
expected labels, prints per-item verdicts + section accuracy + confusion
of review-fallbacks. Exit 0 always (measurement, not gating); the numbers
decide whether gates get built on top.

Usage: run_fh_fixtures.py [--log PATH]
"""
import sys

sys.path.insert(0, "A:/scripts/TheBrain")


def main(argv):
    import json
    log = ""
    model = ""
    _args = list(argv)
    for _i, a in enumerate(_args):
        if a.startswith("--log="):
            log = a.split("=", 1)[1]
        elif a == "--log" and _i + 1 < len(_args):
            log = _args[_i + 1]
        if a.startswith("--model="):
            model = a.split("=", 1)[1]
        elif a == "--model" and _i + 1 < len(_args):
            model = _args[_i + 1]
    logfh = open(log, "a", encoding="utf-8") if log else None

    def say(msg):
        print(msg, flush=True)
        if logfh:
            logfh.write(msg + "\n")
            logfh.flush()

    from core import fh_judge as J
    if model:
        # Temporary in-process override only: routes the judge at the
        # already-loaded model without touching config on disk.
        import config as _C
        try:
            _C.LLM_ENDPOINTS[0]["model"] = model
            say(f"model override: {model}")
        except Exception as e:
            say(f"model override failed: {e}")
            return 2
    d = json.load(open("A:/scripts/TheBrain/tests/fixtures_ground_truth.json",
                       encoding="utf-8"))
    stats = {"faithfulness": [0, 0, 0], "hallucination": [0, 0, 0]}  # hit/total/review
    for it in d.get("faithfulness", []):
        try:
            ans = it.get("answer", it.get("claim", ""))
            label, reason, method = J.judge_faithfulness(ans, it.get("citations", []))
        except Exception as e:
            label, reason, method = "review", str(e)[:120], "rules"
        ok = label == it.get("expected")
        s = stats["faithfulness"]
        s[1] += 1
        s[0] += 1 if ok else 0
        s[2] += 1 if label == "review" else 0
        say(f"[{'OK' if ok else 'MISS'}] {it.get('id')} exp={it.get('expected')}"
            f" got={label} ({method}) {reason[:100]}")
    for it in d.get("hallucination", []):
        try:
            label, reason, method = J.judge_hallucination(it.get("claim", ""),
                                                          it.get("context", ""))
        except Exception as e:
            label, reason, method = "review", str(e)[:120], "rules"
        ok = label == it.get("expected")
        s = stats["hallucination"]
        s[1] += 1
        s[0] += 1 if ok else 0
        s[2] += 1 if label == "review" else 0
        say(f"[{'OK' if ok else 'MISS'}] {it.get('id')} exp={it.get('expected')}"
            f" got={label} ({method}) {reason[:100]}")
    for k, (h, t, r) in stats.items():
        say(f"{k}: {h}/{t} correct ({h * 100.0 / max(t, 1):.0f}%), {r} reviews")
    if logfh:
        logfh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
