"""Trivium expansion seed (phase 87): data-driven detection patterns + inventory.

- ~30 detection_pattern rows: REGEX + TARGET canonical fallacy. The
  detector (core/trivium_detect.py) loads these live from trivium.db —
  patterns are data, never hardcoded.
- 12 more fallacies (modern debate tactics + formal remainder).
- 8 more fields x 3 stages. 8 more devices. 4 more canon techniques.
Idempotent. Run: populate_trivium3.py [db-path].
"""
import sys

sys.path.insert(0, "A:/scripts/TheBrain")

# (slug-suffix, TARGET canonical, REGEX, note)
PATTERNS = [
 ("tu-quoque-1", "trivium:fal:tu-quoque", r"\byou(?: \w+){0,3} too\b",
  "Reciprocity charge with up to 3 words between (you litter too); check relevance."),
 ("tu-quoque-2", "trivium:fal:tu-quoque", r"look who('| i)s talking",
  "Dismissal via accuser hypocrisy; credibility rarely decides the claim."),
 ("scotsman-1", "trivium:fal:no-true-scotsman", r"\bno true \w+",
  "Ad-hoc redefinition flag; verify against prior definition."),
 ("scotsman-2", "trivium:fal:no-true-scotsman", r"\bnot a (true|real) \w+",
  "Purity gatekeeping after counterexample."),
 ("strawman-1", "trivium:fal:strawman", r"\bso you('| a)re saying\b",
  "Possible caricature setup; compare against original claim text."),
 ("strawman-2", "trivium:fal:strawman", r"\bso you think\b",
  "Same check on shorter form."),
 ("slope-1", "trivium:fal:slippery-slope", r"\bwhere will it end\b",
  "Unbounded chain asserted without links."),
 ("slope-2", "trivium:fal:slippery-slope", r"\bthin end of the wedge\b",
  "Idiom form of the same move."),
 ("slope-3", "trivium:fal:slippery-slope", r"\bwill (inevitably|necessarily) lead to\b",
  "Necessity claimed for contingent chain."),
 ("dilemma-1", "trivium:fal:false-dilemma", r"\bwith us or against us\b",
  "Exhaustiveness asserted, alternatives suppressed."),
 ("dilemma-2", "trivium:fal:false-dilemma", r"\beither .* or .* no (other|third) (choice|option|alternative)",
  "Explicit false exhaustiveness."),
 ("posthoc-1", "trivium:fal:post-hoc", r"\bafter .* (therefore|hence|thus|so)\b",
  "Temporal order sold as causation (real-prose connectors, not just therefore)."),
 ("sharpshooter-1", "trivium:fal:texas-sharpshooter", r"\bcherry.?pick",
  "Selective cluster presented as pattern."),
 ("sharpshooter-2", "trivium:fal:texas-sharpshooter", r"\bonly (counting|showing|looking at) the\b",
  "Denominator suppression flag."),
 ("bandwagon-1", "trivium:fal:bandwagon", r"\bevery(one|body) (knows|agrees|believes|says)\b",
  "Popularity as proof."),
 ("authority-1", "trivium:fal:appeal-to-authority", r"\bstudies show\b",
  "Uncited study invocation; demand the actual paper."),
 ("authority-2", "trivium:fal:appeal-to-authority", r"\bexperts agree\b",
  "Unnamed experts; check domain match."),
 ("authority-3", "trivium:fal:appeal-to-authority", r"\bscientists say\b",
  "Same check, science-flavored."),
 ("pity-1", "trivium:fal:ad-misericordiam", r"\b(feel sorry|have a heart)\b",
  "Pity recruited as evidence."),
 ("fear-1", "trivium:fal:ad-baculum", r"\bor else\b",
  "Threat structure; separate consequence from reason."),
 ("tradition-1", "trivium:fal:ad-antiquitatem", r"\bthe way we('| ha)ve always\b",
  "Longevity sold as correctness."),
 ("gambler-1", "trivium:fal:gamblers", r"\bis due\b",
  "Independence denial (streaks, slots, losses)."),
 ("sunk-1", "trivium:fal:sunk-cost", r"\b(already (spent|invested|paid)|in too deep|can('| no)t quit now)\b",
  "Past spend argued as future reason."),
 ("nirvana-1", "trivium:fal:nirvana", r"\bunless .* perfect\b",
  "Perfection demanded of the viable."),
 ("reify-1", "trivium:fal:reification", r"\b(history will judge|science says|evolution (designed|intended))\b",
  "Abstract process cast as agent."),
 ("etym-1", "trivium:fal:etymological", r"\b(originally|literally) mean",
  "Origin sold as true meaning."),
 ("equiv-signal", "trivium:fal:equivocation", r"STRUCT:repeated-term-sense-shift",
  "Structural: same surface term, shifting sense across premises. Detector checks definition drift."),
 ("chiasmus-signal", "trivium:dev:chiasmus", r"STRUCT:abba-reversal",
  "Structural: ABBA phrase reversal. Detector matches mirrored bigrams."),
 ("anaphora-signal", "trivium:dev:anaphora", r"STRUCT:repeated-openings",
  "Structural: 3+ sentences sharing opening 3-gram. Detector counts."),
 ("survivor-1", "trivium:fal:survivorship", r"\bdropped out.*(billionaire|success|rich)\b",
  "Denominator blindness; ask for the failures."),
]

FALLACIES2 = [
 ("trivium:fal:whataboutism", "Whataboutism", "relevance",
  "Deflect to opponent's worse act (our emissions vs Country X). Distinct from tu quoque: changes topic rather than mirroring charge.",
  ["whataboutism", "what about"]),
 ("trivium:fal:anecdotal", "Anecdotal Evidence", "presumption",
  "Single vivid story as statistical proof. Test: ask for denominator and base rate.",
  ["anecdotal", "anecdote as proof"]),
 ("trivium:fal:sunk-cost", "Sunk-Cost Reasoning", "presumption",
  "Past unrecoverable spend argued as reason to continue. Test: would you start this today at zero cost?",
  ["sunk cost", "sunk cost fallacy"]),
 ("trivium:fal:nirvana", "Nirvana Fallacy", "presumption",
  "Reject the viable for not being perfect. Test: compare against real alternatives, not ideals.",
  ["nirvana fallacy", "perfect solution fallacy"]),
 ("trivium:fal:motte-bailey", "Motte and Bailey", "ambiguity",
  "Retreat from bold claim (bailey) to trivial one (motte) under attack, then return. Test: pin which claim is defended.",
  ["motte and bailey"]),
 ("trivium:fal:modal-scope", "Modal Scope Fallacy", "formal",
  "Shift necessity between dictum and modus (necessarily-if vs if-necessarily). Test: formalize modal operators explicitly.",
  ["modal scope", "modal fallacy"]),
 ("trivium:fal:quantifier-shift", "Quantifier Shift", "formal",
  "Every X has some Y, therefore some Y fits every X (order swap). Test: preserve quantifier order in rewrite.",
  ["quantifier shift", "quantifier order"]),
 ("trivium:fal:gish-gallop", "Gish Gallop", "relevance",
  "Overwhelm with quantity of weak arguments; refutation costs more than assertion. Counter: pick strongest, demand one.",
  ["gish gallop", "firehose"]),
 ("trivium:fal:sealioning", "Sealioning", "relevance",
  "Bad-faith endless demands for evidence as harassment. Distinguish from genuine steelmanning by proportionality.",
  ["sealioning", "bad faith demands"]),
 ("trivium:fal:tone-policing", "Tone Policing", "relevance",
  "Dismiss argument via delivery critique. Test: restate claim flatly — does objection survive?",
  ["tone policing"]),
 ("trivium:fal:appeal-silence", "Appeal to Silence", "presumption",
  "No response taken as concession. Test: silence has many causes; ask for positive evidence.",
  ["appeal to silence", "silence means consent"]),
 ("trivium:fal:moralistic", "Moralistic Fallacy", "presumption",
  "Derive is from ought (nature must be fair, so it is). Mirror image of naturalistic fallacy.",
  ["moralistic fallacy", "wishful is-ought"]),
]

FIELDS2 = {
 "calculus": ("Calculus",
  "Limit, derivative, integral; dy/dx notation; core differentiation/integration rules.",
  "Grasp the Fundamental Theorem; prove rules; analyze convergence; optimize with constraints.",
  "Explain to non-math audiences; apply to physics and economics; visualize change."),
 "statistics": ("Statistics",
  "Mean, median, deviation; distributions; z-score and Bayes formulas; graph types.",
  "Audit samples; condition probabilities; test hypotheses; split correlation from cause.",
  "Report findings ethically; critique misused stats; design and analyze surveys."),
 "sociology": ("Sociology",
  "Socialization, stratification, norms; Durkheim, Weber, Marx; research methods.",
  "Analyze structures; read demographic data; critique designs; spot patterns.",
  "Survey and present; write op-eds; lead inequality discussions; infograph trends."),
 "political-science": ("Political Science",
  "Branches, constitution articles, ideologies, voting systems, key cases.",
  "Analyze policy; model campaigns; compare constitutions; catch speech fallacies.",
  "Write briefs; mock legislatures; stump speeches; debate; voter guides."),
 "art": ("Art",
  "Artists, periods, styles, techniques, color theory, core vocabulary.",
  "Read composition and symbolism; compare works; judge technique effectiveness.",
  "Create in studied styles; curate; write statements; give gallery talks."),
 "business": ("Business",
  "Accounting terms, marketing concepts, models, statements, management theories.",
  "Read trends; judge investments; plan businesses; diagnose organizations.",
  "Propose and pitch; lead meetings; negotiate; campaign; teach."),
 "sports": ("Sports",
  "Rules, positions, terminology, equipment, basic techniques.",
  "Analyze strategy; study biomechanics; exploit weaknesses; adjust tactics.",
  "Coach; commentate; plan training; write playbooks; run clinics."),
 "gardening": ("Gardening",
  "Plant names, soils, zones, tools, care requirements.",
  "Grasp plant biology and soil chemistry; manage pests; diagnose problems.",
  "Design gardens; write guides; lead community plots; teach propagation."),
}

DEVICES2 = [
 ("Tetracolon", "scheme", "Four-beat climax (fought, endured, sacrificed, triumphed). Use when three beats understate.",
  ["tetracolon"]),
 ("Antimetabole", "scheme", "Word-order reversal (eat to live, not live to eat). Chiasmus in pure word form.",
  ["antimetabole"]),
 ("Polyptoton", "scheme", "Same root varied (judge not, that ye be not judged). Cohesion through morphology.",
  ["polyptoton"]),
 ("Climax", "scheme", "Gradatio chain (came, saw, conquered, ruled). Each step must entail the next.",
  ["climax", "gradatio"]),
 ("Ellipsis", "scheme", "Omission understood (she loves order; he, chaos). Pace; never omit load-bearing terms.",
  ["ellipsis"]),
 ("Paradox", "trope", "Apparent contradiction resolving true (less is more). Must resolve on inspection.",
  ["paradox"]),
 ("Euphemism-Flag", "trope", "Softening that hides agency (collateral, rightsizing). Flag in verification output.",
  ["euphemism"]),
 ("Dysphemism-Flag", "trope", "Harshening that smuggles judgment (corrupt, stupid). Flag in verification output.",
  ["dysphemism"]),
]

CANONTECH2 = [
 ("canontech:refutatio", "Refutatio Placement", "canon",
  "Answer the strongest objection mid-speech, never only at the end. Buried refutation reads as evasion.",
  ["refutatio", "refutation placement"]),
 ("canontech:peroratio", "Peroratio Close", "canon",
  "Close with recapitulation + emotional peak. Never introduce new claims in the close.",
  ["peroratio", "closing"]),
 ("canontech:narratio", "Narratio Clarity", "canon",
  "State facts briefly, clearly, plausibly before arguing. Confused narrative poisons confirmatio.",
  ["narratio", "statement of facts"]),
 ("canontech:exordium", "Exordium Hook", "canon",
  "Win attention and goodwill in the opening; match register to audience status.",
  ["exordium", "opening hook"]),
]


def populate(conn):
    n_e = n_a = 0

    def add(cid, et, tf, dn, sh, desc, als):
        nonlocal n_e, n_a
        conn.execute("INSERT OR IGNORE INTO entities(canonical_id, entity_type,"
                     " type_family, display_name, shard_key, description)"
                     " VALUES (?,?,?,?,?,?)", (cid, et, tf, dn, sh, desc))
        n_e += 1
        try:
            conn.execute("DELETE FROM aliases_fts WHERE canonical_id=?", (cid,))
        except Exception:
            pass
        for a in als:
            try:
                conn.execute("INSERT OR IGNORE INTO aliases(alias_norm, canonical_id)"
                             " VALUES (?,?)", (a.lower(), cid))
                conn.execute("INSERT INTO aliases_fts(alias_norm, canonical_id)"
                             " VALUES (?,?)", (a.lower(), cid))
                n_a += 1
            except Exception:
                continue

    # Legacy canonical scheme: the original 12 fallacies live under
    # trivium:fallacy:<slug> (phase 86 seed); point targets at them.
    _LEGACY = {"bandwagon", "appeal-to-authority", "equivocation",
               "false-dilemma", "no-true-scotsman", "slippery-slope",
               "strawman"}
    for slug, target, rx, note in PATTERNS:
        for _leg in _LEGACY:
            target = target.replace(f"trivium:fal:{_leg}",
                                    f"trivium:fallacy:{_leg}")
        desc = f"TARGET: {target}. REGEX: {rx}. {note}"
        add(f"trivium:pat:{slug}", "detection_pattern", "regex", f"Pattern: {slug}",
            "trivium:patterns:regex", desc, [slug.replace("-", " ")])
    for cid, dn, fam, desc, als in FALLACIES2:
        add(cid, "fallacy", fam, dn, f"trivium:logic:fallacy:{fam}", desc, als)
    for slug, (name, g, l, r) in FIELDS2.items():
        sh = f"trivium:field:{slug}"
        add(f"trivium:{slug}:grammar", "stage", "stage", f"{name} — Grammar", sh,
            f"Grammar of {name}: {g}", [f"{name.lower()} basics"])
        add(f"trivium:{slug}:logic", "stage", "stage", f"{name} — Logic", sh,
            f"Logic of {name}: {l}", [f"{name.lower()} analysis"])
        add(f"trivium:{slug}:rhetoric", "stage", "stage", f"{name} — Rhetoric", sh,
            f"Rhetoric of {name}: {r}", [f"{name.lower()} expression"])
    for name, fam, desc, als in DEVICES2:
        import re as _re
        slug = _re.sub(r"-+", "-", "".join(
            c.lower() if c.isalnum() else "-" for c in name)).strip("-")
        add(f"trivium:dev:{slug}", "rhet_device", fam, f"{name} [{fam}]",
            f"trivium:rhetoric:device:{fam}", desc, als)
    for cid, dn, tf, desc, als in CANONTECH2:
        add(cid, "canon_technique", tf, dn, "trivium:rhetoric:canon:general",
            desc, als)
    conn.commit()
    return n_e, n_a


if __name__ == "__main__":
    import sys as _sys
    sys.path.insert(0, "A:/scripts/TheBrain/scripts")
    from init_mapping_db import init_mapping_db
    from migrate_hyper_coords import migrate
    import config as _cfg
    _path = _sys.argv[1] if len(_sys.argv) > 1 else _cfg.TRIVIUM_DB_FILE
    _conn = init_mapping_db(_path)
    migrate(_conn)
    _e, _a = populate(_conn)
    print(f"populate_trivium3: {_e} entities, {_a} aliases")
    _conn.close()
