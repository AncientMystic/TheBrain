"""Trivium seed populate (phase 86): curated grammar/logic/rhetoric knowledge.

~110 entities: 3 arts, 6 history anchors (Capella, Aristotle, Cicero,
Quintilian, Sayers, Bauer — dates verified), 12 fallacies, 10 key terms
(incl. 5 canons of rhetoric), 24 fields x 3 stages distilled from the
project rundown. Idempotent (INSERT OR IGNORE / REPLACE). Run:
populate_trivium.py [db-path] (defaults to live trivium.db).
"""
import sqlite3
import sys

sys.path.insert(0, "A:/scripts/TheBrain")

ARTS = [
 ("trivium:art:grammar", "art", "art", "Grammar — the Input Stage",
  "trivium:core",
  "Foundational knowledge of any subject: facts, terms, names, dates, rules. "
  "Core questions: What? Who? Where? When? Method: memorization, recitation, "
  "drilling, timelines, observation. Mastery of essentials before moving on.",
  ["grammar", "grammar stage", "input stage"]),
 ("trivium:art:logic", "art", "art", "Logic (Dialectic) — the Processing Stage",
  "trivium:core",
  "Reasoning, analysis, argumentation: how facts fit together, why they hold, "
  "what follows. Core questions: Why? How? What is the relationship? Method: "
  "syllogisms, Socratic dialogue, cause/effect analysis, fallacy detection.",
  ["logic", "dialectic", "logic stage", "processing stage"]),
 ("trivium:art:rhetoric", "art", "art", "Rhetoric — the Output Stage",
  "trivium:core",
  "Effective, persuasive, beautiful expression of understood knowledge. Core "
  "questions: How to express this? What to do with it? Method: essays, speeches, "
  "debate, teaching others, original projects for real audiences.",
  ["rhetoric", "rhetoric stage", "output stage"]),
]

HISTORY = [
 ("trivium:hist:capella", "person", "person", "Martianus Capella",
  "trivium:history",
  "5th-century author of De nuptiis Philologiae et Mercurii, the work that "
  "fixed the seven liberal arts (trivium + quadrivium) as the classical curriculum.",
  ["martianus capella", "capella"]),
 ("trivium:hist:aristotle", "person", "person", "Aristotle — Organon",
  "trivium:history",
  "Organon (Categories, On Interpretation, Prior Analytics): the syllogism and "
  "formal deductive logic that anchor the trivium's Logic stage.",
  ["aristotle", "organon"]),
 ("trivium:hist:cicero", "person", "person", "Cicero",
  "trivium:history",
  "De Inventione and De Oratore: the five canons of rhetoric (inventio, "
  "dispositio, elocutio, memoria, actio) that structure the Rhetoric stage.",
  ["cicero"]),
 ("trivium:hist:quintilian", "person", "person", "Quintilian",
  "trivium:history",
  "Institutio Oratoria: complete rhetorical education from grammar through "
  "eloquence; the teacher as mentor, coach, and audience.",
  ["quintilian"]),
 ("trivium:hist:sayers", "person", "person", "Dorothy Sayers",
  "trivium:history",
  "1947 Oxford address 'The Lost Tools of Learning': revived the trivium as "
  "developmental stages — poll-parrot (grammar), pert (logic), poetic (rhetoric).",
  ["dorothy sayers", "sayers", "lost tools of learning"]),
 ("trivium:hist:bauer", "person", "person", "Susan Wise Bauer",
  "trivium:history",
  "The Well-Trained Mind (1999, with Jessie Wise): modern classical-education "
  "manual sequencing grammar, logic, and rhetoric across the school years.",
  ["susan wise bauer", "bauer", "well-trained mind"]),
]

FALLACIES = [
 ("ad-hominem", "Attacking the person instead of the argument. Ex: dismissing a claim because of who said it."),
 ("strawman", "Refuting a distorted, weaker version of the argument. Ex: caricaturing then knocking down."),
 ("false-dilemma", "Presenting two options as exhaustive when more exist. Ex: you are with us or against us."),
 ("slippery-slope", "Claiming one step inevitably triggers extremes without evidence of the chain."),
 ("circular-reasoning", "The conclusion hidden in the premises; begging the question."),
 ("hasty-generalization", "General rule from too small a sample. Ex: one bad meal, whole cuisine judged."),
 ("appeal-to-authority", "Treating an unqualified or vague authority as proof."),
 ("appeal-to-ignorance", "True because unproven false (or reverse). Absence of evidence as evidence."),
 ("red-herring", "Irrelevant diversion from the issue under debate."),
 ("bandwagon", "True/popular conflated: everyone believes it, so it must hold."),
 ("no-true-scotsman", "Retro-redefining the group to exclude counterexamples."),
 ("equivocation", "One word shifting meaning mid-argument to fake validity."),
]

TERMS = [
 ("trivium:term:syllogism", "Syllogism", "Two premises yielding a necessary conclusion. Ex: All men are mortal; Socrates is a man; therefore Socrates is mortal."),
 ("trivium:term:dialectic", "Dialectic", "Reasoning through dialogue and contradiction toward truth; the Socratic engine of the Logic stage."),
 ("trivium:term:inventio", "Inventio", "First canon of rhetoric: discovering the available arguments and evidence."),
 ("trivium:term:dispositio", "Dispositio", "Second canon: arranging arguments for maximum persuasive effect."),
 ("trivium:term:elocutio", "Elocutio", "Third canon: style — diction, rhythm, figures of speech fitting audience and purpose."),
 ("trivium:term:memoria", "Memoria", "Fourth canon: internalizing the speech for fluent, confident delivery."),
 ("trivium:term:actio", "Actio", "Fifth canon: delivery — voice, gesture, presence before a real audience."),
 ("trivium:term:trivium-def", "Trivium (definition)", "The three language arts — grammar, logic, rhetoric: foundation of the seven liberal arts."),
 ("trivium:term:quadrivium-def", "Quadrivium (definition)", "The four mathematical arts — arithmetic, geometry, music, astronomy — studied after the trivium."),
 ("trivium:term:socratic", "Socratic questioning", "Guided probing (clarify, probe assumptions, evidence, implications) that moves learners from grammar to logic."),
]

FIELDS = {
 "literature": ("Literature",
  "Plots, characters, settings, authors, periods; terms like metaphor, irony, protagonist.",
  "Analyze themes, symbolism, motives, structure; compare works; test plot consistency.",
  "Write criticism; deliver monologues; adapt stories; lead discussions."),
 "language-arts": ("Language Arts",
  "Parts of speech, diagramming, punctuation, spelling, roots, conjugations.",
  "Analyze syntax and word order; parse complex sentences; map conjunction logic.",
  "Write essays and stories; deliver speeches; edit for clarity; teach rules."),
 "foreign-languages": ("Foreign Languages",
  "Vocabulary, verb paradigms, genders, pronunciation, sentence patterns by drill.",
  "Grasp how case, aspect, order encode meaning; translate; compare with native tongue.",
  "Converse, write, present, debate in the language; translate preserving tone."),
 "arithmetic": ("Arithmetic",
  "Addition/subtraction facts, times tables, divisibility, primes; factor, multiple.",
  "Prove why algorithms work; derive number properties; translate word problems.",
  "Explain proofs; write problems for others; teach; essay on a math idea's history."),
 "algebra": ("Algebra",
  "Symbols, order of operations, equality properties, quadratic formula, vocabulary.",
  "Solve multi-step equations; derive formulas; analyze functions and inverses.",
  "Model real situations; present solutions; create tutorials; defend approaches."),
 "geometry": ("Geometry",
  "Point, line, angle, polygon; postulates; Pythagorean and congruence theorems.",
  "Write deductive proofs; relate shapes; derive areas; spot argument gaps.",
  "Present proofs orally; build models; essay on non-Euclidean ideas; teach bisection."),
 "calculus": ("Calculus",
  "Limit, derivative, integral; dy/dx and integral notation; core rules.",
  "Grasp the Fundamental Theorem; prove rules; analyze convergence; optimize.",
  "Explain to non-math audiences; apply to physics/economics; visualize derivatives."),
 "biology": ("Biology",
  "Taxonomy, organelles, species names, pathways like photosynthesis; lab safety.",
  "Study mechanisms; analyze data; build trees; evaluate evolution evidence; experiment.",
  "Write lab reports; present posters; debate bioethics; teach anatomy."),
 "chemistry": ("Chemistry",
  "Periodic table, symbols, formulas, nomenclature, reaction types, equipment.",
  "Balance equations; predict products; use stoichiometry; interpret spectra.",
  "Demo with explanation; write synthesis papers; defend safety positions; tutor."),
 "physics": ("Physics",
  "Units, constants, Newton/Ohm laws, F=ma and E=mc2, velocity/force/energy.",
  "Derive from first principles; solve multi-step; diagram forces; assess error.",
  "Explain in plain language; build demos; paper on relativity; solve live problems."),
 "astronomy": ("Astronomy",
  "Constellations, star types, planets, light-year/AU units, Kepler's laws.",
  "Compute orbits; grasp stellar evolution; read redshift; weigh dark matter.",
  "Run planetarium shows; write discovery articles; defend funding; chart for beginners."),
 "history": ("History",
  "Dates, names, places, events, timelines, primary documents.",
  "Trace causes/effects; compare civilizations; weigh interpretations; spot bias.",
  "Write persuasive essays; speak as figures; film documentaries; debate decisions."),
 "psychology": ("Psychology",
  "Cognition/behavior/conditioning terms; Freud, Skinner, Piaget; DSM criteria.",
  "Judge methods; read case studies; grasp significance; name biases; experiment.",
  "Analyze cases; propose research; mock-counsel; campaign for mental health."),
 "economics": ("Economics",
  "Supply, demand, GDP, inflation; elasticity and opportunity-cost formulas; schools.",
  "Model market shifts; judge fiscal/monetary policy; reason about incentives.",
  "Write policy papers; mock central banking; debate trade; explain to laypeople."),
 "philosophy": ("Philosophy",
  "Metaphysics/epistemology/ethics terms; major thinkers; ontological arguments.",
  "Dissect arguments; build syllogisms; test premises; expose fallacies; dialogue.",
  "Write essays; debate formally; lecture on dilemmas; defend a worldview."),
 "music": ("Music",
  "Notation, scales, chords, key signatures, instruments, theory terms.",
  "Analyze harmony and form; compose; judge performance; transpose.",
  "Perform expressively; compose; conduct; annotate programs; teach lessons."),
 "law": ("Law",
  "Legal terms, case names, statutes, court hierarchy, procedure.",
  "Read case law; build arguments; spot issues; apply rules; weigh evidence.",
  "Write briefs; argue orally; negotiate; draft; teach law to laypeople."),
 "medicine": ("Medicine",
  "Anatomy, physiology, pharmacology, terminology, disease classes, criteria.",
  "Read symptoms and labs; differential diagnosis; weigh treatments; mechanisms.",
  "Write notes and reports; present rounds; deliver diagnoses kindly; teach."),
 "engineering": ("Engineering",
  "Math, physics, materials, standards, CAD, safety codes.",
  "Analyze loads; design systems; debug failures; optimize; model behavior.",
  "Pitch designs; write reports; draw schematics; lead teams; explain simply."),
 "computer-science": ("Computer Science",
  "Syntax, types, control flow, algorithms, libraries, toolchains.",
  "Debug; analyze complexity; design structures; reason about correctness.",
  "Write clean documented code; open-source; present; teach workshops."),
 "cooking": ("Cooking",
  "Ingredients, saute/braise techniques, knife skills, measures, recipe terms.",
  "Grasp why recipes work (reactions, heat); adapt; rescue failures; pair flavors.",
  "Invent recipes; blog; teach classes; plate beautifully; compete."),
 "personal-finance": ("Personal Finance",
  "Interest, APR, diversification; account types; tax rules; budget categories.",
  "Compare investments; compound interest; weigh risk; plan taxes; build a plan.",
  "Write the plan; teach money classes; explain products; advocate literacy."),
 "data-science": ("Data Science",
  "Python/R, statistics vocabulary, data types, SQL/Tableau tooling.",
  "Clean and model data; judge performance; find patterns; design experiments.",
  "Dashboard insights; write data articles; give findings talks; teach literacy."),
 "ai-ethics": ("AI Ethics",
  "Neural-network/ML terms, ethical frameworks, case studies, regulations.",
  "Audit bias; weigh dilemmas; trace consequences; design fair systems.",
  "Write ethics policy; debate AI rights; present research; teach with video."),
}


def _rows():
    rows = []
    for cid, et, tf, dn, sh, desc, als in ARTS + HISTORY:
        rows.append((cid, et, tf, dn, sh, desc, als))
    for slug, desc in FALLACIES:
        rows.append((f"trivium:fallacy:{slug}", "fallacy", "tool",
                     slug.replace("-", " ").title(), "trivium:logic-tools",
                     "Logical fallacy — " + desc, [slug.replace("-", " ")]))
    for cid, dn, desc in TERMS:
        rows.append((cid, "term", "tool", dn, "trivium:logic-tools", desc,
                     [dn.lower()]))
    for slug, (name, g, l, r) in FIELDS.items():
        sh = f"trivium:field:{slug}"
        rows.append((f"trivium:{slug}:grammar", "stage", "stage",
                     f"{name} — Grammar", sh,
                     f"Grammar of {name}: {g}", [f"{name.lower()} basics"]))
        rows.append((f"trivium:{slug}:logic", "stage", "stage",
                     f"{name} — Logic", sh,
                     f"Logic of {name}: {l}", [f"{name.lower()} analysis"]))
        rows.append((f"trivium:{slug}:rhetoric", "stage", "stage",
                     f"{name} — Rhetoric", sh,
                     f"Rhetoric of {name}: {r}", [f"{name.lower()} expression"]))
    return rows


def populate(conn):
    rows = _rows()
    n_e = n_a = 0
    for cid, et, tf, dn, sh, desc, als in rows:
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
    conn.commit()
    return n_e, n_a


if __name__ == "__main__":
    import sys as _sys
    sys.path.insert(0, "A:/scripts/TheBrain/scripts")
    from init_mapping_db import init_mapping_db
    from migrate_hyper_coords import migrate
    import config as _cfg
    # Same schema + geometry columns as mapping: the whole pipeline
    # (embed/derive/oct8/e8/scoring) works on trivium.db unchanged.
    _path = _sys.argv[1] if len(_sys.argv) > 1 else _cfg.TRIVIUM_DB_FILE
    _conn = init_mapping_db(_path)
    migrate(_conn)
    _e, _a = populate(_conn)
    print(f"populate_trivium: {_e} entities, {_a} aliases")
    _conn.close()
