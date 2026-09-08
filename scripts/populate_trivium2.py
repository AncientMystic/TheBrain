"""Trivium extension seed (phase 86): HOW-to-think procedures from specialist research.

Adds ~155 rows: 23 logical forms (validity rules included), 47 fallacies
(definition + example + detection test), 30 Socratic stems (6 purposes),
29 rhetorical devices (with examples), 14 progymnasmata (ordered ladder),
8 canon techniques, 4 gate rules (checkable IF/THEN), 4 metacognitive
prompts. Idempotent. Run: populate_trivium2.py [db-path].
Sources: bellion taxonomy + tank research (both retrieved 2026-09-08).
"""
import sys

sys.path.insert(0, "A:/scripts/TheBrain")

# (cid, etype, tfam, display, shard, description, aliases)
FORMS = [
 ("trivium:form:barbara", "logical_form", "syllogistic", "Barbara (AAA-1)",
  "trivium:logic:forms:syllogistic",
  "Universal affirmative Figure 1: All M are P; All S are M; therefore All S are P. "
  "Valid iff middle term distributed once and no illicit process. Ex: mammals breathe; whales are mammals; therefore whales breathe.",
  ["barbara", "aaa-1"]),
 ("trivium:form:celarent", "logical_form", "syllogistic", "Celarent (EAE-1)",
  "trivium:logic:forms:syllogistic",
  "No M are P; All S are M; therefore No S are P. Ex: no fish are birds; trout are fish; therefore no trout are birds.",
  ["celarent"]),
 ("trivium:form:darii", "logical_form", "syllogistic", "Darii (AII-1)",
  "trivium:logic:forms:syllogistic",
  "All M are P; Some S are M; therefore Some S are P. Ex: roses are flowers; some garden plants are roses; therefore some are flowers.",
  ["darii"]),
 ("trivium:form:ferio", "logical_form", "syllogistic", "Ferio (EIO-1)",
  "trivium:logic:forms:syllogistic",
  "No M are P; Some S are M; therefore Some S are not P. Ex: no tyrants are just; some rulers are tyrants; therefore some rulers are not just.",
  ["ferio"]),
 ("trivium:form:modus-ponens", "logical_form", "propositional", "Modus Ponens",
  "trivium:logic:forms:propositional",
  "If P then Q; P; therefore Q. Valid; fails if antecedent equivocated or merely conditional. Detection: rewrite as conditional.",
  ["modus ponens", "affirming the antecedent", "mp"]),
 ("trivium:form:modus-tollens", "logical_form", "propositional", "Modus Tollens",
  "trivium:logic:forms:propositional",
  "If P then Q; not-Q; therefore not-P. Valid contrapositive; requires same sense of P/Q across premises.",
  ["modus tollens", "denying the consequent", "mt"]),
 ("trivium:form:hypothetical-syll", "logical_form", "propositional", "Hypothetical Syllogism",
  "trivium:logic:forms:propositional",
  "If P then Q; if Q then R; therefore if P then R. Valid chain; each link must hold in the same sense.",
  ["hypothetical syllogism", "chain argument"]),
 ("trivium:form:disjunctive-syll", "logical_form", "propositional", "Disjunctive Syllogism",
  "trivium:logic:forms:propositional",
  "P or Q; not-P; therefore Q. Valid only for EXCLUSIVE or; with inclusive or the second premise must exclude properly.",
  ["disjunctive syllogism", "either-or"]),
 ("trivium:form:dilemma", "logical_form", "propositional", "Dilemma",
  "trivium:logic:forms:propositional",
  "P implies R; Q implies R; P or Q; therefore R. Test: are P/Q exhaustive, and do both really imply R?",
  ["dilemma", "constructive dilemma"]),
 ("trivium:form:reductio", "logical_form", "propositional", "Reductio ad Absurdum",
  "trivium:logic:forms:propositional",
  "Assume P; derive contradiction; therefore not-P. Valid iff the contradiction follows strictly from P alone.",
  ["reductio", "proof by contradiction"]),
 ("trivium:form:enthymeme", "logical_form", "dialectical", "Enthymeme",
  "trivium:logic:forms:dialectical",
  "Syllogism with an unstated premise the audience supplies. Flag missing premises explicitly before validating.",
  ["enthymeme", "rhetorical syllogism"]),
 ("trivium:form:toulmin", "logical_form", "dialectical", "Toulmin Model",
  "trivium:logic:forms:dialectical",
  "Claim + Grounds + Warrant + Backing (+Qualifier/Rebuttal). Every chat answer should be decomposable into these slots.",
  ["toulmin", "claim ground warrant"]),
 ("trivium:form:mills-methods", "logical_form", "inductive", "Mill's Methods",
  "trivium:logic:forms:inductive",
  "Agreement/difference: isolate X present when E present, absent when absent. Strength graded by controls, never binary.",
  ["mills methods", "method of difference"]),
 ("trivium:form:ibe", "logical_form", "abductive", "Inference to Best Explanation",
  "trivium:logic:forms:abductive",
  "From O prefer H maximizing consilience, simplicity, testability. Defeasible: withdraw on new O or better H. (Peirce)",
  ["abduction", "ibe", "best explanation"]),
 ("trivium:form:analogical", "logical_form", "inductive", "Analogical Argument",
  "trivium:logic:forms:inductive",
  "X resembles Y in known respects; therefore resembles in further respect. 6-point strength test: relevance, number, disanalogies.",
  ["analogy", "argument from analogy"]),
 ("trivium:form:statistical-syll", "logical_form", "inductive", "Statistical Syllogism",
  "trivium:logic:forms:inductive",
  "Z% of F are G; x is F; therefore probably x is G (probability Z). Requires representative reference class.",
  ["statistical syllogism"]),
 ("trivium:form:prop-a", "logical_form", "propositional", "A Proposition (Universal Affirmative)",
  "trivium:logic:forms:propositional",
  "All S are P (ex: all men are mortal). Distributes subject only. AffIrmo mnemonic.",
  ["a proposition", "universal affirmative"]),
 ("trivium:form:prop-e", "logical_form", "propositional", "E Proposition (Universal Negative)",
  "trivium:logic:forms:propositional",
  "No S are P (ex: no reptiles are mammals). Distributes both subject and predicate. nEgO mnemonic.",
  ["e proposition", "universal negative"]),
 ("trivium:form:prop-i", "logical_form", "propositional", "I Proposition (Particular Affirmative)",
  "trivium:logic:forms:propositional",
  "Some S are P (ex: some philosophers are women). Distributes neither term.",
  ["i proposition", "particular affirmative"]),
 ("trivium:form:prop-o", "logical_form", "propositional", "O Proposition (Particular Negative)",
  "trivium:logic:forms:propositional",
  "Some S are not P (ex: some animals are not dogs). Distributes predicate only.",
  ["o proposition", "particular negative"]),
 ("trivium:form:deduction", "logical_form", "inductive", "Deduction Pattern",
  "trivium:logic:forms:inductive",
  "General to particular, truth-preserving: valid form + true premises (soundness) guarantees conclusion.",
  ["deduction", "deductive reasoning"]),
 ("trivium:form:induction", "logical_form", "inductive", "Induction Pattern",
  "trivium:logic:forms:inductive",
  "Particulars to generalization, ampliative and probable: needs large representative sample; always defeasible (one black swan).",
  ["induction", "inductive reasoning"]),
 ("trivium:form:necessary-sufficient", "logical_form", "propositional", "Necessary vs Sufficient",
  "trivium:logic:forms:propositional",
  "Necessary (no P, no Q; oxygen for fire) vs sufficient (P guarantees Q; match in air for flame). Test each direction separately; confusion drives affirming/denying errors.",
  ["necessary sufficient", "necessary condition"]),
]

FALLACIES = [
 ("trivium:fal:affirming-consequent", "Affirming the Consequent", "formal",
  "If P then Q; Q; therefore P. Invalid: Q may hold for other reasons (wet streets, sprinkler). Detection: rewrite as conditional.",
  ["affirming the consequent", "converse error"]),
 ("trivium:fal:denying-antecedent", "Denying the Antecedent", "formal",
  "If P then Q; not-P; therefore not-Q. Invalid: Q may hold anyway (unstudied yet passed).",
  ["denying the antecedent"]),
 ("trivium:fal:undistributed-middle", "Undistributed Middle", "formal",
  "All Z are P; all S are P; therefore all S are Z. Invalid: middle never distributed (fish/whales both live in water).",
  ["undistributed middle"]),
 ("trivium:fal:illicit-major", "Illicit Major", "formal",
  "Major term distributed in conclusion but not premise. Ex: all dogs are mammals; no cats are dogs; therefore no cats are mammals.",
  ["illicit major"]),
 ("trivium:fal:illicit-minor", "Illicit Minor", "formal",
  "Minor term distributed in conclusion only. Ex: all Greeks are mortal men; therefore all men are mortal.",
  ["illicit minor"]),
 ("trivium:fal:four-terms", "Four Terms", "formal",
  "Syllogism with 4 terms via ambiguous middle (river bank vs money bank holds money).",
  ["four terms", "quaternio terminorum"]),
 ("trivium:fal:existential", "Existential Fallacy", "formal",
  "Deriving 'some X exists' from two universals without existential import (unicorns with horns).",
  ["existential fallacy"]),
 ("trivium:fal:affirming-disjunct", "Affirming a Disjunct", "formal",
  "P or Q; P; therefore not-Q. Fails for inclusive or (tea or coffee; can have both).",
  ["affirming a disjunct", "false exclusion"]),
 ("trivium:fal:ad-baculum", "Appeal to Fear", "relevance",
  "Threat instead of reason (vote or criminals overrun streets).",
  ["appeal to fear", "ad baculum"]),
 ("trivium:fal:ad-misericordiam", "Appeal to Pity", "relevance",
  "Pity as proof (divorced parents deserve an A).",
  ["appeal to pity", "ad misericordiam"]),
 ("trivium:fal:ad-flattery", "Appeal to Flattery", "relevance",
  "Compliment to win assent (a smart person sees this investment is sound).",
  ["appeal to flattery", "adulation"]),
 ("trivium:fal:ad-ridicule", "Appeal to Ridicule", "relevance",
  "Mockery as refutation (economics from cavemen).",
  ["appeal to ridicule", "mockery"]),
 ("trivium:fal:ad-spite", "Appeal to Spite", "relevance",
  "Exploiting bitterness (reject the rich kid's plan).",
  ["appeal to spite"]),
 ("trivium:fal:ad-naturam", "Appeal to Nature", "relevance",
  "Natural equals good (herb is natural so safe and effective).",
  ["appeal to nature"]),
 ("trivium:fal:ad-novitatem", "Appeal to Novelty", "relevance",
  "New equals better (version 11 must be superior).",
  ["appeal to novelty"]),
 ("trivium:fal:ad-antiquitatem", "Appeal to Tradition", "relevance",
  "Old equals correct (always curved grades; change is wrong).",
  ["appeal to tradition"]),
 ("trivium:fal:ad-consequentiam", "Appeal to Consequences", "relevance",
  "Truth judged by welcome outcomes (free will must exist or morality collapses).",
  ["appeal to consequences"]),
 ("trivium:fal:tu-quoque", "Tu Quoque", "relevance",
  "Deflect by accusing accuser (you litter too, so no criticism).",
  ["tu quoque", "you too"]),
 ("trivium:fal:whataboutism", "Whataboutism", "relevance",
  "Change subject to opponent's worse act (our emissions vs Country X).",
  ["whataboutism"]),
 ("trivium:fal:guilt-association", "Guilt by Association", "relevance",
  "Discredit by company (spoke at their rally, research tainted).",
  ["guilt by association"]),
 ("trivium:fal:honor-association", "Honor by Association", "relevance",
  "Credit by company (trained with Nobelists so claim true).",
  ["honor by association"]),
 ("trivium:fal:genetic", "Genetic Fallacy", "relevance",
  "Judge claim by origin (enemy-nation math rejected).",
  ["genetic fallacy", "origin fallacy"]),
 ("trivium:fal:poisoning-well", "Poisoning the Well", "relevance",
  "Pre-emptive smear (know he's a liar before he speaks).",
  ["poisoning the well"]),
 ("trivium:fal:burden-shifting", "Burden Shifting", "relevance",
  "Demand opponent disprove (prove ghosts don't exist). Proof by assertion.",
  ["burden shifting", "shifting burden of proof"]),
 ("trivium:fal:complex-question", "Complex Question", "presumption",
  "Loaded multi-part question (stopped cheating? yes/no both admit).",
  ["complex question", "loaded question", "plurium interrogationum"]),
 ("trivium:fal:false-cause", "False Cause", "presumption",
  "Cause asserted without warrant (lucky socks won the game).",
  ["false cause", "non causa pro causa"]),
 ("trivium:fal:post-hoc", "Post Hoc", "presumption",
  "After equals because (rooster crows, sun rises; rooster causes sunrise).",
  ["post hoc", "post hoc ergo propter hoc"]),
 ("trivium:fal:cum-hoc", "Cum Hoc", "presumption",
  "Correlation equals causation (ice cream and drownings rise together).",
  ["cum hoc", "correlation causation"]),
 ("trivium:fal:texas-sharpshooter", "Texas Sharpshooter", "presumption",
  "Cherry-pick cluster, draw target around it (5 cured of 100, claim cure).",
  ["texas sharpshooter", "cherry picking"]),
 ("trivium:fal:regression", "Regression Fallacy", "presumption",
  "Mistake return-to-mean for intervention effect (yelled after bad game, next better).",
  ["regression fallacy", "regression to mean"]),
 ("trivium:fal:gamblers", "Gambler's Fallacy", "presumption",
  "Past independents affect odds (five reds, black is due).",
  ["gamblers fallacy", "monte carlo fallacy"]),
 ("trivium:fal:accident", "Accident (dicto simpliciter)", "presumption",
  "Apply general rule to exceptional case (cutting wrong, so surgeons wrong).",
  ["accident fallacy", "sweeping generalization rule"]),
 ("trivium:fal:converse-accident", "Converse Accident", "presumption",
  "Exception to rule (one helpful tax cut, so all cuts always help).",
  ["converse accident", "hasty generalization exception"]),
 ("trivium:fal:false-analogy", "False Analogy", "presumption",
  "Weak comparison as strong (brain is computer so sleep is reboot).",
  ["false analogy", "weak analogy"]),
 ("trivium:fal:slothful-induction", "Slothful Induction", "presumption",
  "Deny obvious inductive conclusion (smoking link denied despite data).",
  ["slothful induction", "inductive denial"]),
 ("trivium:fal:ad-ignorantiam", "Appeal to Ignorance", "presumption",
  "Lack of proof equals proof (none disproved pyramid aliens).",
  ["appeal to ignorance", "ad ignorantiam"]),
 ("trivium:fal:amphiboly", "Amphiboly", "ambiguity",
  "Grammar allows two readings (duke marries princess — who announced?).",
  ["amphiboly", "syntactic ambiguity"]),
 ("trivium:fal:accent", "Accent Fallacy", "ambiguity",
  "Meaning shifted by stress or cropped quotation (WE forbid vs We FORBID).",
  ["accent fallacy", "emphasis shift"]),
 ("trivium:fal:composition", "Composition", "ambiguity",
  "Parts to whole (each brick light, so wall light).",
  ["fallacy of composition"]),
 ("trivium:fal:division", "Division", "ambiguity",
  "Whole to parts (champion team, so each player champion).",
  ["fallacy of division"]),
 ("trivium:fal:reification", "Reification", "ambiguity",
  "Abstract as concrete agent (Science says; History will judge).",
  ["reification", "hypostatization"]),
 ("trivium:fal:etymological", "Etymological Fallacy", "ambiguity",
  "Original meaning equals true meaning (decimate is 1/10, never devastate).",
  ["etymological fallacy"]),
 ("trivium:fal:weasel", "Weasel Words", "ambiguity",
  "Strategic fuzz (helps possibly reduce up to 50% — unfalsifiable).",
  ["weasel words", "vagueness exploit"]),
 ("trivium:fal:ecological", "Ecological Fallacy", "presumption",
  "Group average applied to individual (district 80% literate, so this child reads).",
  ["ecological fallacy"]),
 ("trivium:fal:atomistic", "Atomistic Fallacy", "presumption",
  "Single datum explains system (one gene explains intelligence). Converse of ecological.",
  ["atomistic fallacy", "reductionist fallacy"]),
 ("trivium:fal:base-rate", "Base-Rate Neglect", "presumption",
  "Ignore priors for vivid case (95% test on 0.1% disease: positive still unlikely ill).",
  ["base rate neglect", "base rate fallacy"]),
 ("trivium:fal:survivorship", "Survivorship Bias", "presumption",
  "Only survivors counted (study winners, conclude dropout wins; the dead invisible).",
  ["survivorship bias"]),
]

SOCRATIC = [
 ("clarification", "What exactly do you mean by {term}?", "Forces grammar-stage disambiguation before logic runs."),
 ("clarification", "Restate that in different words, or give an example.", "Tests whether the claim survives paraphrase."),
 ("clarification", "What distinction are you drawing between {a} and {b}?", "Surfaces hidden category boundaries."),
 ("clarification", "How does this relate to the question we are asking?", "Relevance check against the quaestio."),
 ("clarification", "Why does this question matter — what hangs on it?", "Stakes check; routes effort."),
 ("assumption", "What are you assuming that might be questioned?", "Direct assumption probe."),
 ("assumption", "What would have to be true for {claim} to hold?", "Surfaces enthymematic premises for validation."),
 ("assumption", "Are you treating {x} as fixed when it could vary?", "Rigidity check on parameters."),
 ("assumption", "What alternative assumptions explain the same facts?", "Underdetermination probe."),
 ("assumption", "How would a skeptic frame your starting point?", "Adversarial reframe."),
 ("evidence", "What evidence supports that, and how strong is it?", "Evidence grading trigger."),
 ("evidence", "How could we test that — what would falsify it?", "Falsifiability demand."),
 ("evidence", "Fact, inference, or opinion — and how do you tell?", "Epistemic sorting."),
 ("evidence", "What is the source, and why trust it here?", "Source reliability probe."),
 ("evidence", "Are you ignoring base rates or counter-examples?", "Base-rate / selection probe."),
 ("perspective", "How would an informed opponent object?", "Charity check; blocks strawman."),
 ("perspective", "Steelman the strongest case against your view.", "Strongest-objection construction."),
 ("perspective", "How does this look from the opposite premise?", "Inversion test."),
 ("perspective", "Whose voice or data is missing here?", "Completeness probe."),
 ("perspective", "Where might bias shape this judgment?", "Bias self-check."),
 ("implication", "If that is true, what follows necessarily?", "Deductive closure test."),
 ("implication", "Short-term vs long-term consequences of acting on this?", "Temporal scope test."),
 ("implication", "Does this commit you to {parallel} in a parallel case?", "Consistency stress test."),
 ("implication", "What happens at the extreme or boundary case?", "Boundary stress test."),
 ("implication", "What would change your mind, and at what threshold?", "Revisability demand."),
 ("meta", "Which question are we really answering — is it the right one?", "Quaestio audit."),
 ("meta", "What kind of reasoning fits: deduction, induction, abduction?", "Method selection (links to forms)."),
 ("meta", "Are we confusing necessary with sufficient conditions?", "Condition-direction check."),
 ("meta", "Is our language doing hidden work (definition, ambiguity, scope)?", "Language audit."),
 ("meta", "What did we learn about our method, not just our conclusion?", "Meta-learning capture."),
]

DEVICES = [
 ("Anaphora", "scheme", "Repeat openings across clauses (We parse... We validate... We compose...). Use for enumerating steps.",
  ["anaphora"]),
 ("Epistrophe", "scheme", "Repeat endings (of the people, by the people, for the people). Closes arguments with cadence.",
  ["epistrophe"]),
 ("Antithesis", "scheme", "Balanced opposites (best of times, worst of times). Corrects without hostility.",
  ["antithesis"]),
 ("Chiasmus", "scheme", "ABBA reversal (ask not what your country can do for you...). Memorable framing.",
  ["chiasmus"]),
 ("Tricolon", "scheme", "Triple cadence for completeness (parse, prove, persuade). Reserve for closings.",
  ["tricolon", "rule of three"]),
 ("Hypophora", "scheme", "Ask then answer (What makes a good school? Teachers who listen.). Controls objections.",
  ["hypophora"]),
 ("Procatalepsis", "scheme", "Prebuttal (you will say it costs too much — yet ignorance costs more).",
  ["procatalepsis", "prebuttal"]),
 ("Aporia", "scheme", "Feigned doubt (at a loss where to begin — so vast...). Earnest emphasis.",
  ["aporia"]),
 ("Litotes", "trope", "Affirm by denying opposite (not unwise = wise). Understated precision.",
  ["litotes"]),
 ("Paralipsis", "scheme", "Mention by claiming to omit (I will not mention scandals). Use sparingly; it shows.",
  ["paralipsis", "apophasis"]),
 ("Hyperbole", "trope", "Deliberate overstatement (told you a thousand times). Never in verification output.",
  ["hyperbole"]),
 ("Asyndeton", "scheme", "Omit conjunctions (came, saw, conquered). Speed and force.",
  ["asyndeton"]),
 ("Polysyndeton", "scheme", "Multiply conjunctions (night and day and storm). Weight and endurance.",
  ["polysyndeton"]),
 ("Anadiplosis", "scheme", "End-to-start chaining (fear leads to anger; anger to hate). Shows causal flow.",
  ["anadiplosis"]),
 ("Zeugma", "scheme", "One verb governing mismatched objects (stole my heart and wallet). Wit; avoid in formal claims.",
  ["zeugma"]),
 ("Oxymoron", "trope", "Compressed contradiction (deafening silence). Flags nuance.",
  ["oxymoron"]),
 ("Irony", "trope", "Say opposite of intent (a fine mess). Risky in text; pair with plain gloss.",
  ["irony", "verbal irony"]),
 ("Metonymy", "trope", "Attribute for entity (the crown decreed). Compact reference.",
  ["metonymy"]),
 ("Synecdoche", "trope", "Part for whole (all hands on deck). Concrete shorthand.",
  ["synecdoche"]),
 ("Metaphor", "trope", "Compressed analogy (fallacy as type-error). Always pair with plain gloss in technical output.",
  ["metaphor"]),
 ("Simile", "trope", "Explicit comparison (like/mind as...). Safer than metaphor; mark the mapping.",
  ["simile"]),
 ("Rhetorical Question", "scheme", "Question as assertion (shall we accept decline? No.). Engages without evidence burden — never as proof.",
  ["rhetorical question", "erotema"]),
 ("Understatement", "trope", "Deliberate minimization (slight inconvenience, house destroyed). Dry emphasis.",
  ["understatement"]),
 ("Parallelism", "scheme", "Matched structures across clauses. Default skeleton for lists and comparisons.",
  ["parallelism"]),
 ("Exemplum", "proof", "Concrete illustrative case as evidence. Must be representative, not cherry-picked (see Texas sharpshooter).",
  ["exemplum", "example proof"]),
 ("Testimony", "proof", "Quoted authority or witness as support. Weight by expertise + independence, per evidence stems.",
  ["testimony", "ethos proof"]),
 ("Enthymeme", "proof", "Compressed syllogism the audience completes. Powerful but hides premises — flag for logic check.",
  ["enthymeme proof", "rhetorical syllogism"]),
 ("Kairos", "proof", "Timeliness framing (this moment demands...). Legitimate urgency vs manufactured pressure.",
  ["kairos", "timeliness"]),
 ("Allusion", "trope", "Indirect reference (a Sisyphean task). Requires shared canon; gloss for mixed audiences.",
  ["allusion"]),
]

PROGYM = [
 ("Lv01 Fable", "Retell a fable in own words with moral explicit. Trains paraphrase = grammar mastery gate."),
 ("Lv02 Narrative", "Clear recounting (myth/history/event) with 5W completeness. Trains sequence and concision."),
 ("Lv03 Chreia", "Expand a saying/deed in 8 moves: praise, paraphrase, cause, contrast, example, testimony, epilogue. Core chat-answer template."),
 ("Lv04 Maxim", "Expand a general saying without named agent. Trains universalizing and proof by reason."),
 ("Lv05 Refutation", "Overturn a narrative (obscure, implausible, inconsistent). Trains verification mindset."),
 ("Lv06 Confirmation", "Defend the same narrative. Trains evidence-building; pair with refutation for both sides."),
 ("Lv07 Commonplace", "Attack a generic vice. Trains amplification and ethical argument."),
 ("Lv08 Encomium", "Praise person/thing by origin, deeds, virtues, comparison. Trains epideictic structure."),
 ("Lv09 Invective", "Blame as mirror of encomium. Trains proportional censure without abuse."),
 ("Lv10 Comparison", "Parallel two persons/things, prefer one. Trains balanced judgment by criteria."),
 ("Lv11 Impersonation", "Speak in character (what would X say?). Trains voice, empathy, decorum."),
 ("Lv12 Description", "Vivid scene rendering (enargeia). Trains sensory detail and order."),
 ("Lv13 Thesis", "Argue a general question pro/contra without proper names. Capstone deliberative logic."),
 ("Lv14 Law", "Propose or oppose a statute (justice/utility). Legislative capstone combining all."),
]

CANONTECH = [
 ("canontech:topics", "Topics / Loci Method", "canon",
  "Run {issue} through definition, genus/species, cause/effect, testimony checklist. Guarantees inventio coverage.",
  ["loci", "topics method"]),
 ("canontech:stasis", "Stasis Theory", "canon",
  "Locate disagreement: fact (did it happen?), definition (what is it?), quality (was it right?), jurisdiction. Argue at the right stasis.",
  ["stasis", "status theory"]),
 ("canontech:nestorian", "Nestorian Order", "canon",
  "Strongest proof last, second-strongest first. Dispositio rule for arrangement under hostile audiences.",
  ["nestorian order", "arrangement rule"]),
 ("canontech:virtues", "Virtues of Style", "canon",
  "Correctness, clarity, ornament, propriety to audience/register. Elocutio checklist before any polish pass.",
  ["style virtues", "elocutio checklist"]),
 ("canontech:outline", "Retrievable Outline", "canon",
  "Memoria for AI: outline + citations stored retrievably, never regenerated from vibes. Every claim traceable.",
  ["outline memory", "citation bundle"]),
 ("canontech:register", "Register Adaptation", "canon",
  "Actio for AI: formatting, pacing, tone matched to channel (chat terse, report formal). Audience profile lookup first.",
  ["register", "tone adaptation"]),
 ("canontech:prebuttal", "Prebuttal Placement", "canon",
  "Strongest objection answered BEFORE the close (procatalepsis positioned post-confirmatio). Defuses refutatio in advance.",
  ["prebuttal", "anticipated objection"]),
 ("canontech:chreia-template", "Chreia Answer Template", "canon",
  "Default chat shape: claim, cause, contrast, example, testimony, close. Full inventio+dispositio in six moves.",
  ["chreia template", "answer shape"]),
]

GATES = [
 ("trivium:gate:grammar-complete", "No Logic Without Grammar",
  "IF every key term in {draft} has 1 definition + 1 alias resolved ELSE block inference, return clarification stems. Check: terms_undefined == 0.",
  ["grammar gate", "define first"]),
 ("trivium:gate:logic-complete", "No Rhetoric Without Logic",
  "IF {argument} passes form check + zero high-severity fallacies ELSE block stylistic polish, run refutation pass. Check: fallacy_sev_high == 0.",
  ["logic gate", "prove before polish"]),
 ("trivium:gate:evidence-grounded", "No Claim Without Ground",
  "IF every factual claim has citation or commonplace tag ELSE downgrade to conjecture. Check: claims_uncited == 0.",
  ["citation gate", "grounding rule"]),
 ("trivium:gate:audience-fit", "Register Fit Check",
  "IF device density mismatched to {audience} THEN re-run style with simpler scheme set. Check: audience profile lookup.",
  ["audience gate", "propriety check"]),
]

META = [
 ("trivium:meta:inventory", "Inventory Check", "monitor",
  "What do I know / not know / need to check about {task}? Run at plan start; route unknowns to retrieval.",
  ["kwl", "inventory check"]),
 ("trivium:meta:term-drift", "Term-Drift Monitor", "monitor",
  "Did {term} change meaning between step N and M? Run on embedding drift or alias collision.",
  ["equivocation self-check", "term drift"]),
 ("trivium:meta:objection", "Adversarial Self-Test", "repair",
  "What is the strongest reason my {draft} is wrong? Mandatory pre-finalize; triggers refutation exercise.",
  ["red team self", "strongest objection"]),
 ("trivium:meta:method-fit", "Method-Fit Check", "monitor",
  "Does this task need deduction, induction, or abduction? Wrong method is the commonest silent failure.",
  ["method fit", "reasoning mode check"]),
]


def _slug(s):
    import re as _re
    return _re.sub(r"-+", "-", "".join(
        c.lower() if c.isalnum() else "-" for c in s)).strip("-")


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

    for cid, et, tf, dn, sh, desc, als in FORMS:
        add(cid, "logical_form", tf, dn, sh, desc, als)
    for cid, dn, fam, desc, als in FALLACIES:
        add(cid, "fallacy", fam, dn, f"trivium:logic:fallacy:{fam}", desc, als)
    for i, (purpose, stem, why) in enumerate(SOCRATIC):
        slug = _slug(stem[:40])
        add(f"trivium:soc:{purpose}:{i:02d}", "socratic_stem", purpose,
            f"Socratic Stem ({purpose}): {stem[:60]}",
            f"trivium:logic:socratic:{purpose}",
            f"{stem} — {why}", [stem.lower()[:60], purpose])
    for i, (name, fam, desc, als) in enumerate(DEVICES):
        add(f"trivium:dev:{_slug(name)}", "rhet_device", fam, f"{name} [{fam}]",
            f"trivium:rhetoric:device:{fam}", desc, als)
    for i, (dn, desc) in enumerate(PROGYM):
        lv = dn.split()[0]
        add(f"trivium:prog:{lv.lower()}", "progym_exercise", "progym", dn,
            "trivium:rhetoric:progym:ladder", f"{dn}: {desc}",
            [dn.lower(), lv.lower()])
    for cid, dn, tf, desc, als in CANONTECH:
        canon = cid.split(":")[1] if ":" in cid else "general"
        add(cid, "canon_technique", tf, dn, f"trivium:rhetoric:canon:{canon}",
            desc, als)
    for cid, dn, desc, als in GATES:
        add(cid, "gate_rule", "gate", dn, "trivium:gates:stage", desc, als)
    for cid, dn, tf, desc, als in META:
        add(cid, "meta_prompt", tf, dn, f"trivium:meta:{tf}", desc, als)
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
    print(f"populate_trivium2: {_e} entities, {_a} aliases")
    _conn.close()
