"""Programming track seed (phase 88): grammar/logic/rhetoric/standards for code.

~165 rows from tank research: language-family grammar, universal grammar
(HTTP/git/DS/numbers/regex/testing), debug/complexity/review/bugs/design,
clean/docs/git/PR/writing/naming/teaching, standards (format/error/test/
security/perf), code-review Socratic stems, 12 fallacy-link notes tying
existing fallacy rows to programmer contexts. Idempotent.
Run: populate_trivium4.py [db-path].
"""
import sys

sys.path.insert(0, "A:/scripts/TheBrain")

SHARD = "trivium:field:programming"

# (TAG, display, description, aliases) — TAG prefix selects family.
ROWS = [
# ---- language grammar ----
("GRAMMAR_PYTHON", "Python significant indentation", "Blocks are defined by 4-space indentation, not braces; mixing tabs and spaces raises TabError.", ["python-indent", "whitespace-syntax"]),
("GRAMMAR_PYTHON", "Python dynamic strong typing", 'Variables are untyped references; "5"+5 raises TypeError — no implicit coercion unlike JS.', ["python-typing", "duck-typing"]),
("GRAMMAR_PYTHON", "Python GIL and venv/pip", "CPython threads share the GIL so CPU-bound code needs multiprocessing; isolate deps per project with venv plus pip.", ["gil", "venv-pip"]),
("GRAMMAR_PYTHON", "Python comprehensions and slicing", "List comprehensions build declaratively; slices use half-open [start:stop:step].", ["comprehension-slice"]),
("GRAMMAR_PYTHON", "Python exceptions and context managers", "Use try/except/else/finally and with-blocks to guarantee resource cleanup.", ["with-statement"]),
("GRAMMAR_JS", "JS var/let/const and hoisting", "var is function-scoped and hoisted; prefer block-scoped let/const (const blocks rebinding, not mutation).", ["let-const-hoisting"]),
("GRAMMAR_JS", "JS triple-equals rule", "== coerces (0=='' is true); use ===/!== except intentional null-check x==null.", ["triple-equals"]),
("GRAMMAR_JS", "JS event loop and promises", "Single-threaded loop with microtask queue; await yields without blocking; unhandled rejection crashes Node.", ["event-loop-async"]),
("GRAMMAR_JS", "TypeScript narrowing", "Types erase at runtime; use unknown plus typeof/in narrowing before use.", ["typescript-narrowing"]),
("GRAMMAR_JS", "ESM modules and npm scripts", "import/export modules; entry plus scripts in package.json; lock with package-lock.json.", ["esm-npm"]),
("GRAMMAR_SQL", "SQL clause order", "SELECT FROM WHERE GROUP BY HAVING ORDER BY LIMIT; WHERE filters rows, HAVING filters groups.", ["sql-clause-order"]),
("GRAMMAR_SQL", "SQL JOIN types", "INNER keeps matches; LEFT keeps all left rows with NULLs; CROSS is cartesian product.", ["sql-joins"]),
("GRAMMAR_SQL", "SQL NULL three-valued logic", "NULL=NULL is UNKNOWN not TRUE; test with IS NULL; WHERE keeps only TRUE rows.", ["sql-null"]),
("GRAMMAR_SQL", "SQL transactions ACID", "Wrap multi-statement writes in BEGIN/COMMIT with ROLLBACK on error; pick isolation level deliberately.", ["acid-transactions"]),
("GRAMMAR_SQL", "SQL indexes and EXPLAIN", "Indexes speed selective lookups but slow writes; verify with EXPLAIN ANALYZE before assuming.", ["sql-index-explain"]),
("GRAMMAR_SYSTEMS", "Rust ownership and borrows", "One owner per value; either one &mut or many & borrows; violations are compile errors.", ["rust-ownership"]),
("GRAMMAR_SYSTEMS", "Rust Result/Option discipline", "Fallible ops return Result/Option; handle with match/? — never unwrap in production.", ["rust-result-option"]),
("GRAMMAR_SYSTEMS", "Go goroutines and channels", "go f() spawns lightweight goroutines; communicate via chan; guard sharing with Mutex/WaitGroup.", ["go-concurrency"]),
("GRAMMAR_SYSTEMS", "Go explicit error returns", "Functions return (T, error); check err != nil at every call — no try/catch.", ["go-errors"]),
("GRAMMAR_SYSTEMS", "Cargo and Go toolchains", "cargo build/test/clippy; go build/test/vet plus gofmt; both pin deps via lockfiles.", ["cargo-go-toolchain"]),
("GRAMMAR_WEB", "Semantic HTML", "Use header/main/nav/article/section/footer plus alt text; div-only pages destroy accessibility and SEO.", ["semantic-html"]),
("GRAMMAR_WEB", "CSS specificity order", "Inline beats ID beats class beats element; later equal rules win; prefer classes over !important.", ["css-specificity"]),
("GRAMMAR_WEB", "Box model plus flex/grid", "Content+padding+border+margin with border-box; Flexbox for 1-D, Grid for 2-D.", ["box-model-flex-grid"]),
("GRAMMAR_WEB", "Responsive units", "Relative rem/%/vw plus media breakpoints; fixed px widths break zoom and mobile.", ["responsive-css"]),
("GRAMMAR_WEB", "Form labels and server validation", "Label every input; use type/required/pattern; always re-validate server-side.", ["form-validation"]),
# ---- universal grammar ----
("UNIV_HTTP", "HTTP safe/idempotent matrix", "GET/HEAD/OPTIONS are safe; PUT/DELETE idempotent, POST/PATCH not; retry only idempotent calls.", ["http-methods"]),
("UNIV_HTTP", "HTTP status classes", "2xx success, 3xx redirect, 4xx fix request, 5xx retry/backoff; 429 respects Retry-After.", ["status-codes"]),
("UNIV_HTTP", "REST resource semantics", "Nouns as paths, verbs as methods, stateless requests, cursor pagination, header or /v1 versioning.", ["rest-design"]),
("UNIV_HTTP", "Auth headers and caching", "Authorization Bearer, JSON content type, ETag/If-None-Match and Cache-Control for freshness.", ["http-headers"]),
("UNIV_HTTP", "Cookies versus tokens", "HttpOnly Secure SameSite cookies resist XSS but need CSRF tokens; short-lived JWT Bearer suits APIs.", ["cookies-jwt"]),
("UNIV_HTTP", "TLS plus idempotency keys", "HTTPS encrypts transit; POST mutations need Idempotency-Key headers for safe retries.", ["tls-idempotency"]),
("UNIV_GIT", "Git stage-commit-push model", "Working dir to stage to snapshot to publish; history is a commit DAG.", ["git-model"]),
("UNIV_GIT", "Branching: merge vs rebase", "Branches isolate work; rebase linearizes private history, never shared branches.", ["branch-rebase"]),
("UNIV_GIT", "Pull is fetch plus merge", "git pull fetches then merges; prefer pull --rebase on private branches.", ["git-pull"]),
("UNIV_GIT", "Stash, reset, revert", "Stash shelves work; reset --hard discards (destructive); revert safely undoes published commits.", ["stash-reset-revert"]),
("UNIV_GIT", "Atomic commits and gitignore", "One logical change per message; never commit secrets or build output; list them in .gitignore.", ["atomic-commits"]),
("UNIV_GIT", "Conflict marker discipline", "Edit conflict markers to intended result, stage, complete — never commit markers.", ["merge-conflicts"]),
("UNIV_DS", "Array Big-O", "Indexed O(1), append amortized O(1), middle insert/delete O(n); cache-friendly.", ["array-complexity"]),
("UNIV_DS", "Linked list Big-O", "Insert/delete at known node O(1), search O(n); pointer overhead, poor locality.", ["linked-list"]),
("UNIV_DS", "Hash map Big-O", "Average O(1); worst O(n) on collision attack; needs good hash plus resize.", ["hashmap"]),
("UNIV_DS", "Balanced BST Big-O", "Sorted ops O(log n), inorder O(n); red-black/AVL.", ["bst-balanced"]),
("UNIV_DS", "Heap Big-O", "Peek O(1), push/pop O(log n); schedulers and top-K, not search.", ["heap-pq"]),
("UNIV_DS", "Stack and queue Big-O", "O(1) ends; LIFO for frames/undo, FIFO for BFS/job queues.", ["stack-queue"]),
("UNIV_DS", "Trie Big-O", "Prefix ops O(k); ideal autocomplete, heavy for sparse keys.", ["trie"]),
("UNIV_DS", "Graph adjacency Big-O", "List: O(V+E) space/time BFS/DFS; matrix O(1) edge check, O(V^2) space.", ["graph-representation"]),
("UNIV_DS", "Bloom filter rule", "O(k) add/query, no false negatives, possible false positives; never for exact membership.", ["bloom-filter"]),
("UNIV_DS", "LRU cache pattern", "HashMap plus doubly-linked list: O(1) get/put with eviction for bounded memoization.", ["lru-cache"]),
("UNIV_NUM", "Two's complement", "Integers are base-2; negatives are complement so -1 is all-ones bits.", ["twos-complement"]),
("UNIV_NUM", "Float epsilon rule", "0.1+0.2 != 0.3 in binary float; never == floats, use epsilon; Decimal/int-cents for money.", ["float-epsilon"]),
("UNIV_NUM", "UTF-8 vs ASCII vs Base64", "ASCII is 7-bit subset; UTF-8 variable 1-4 bytes; Base64 encodes bytes as ASCII at 4/3 size.", ["utf8-base64"]),
("UNIV_NUM", "Overflow and endianness", "Little-endian x86 stores LSB first; fixed-width ints wrap — check bounds.", ["endianness-overflow"]),
("UNIV_NUM", "UTC epoch storage", "Store UTC ISO-8601 or Unix epoch; apply timezones only at display to dodge DST bugs.", ["utc-epoch"]),
("UNIV_REGEX", "Regex anchors and classes", "Anchor full matches; \\d \\w \\s with + * ? {n,m}; default greedy.", ["regex-basics"]),
("UNIV_REGEX", "Capturing vs non-capturing", "(a|b) captures, (?:a|b) does not; extract $1, name (?<year>\\d{4}).", ["regex-groups"]),
("UNIV_REGEX", "Greedy vs lazy, escaping", ".* maximal, .*? minimal; escape . * + ? ( ) [ ].", ["greedy-lazy"]),
("UNIV_REGEX", "Lookaround assertions", "(?=...) and (?<!...) assert context without consuming characters.", ["lookaround"]),
("UNIV_REGEX", "Catastrophic backtracking guard", "Nested quantifiers hang on hostile input; use atomic groups or length pre-check.", ["redos"]),
("UNIV_TEST", "Test pyramid levels", "Unit isolated/fast, integration at boundaries, e2e full-stack slow and brittle.", ["test-pyramid"]),
("UNIV_TEST", "Mock stub spy fixture", "Stub returns canned data, mock asserts calls, spy records, fixture builds state.", ["test-doubles"]),
("UNIV_TEST", "AAA assertion pattern", "Arrange, Act, Assert one outcome per test with a behavior-revealing name.", ["aaa-pattern"]),
("UNIV_TEST", "Flaky test rule", "Non-deterministic pass/fail from timing/order; quarantine and fix, never silent-skip.", ["flaky-test"]),
("UNIV_TEST", "Coverage honesty", "Coverage measures executed lines, not correctness; 100% without assertions proves nothing.", ["coverage-honesty"]),
("UNIV_TEST", "TDD red-green-refactor", "Failing test first, minimal pass, then clean up keeping green.", ["tdd-cycle"]),
# ---- logic ----
("LOGIC_DEBUG", "Reproduce deterministically", "Capture inputs/env/seed and script it; non-reproducible bugs are non-fixable.", ["reproduce-bug"]),
("LOGIC_DEBUG", "Bisect to isolate", "Halve code, data, or history (git bisect) to the minimal failing case.", ["bisect-isolate"]),
("LOGIC_DEBUG", "Hypothesize then falsify", "State one mechanism, run the cheapest discriminating experiment; kill hypotheses.", ["debug-hypothesis"]),
("LOGIC_DEBUG", "Read stacks top-down", "Top frame is failure site, bottom is entry; fix root throw, add context per re-throw.", ["stack-trace-reading"]),
("LOGIC_DEBUG", "Rubber-duck and binary logging", "Explain aloud line by line; temporary binary-search prints removed before commit.", ["rubber-duck"]),
("LOGIC_DEBUG", "Verify fix with regression test", "Re-run repro plus suite; lock fix with a test failing without the patch.", ["regression-verify"]),
("LOGIC_COMP", "Worst-case Big-O rule", "Report upper-bound growth for large n; state variable, worst vs average.", ["big-o-definition"]),
("LOGIC_COMP", "Dominant term only", "O(2n+50) is O(n); constants matter below ~10k inputs — benchmark real sizes.", ["dominant-term"]),
("LOGIC_COMP", "Amortized vs worst-case", "Append O(1) amortized despite O(n) resize; realtime paths budget the worst case.", ["amortized-analysis"]),
("LOGIC_COMP", "Time-space tradeoff", "Memoization trades memory for recompute; streaming trades CPU for O(1) memory.", ["time-space-tradeoff"]),
("LOGIC_COMP", "NP-hard recognition", "Exponential exact solutions need heuristics past n~30 — never brute-force.", ["np-hard-heuristic"]),
("LOGIC_COMP", "Profile before optimizing", "Flamegraph/benchmark first; complexity wins beat micro-opts by orders of magnitude.", ["profile-first"]),
("LOGIC_REVIEW", "Review happy plus edge", "Empty, null, zero, one, max, unicode, concurrent — check all.", ["review-correctness"]),
("LOGIC_REVIEW", "Review error paths", "Every fallible call checked, errors carry context, nothing swallowed.", ["review-errors"]),
("LOGIC_REVIEW", "Review resource leaks", "Files/sockets/locks closed in finally/with; no unbounded caches.", ["review-resources"]),
("LOGIC_REVIEW", "Review injection surface", "SQL/HTML/shell via parameters/escaping, never interpolation of user input.", ["review-injection"]),
("LOGIC_REVIEW", "Review authZ per route", "Authentication plus object-ownership checks, not just login.", ["review-authz"]),
("LOGIC_REVIEW", "Review concurrency safety", "Shared state guarded, no TOCTOU, idempotent retries, correct isolation.", ["review-concurrency"]),
("LOGIC_REVIEW", "Review API stability", "No breaking renames without versioning; backwards-compatible defaults.", ["review-api-contract"]),
("LOGIC_REVIEW", "Review test quality", "Failing-without-fix tests asserting behavior, no sleep-based sync.", ["review-tests"]),
("LOGIC_REVIEW", "Review naming readability", "Intent-revealing names, small single-purpose functions, no magic numbers.", ["review-readability"]),
("LOGIC_REVIEW", "Review performance budget", "No N+1, no O(n^2) on unbounded input, pagination and matching indexes.", ["review-performance"]),
("LOGIC_REVIEW", "Review PII-safe logging", "Correlation IDs in, passwords/tokens/PII out; correct levels.", ["review-logging"]),
("LOGIC_REVIEW", "Review secrets handling", "Env/vault secrets, safe defaults, flag cleanup plans.", ["review-config"]),
("LOGIC_REVIEW", "Review a11y basics", "Keyboard reachable, labels/alt, contrast, visible focus.", ["review-a11y"]),
("LOGIC_REVIEW", "Review docs updated", "README/API/migration notes move with behavior change.", ["review-docs"]),
("LOGIC_REVIEW", "Review diff scope", "Under ~400 lines, one concern; refactor split from behavior.", ["review-scope"]),
("LOGIC_REVIEW", "Review hallucinated APIs", "Every new library call verified against docs; no invented flags.", ["review-hallucination"]),
("LOGIC_BUG", "Off-by-one boundary", "Test empty, single, last, len+1 cases for every boundary loop.", ["off-by-one"]),
("LOGIC_BUG", "Null dereference", "Null-check every boundary; enable strict null checks.", ["null-deref"]),
("LOGIC_BUG", "Integer wraparound", "Test MAX/MIN and fuzz large inputs on fixed-width counters.", ["int-overflow"]),
("LOGIC_BUG", "Float equality", "Search == on floats; replace with epsilon or Decimal.", ["float-compare"]),
("LOGIC_BUG", "Race interleaving", "Stress with threads plus sanitizer/race detector on shared read-modify-write.", ["race-condition"]),
("LOGIC_BUG", "Deadlock ordering", "Lock hierarchy plus timeouts; acquire in consistent global order.", ["deadlock"]),
("LOGIC_BUG", "Injection concatenation", "Probe with quote-break payloads; fix with prepared statements.", ["injection"]),
("LOGIC_BUG", "Unescaped output XSS", "Submit script tags; fix with autoescaping and CSP.", ["xss"]),
("LOGIC_BUG", "N+1 query storm", "Assert query counts; fix with joins/eager loading.", ["n-plus-one"]),
("LOGIC_BUG", "Timezone DST shift", "Test DST transitions; store UTC only.", ["timezone-bug"]),
("LOGIC_BUG", "Retry thundering herd", "Kill downstream in chaos test; require backoff+jitter+breaker.", ["retry-storm"]),
("LOGIC_BUG", "Stale cache", "Write-then-read tests with TTL eviction checks; version keys.", ["cache-stale"]),
("LOGIC_BUG", "Secrets in code", "Secret-scan CI plus log review for password/token/key.", ["secret-leak"]),
("LOGIC_BUG", "Dependency drift", "Clean-install plus lockfile diff in CI; pin and audit.", ["version-drift"]),
("LOGIC_DESIGN", "Tradeoff matrix", "Score latency/cost/complexity/operability; reversible doors fast, irreversible slow.", ["tradeoff-matrix"]),
("LOGIC_DESIGN", "YAGNI with exception", "No speculative generality; except when later change costs 10x (schema, public API).", ["yagni"]),
("LOGIC_DESIGN", "KISS sufficient", "Boring readable solution meeting budget; document why when complex.", ["kiss-principle"]),
("LOGIC_DESIGN", "DRY third-use rule", "Duplicate twice, abstract on third with stable interface; never abstract volatile single-use.", ["dry-rule"]),
("LOGIC_DESIGN", "Composition over inheritance", "Small composable modules; inheritance only for true is-a with Liskov compliance.", ["composition-inheritance"]),
("LOGIC_DESIGN", "Fail fast inside, degrade at edges", "Assert/throw internally; fallback/cache/default at edges with monitoring.", ["fail-fast-degrade"]),
("LOGIC_DESIGN", "No optimization without profile", "Profile showing 20%+ share first; algorithm, then I/O, last micro-opts.", ["premature-optimization"]),
("LOGIC_DESIGN", "Reversible decisions bias", "Flags, strangler migration, additive changes keep rewrites revertible.", ["reversible-decisions"]),
# ---- rhetoric ----
("RHET_CLEAN", "Single-purpose functions", "One verb-named function, 3 or fewer params; split when needing 'and' to describe.", ["clean-functions"]),
("RHET_CLEAN", "Guard clauses", "Return early on invalid cases; happy path left-aligned, one nesting level max.", ["guard-clauses"]),
("RHET_CLEAN", "Named constants", "Replace literals with named constants carrying unit and origin (MAX_RETRIES=3).", ["no-magic-values"]),
("RHET_CLEAN", "DRY intent", "Extract shared intent with stable name; tolerate text duplicated for different reasons.", ["clean-dry"]),
("RHET_CLEAN", "Why-comments", "Comments capture non-obvious why, constraints, issue links — never restate code.", ["why-comments"]),
("RHET_CLEAN", "Boy Scout rule", "One small cleanup per touch; never whole-file reformat inside a feature PR.", ["boy-scout"]),
("RHET_CLEAN", "Delete dead code", "Remove commented blocks and unused branches; history preserves them.", ["dead-code-delete"]),
("RHET_CLEAN", "Enforced formatting", "Formatter/linter in CI; style debates end at the config file.", ["autoformat"]),
("RHET_DOCS", "README quickstart", "Problem, prerequisites, commands, config, tests — runnable in under 5 minutes.", ["readme-standard"]),
("RHET_DOCS", "Docstring contract", "Purpose, args, return, raises, one example; fits one screen.", ["docstrings"]),
("RHET_DOCS", "API reference entries", "Method/path/auth/params/example request-response/error codes per endpoint.", ["api-docs"]),
("RHET_DOCS", "ADRs", "One markdown per decision: context, options, decision, consequences, date.", ["adr"]),
("RHET_DOCS", "Diagrams as code", "Mermaid beside code so refactors update both together.", ["diagrams-as-code"]),
("RHET_DOCS", "User-facing changelog", "Added/fixed/breaking per release with migration steps.", ["changelog"]),
("RHET_GIT", "Conventional commits", "feat/fix/docs/style/refactor/test/chore plus ! or BREAKING CHANGE for incompatibility.", ["conventional-commits"]),
("RHET_GIT", "Imperative subject", "50 chars max, imperative mood, body wraps at 72 explaining what/why.", ["commit-subject"]),
("RHET_GIT", "Atomic commits", "One intent, passing tests; refactor split from feat.", ["atomic-commit"]),
("RHET_GIT", "Contextual body", "Problem, approach, alternatives, Closes #123 for traceability.", ["commit-body"]),
("RHET_GIT", "No rewritten public history", "Never force-push shared branches; revert published mistakes.", ["no-force-push"]),
("RHET_PR", "PR title and issue link", "Title mirrors commit type; description links issue and states scope.", ["pr-title"]),
("RHET_PR", "PR test plan", "Behavior delta, repro steps, commands run, UI screenshots.", ["pr-test-plan"]),
("RHET_PR", "PR risk and rollback", "Migrations, flags, perf impact, exact revert command.", ["pr-risk-rollback"]),
("RHET_PR", "PR reviewable size", "Under 400 lines; stacked PRs for epics; checklist ticked first.", ["pr-size"]),
("RHET_PR", "PR self-review", "Author re-reads diff, drops debug prints, annotates subtle lines first.", ["pr-self-review"]),
("RHET_WRITE", "BLUF first", "Conclusion first, then evidence; busy readers decide on paragraph one.", ["bluf-writing"]),
("RHET_WRITE", "Active voice", "Service retries twice, not retries are performed; sentences under 25 words.", ["active-voice"]),
("RHET_WRITE", "Define terms once", "Expand acronyms on first use; keep a domain glossary.", ["define-terms"]),
("RHET_WRITE", "Numbers with context", "p95 with n, window, baseline and delta — never bare fast.", ["numbers-context"]),
("RHET_WRITE", "Scannable headings", "Task headings plus comparison tables; one idea per paragraph.", ["scannable-docs"]),
("RHET_WRITE", "Link don't paste", "Reference canonical URLs; duplicates rot.", ["link-canonical"]),
("RHET_NAME", "Intent-revealing names", "days_since_last_login beats d; booleans is_/has_; verbs for functions.", ["intent-names"]),
("RHET_NAME", "Pronounceable names", "customer_repository beats cstmrRepo2; single letters for loop indices only.", ["searchable-names"]),
("RHET_NAME", "Ubiquitous lexicon", "One word per concept project-wide; mirror the domain glossary.", ["ubiquitous-language"]),
("RHET_NAME", "Units in names", "timeout_ms, price_cents prevent unit bugs.", ["units-in-names"]),
("RHET_NAME", "Positive names", "is_ready beats is_not_ready; no double negation.", ["no-negative-names"]),
("RHET_NAME", "Threshold constants", "retries > MAX_RETRIES beats > 3; name cites the source.", ["constant-names"]),
("RHET_NAME", "Spec-style test names", "test_rejects_expired_token documents behavior.", ["test-naming"]),
("RHET_NAME", "Honest module scope", "auth/password_hasher.py beats utils/misc.py; helpers/ signals missing abstraction.", ["module-naming"]),
("RHET_TEACH", "Minimal examples", "One concept in under 20 runnable lines; learner runs then modifies.", ["minimal-example"]),
("RHET_TEACH", "Socratic debugging", "Guide with prompts to hypothesis; no keyboard-grabbing in 10 minutes.", ["socratic-debugging"]),
("RHET_TEACH", "Think-aloud pairing", "Verbalize hypothesis, experiment, observation so reasoning transfers.", ["think-aloud-pairing"]),
("RHET_TEACH", "Predict then run", "Learner predicts output before running; mismatch cements the model.", ["predict-then-run"]),
("RHET_TEACH", "Specific process praise", "Praise the method used, not the person — reinforces technique.", ["process-praise"]),
# ---- standards ----
("STD_FORMAT", "Formatter is law", "Format/lint on save and CI; reject unformatted PRs automatically.", ["formatter-law"]),
("STD_FORMAT", "Line length discipline", "~88-100 cols, one statement per line; named intermediates for long chains.", ["line-length"]),
("STD_FORMAT", "Import hygiene", "Sorted stdlib/third-party/local groups; pinned lockfiles; no unused imports.", ["import-hygiene"]),
("STD_ERROR", "Never swallow exceptions", "No bare except; log with context, chain causes, rethrow or typed error.", ["no-swallow"]),
("STD_ERROR", "Fail closed", "Default-deny on auth/validation; generic user messages, detailed server logs with ID.", ["fail-closed"]),
("STD_ERROR", "Backoff with jitter", "Retry transient idempotent ops only; exponential backoff, max attempts, deadline, breaker.", ["retry-backoff"]),
("STD_ERROR", "Validate at boundaries", "Schema-validate all external input; assert invariants internally.", ["validate-boundaries"]),
("STD_TEST", "Fast isolated units", "Pure logic under 100ms, no network/DB/sleep; run on every save.", ["unit-standard"]),
("STD_TEST", "Real-dependency integration", "Test DB/containers for repos and APIs; seeded fixtures, per-test transactions.", ["integration-standard"]),
("STD_TEST", "Coverage gating", "Gate e.g. 80% on new code plus assertion review; block silent drops.", ["coverage-gate"]),
("STD_TEST", "Deterministic clocks", "Inject clock/RNG; freeze time and seeds so CI is identical every run.", ["deterministic-tests"]),
("STD_TEST", "Mutation spot-checks", "Surviving mutants reveal weak assertions behind green suites.", ["mutation-testing"]),
("STD_SEC", "Parameterized injection defense", "All SQL/OS/LDAP parameterized; HTML autoescaped; shell via arg arrays.", ["owasp-injection"]),
("STD_SEC", "Ownership authorization", "Server-side ownership checks per object ID; IDOR-probe every route.", ["owasp-access-control"]),
("STD_SEC", "Secret storage discipline", "Argon2/bcrypt with per-user salt; KMS-encrypted secrets, rotated.", ["owasp-crypto-failures"]),
("STD_SEC", "SSRF and deserialization guard", "Allowlist fetch URLs/schemas; block metadata IPs; validate redirects.", ["owasp-ssrf-deser"]),
("STD_SEC", "Header hardening", "CSP, HttpOnly+Secure+SameSite, CSRF tokens, nosniff.", ["owasp-xss-csrf"]),
("STD_SEC", "Logging and monitoring", "Central auth-failure/error logs with alert thresholds; runbook plus SCA scans.", ["owasp-logging"]),
("STD_SEC", "Supply-chain pinning", "Lockfile pins plus hashes; audit tools; drop unused deps quarterly.", ["supply-chain"]),
("STD_SEC", "Least privilege", "Minimal service scopes; dev/prod separated; default-deny firewalls and grants.", ["least-privilege"]),
("STD_PERF", "Budget algorithms first", "SLOs first; fix O(n^2)/N+1 before caching or parallelism.", ["perf-budget"]),
("STD_PERF", "Cache at right layer", "Idempotent GETs with TTL/versioned keys; measure hit rate; test invalidation.", ["cache-layer"]),
("STD_PERF", "Paginate and stream", "Never unbounded lists in memory; cursors, streaming, indexed LIMIT.", ["paginate-stream"]),
("STD_PERF", "Batch and parallelize I/O", "Batched writes, bounded parallel fetches, pooled connections.", ["batch-parallel"]),
("STD_PERF", "Lean payloads", "Binary/compressed payloads and field selection on hot paths.", ["payload-efficiency"]),
# ---- review socratic stems ----
("SOCRATIC", "Loop invariant probe", "What invariant does this loop maintain — established, preserved, used where?", ["loop-invariant"]),
("SOCRATIC", "Edge-case probe", "Empty, null, zero, one, max-size, unicode — what happens?", ["edge-cases"]),
("SOCRATIC", "Failure-mode probe", "How does this fail, what does the caller see, what gets logged?", ["failure-modes"]),
("SOCRATIC", "Concurrency probe", "What breaks under simultaneous threads, requests, or overlapping retries?", ["concurrency-question"]),
("SOCRATIC", "Complexity probe", "Big-O in input size; at what n does this bottleneck?", ["complexity-question"]),
("SOCRATIC", "Simplicity probe", "Simplest sufficient version? What forced this complexity?", ["simpler-alternative"]),
("SOCRATIC", "Evidence probe", "What test, log, or benchmark proves this — what would falsify it?", ["evidence-question"]),
("SOCRATIC", "Scope probe", "YAGNI generality or requirement? Cost of adding later?", ["yagni-question"]),
("SOCRATIC", "Naming probe", "Could a new reader guess behavior from the name alone?", ["naming-question"]),
("SOCRATIC", "Security probe", "Where does untrusted input enter; how validated, escaped, authorized?", ["security-question"]),
("SOCRATIC", "Rollback probe", "How do we roll back; what breaks mid-rollback?", ["rollback-question"]),
("SOCRATIC", "Ownership probe", "Who owns this state, who mutates it, how are leaks/double-free prevented?", ["ownership-question"]),
]

FAM2SHARD = {"GRAMMAR": "grammar", "UNIV": "grammar", "LOGIC": "logic",
             "RHET": "rhetoric", "STD": "standard", "SOCRATIC": "review"}

LINK_NOTES = [
 ("false-cause", "Debugging causality trap: 'last deploy caused the bug' without bisection."),
 ("hasty-generalization", "Works-on-my-machine: one passing test treated as correctness."),
 ("confirmation-bias", "Happy-path-only testing; failing edges never exercised. (alias row if present)"),
 ("sunk-cost", "Keeping a doomed rewrite/branch because months are invested."),
 ("nirvana", "Rejecting a working solution for an unavailable ideal architecture."),
 ("appeal-to-novelty", "Adopting a framework because new, not better."),
 ("appeal-to-tradition", "Keeping stored procedures because always done so."),
 ("appeal-to-authority", "Senior said so, without benchmark or spec."),
 ("survivorship", "Copying FAANG stacks without their scale or team."),
 ("texas-sharpshooter", "Performance claims from cherry-picked benchmarks."),
 ("slippery-slope", "Blocking pragmatic fixes with codebase-collapse prophecies."),
 ("strawman", "Restating an RFC design in weakest form during review."),
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

    import re as _re
    for tag, dn, desc, als in ROWS:
        fam = _re.split(r"_", tag, maxsplit=1)[0]
        area = FAM2SHARD.get(fam, "general")
        slug = _re.sub(r"-+", "-", "".join(
            c.lower() if c.isalnum() else "-" for c in dn)).strip("-")[:48]
        et = "socratic_stem" if tag == "SOCRATIC" else "prog_rule"
        add(f"trivium:prog:{slug}", et, f"{area}", dn, SHARD, desc, als)
    for slug, note in LINK_NOTES:
        add(f"trivium:prog:link:{slug}", "fallacy_link", "programmer-note",
            f"Programmer note: {slug}", "trivium:logic:fallacy-notes",
            f"{note} See fallacy catalog for the general form and detection tests.",
            [f"{slug} programmers", f"programmer {slug}"])
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
    print(f"populate_trivium4: {_e} entities, {_a} aliases")
    _conn.close()
