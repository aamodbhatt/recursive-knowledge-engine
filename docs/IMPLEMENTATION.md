# IMPLEMENTATION

## Recursive Loop Logic
Controller flow:
1. Receive user question
2. Retrieve top-k chunks via hybrid fusion (dense FAISS + lexical evidence)
3. Optionally apply RLM source bias based on prior feedback
4. Generate answer from retrieved context
5. Score groundedness and unsupported-claim ratio
6. Critique answer and propose refined query
7. Compute combined confidence from critique + retrieval + groundedness
8. Repeat up to `max_iterations` (2-3, default 2)
9. Return highest-confidence answer, loop details, and total runtime

Fast mode:
- Uses heuristic critique to avoid extra LLM call overhead
- Supports early stop when query stabilizes and confidence is high
- Returns per-loop `duration_ms` and overall `total_duration_ms`
- Applies groundedness penalties to reduce false-positive confidence

## How Critique Refines Queries
Two paths:
- Full mode: LLM critic returns JSON (`critique`, `refined_query`, `confidence`)
- Fast mode: heuristic keyword extraction from top retrieved chunks expands query for next pass

This improves retrieval targeting while allowing latency control.

## API Response Shape (Query)
`POST /api/query` returns:
- `response_id`
- `final_answer`
- `confidence`
- `answer_verbosity`
- `source_filters` (request): optional list of attached indexed doc IDs to scope retrieval
- `reliability_grade`
- `challenge_risk`
- `support_redundancy`
- `single_source_dependency`
- `loops[]` with per-iteration query, critique, refined query, retrieval confidence, and `duration_ms`
- `total_duration_ms`
- `stopped_early` and `stop_reason`
- `created_at`

Loop metrics now include:
- `groundedness`
- `unsupported_claim_ratio`
- `source_coverage`

## Robustness Audit (Novel Feature)
- Optional `challenge_mode` runs a challenge retrieval query aimed at contradictions/caveats.
- Computes:
  - `challenge_risk`: overlap-adjusted contradiction risk from challenge evidence.
  - `support_redundancy`: answer-token support from at least 2 distinct sources.
  - `single_source_dependency`: fraction of answer tokens supported by only one source.
  - `reliability_grade`: `high` / `medium` / `low` based on coverage, challenge risk, dependency, and fallback state.
- Returns challenge evidence snippets in `challenge_chunks` for auditability.

## Why FAISS
- High-performance local vector search
- No paid dependency or external DB required
- File-based persistence for fast restarts
- Ideal for free-tier deployments

## Why MiniLM
- Strong semantic retrieval for CPU-only environments
- Smaller footprint than larger embedding models
- Good latency-quality tradeoff for recursive loops

## Provider-Agnostic LLM Layer
- `Answerer`: OpenAI-compatible chat completion grounded in retrieved context
- `Critic`: structured JSON critique in full mode
- Supports provider auto-detection (`openrouter`, `openai`, `groq`, `custom`)
- Uses `LLM_*` env vars with legacy `OPENROUTER_*` backward compatibility
- Applies provider-specific default model/base URL when only an API key is supplied
- OpenRouter model discovery is enabled only when OpenRouter is the active provider
- Retry strategy handles transient 429/5xx responses and fallback paths preserve usability during outages

## RLM Feedback Memory
- Every answer is assigned a `response_id` and associated source set.
- User feedback (`-1`/`+1`) updates source weights.
- Future retrieval scores receive bounded source-level bias.
- Stats endpoint exposes tracked source quality trends.

## Batch Ingestion + Benchmarking
- `POST /api/upload/batch` supports multi-document ingestion for test suites.
- `scripts/benchmark_harness.py` runs repeatable quality/latency evaluations and exports JSON reports.

## One-Command Startup
- `./start.sh` bootstraps dependencies, builds frontend, and starts the app in one command.
- This is intended for GitHub users to run locally with their own provider key via `LLM_API_KEY`.

## Frontend UX Implementation
- Built with Vite + React for cloud-hosting readiness
- Tailwind CSS stack enabled (tailwind + postcss + autoprefixer)
- Styles organized using `@tailwind` base/components/utilities and class-level `@apply` mappings
- Dark-mode default theme with readable high-contrast cards
- Chat-centric layout with central pane focus
- Split workspace layout: primary chat surface + sticky insights rail
- Tabbed insights rail:
  - `Ask + Tune`: controls glossary and iteration timeline
  - `Research Stats`: sources, query evolution, robustness, workspace metrics
- Attachment chips in composer and chat bubbles with remove-before-send support
- Multi-file attach from paperclip with document-type badges
- Attached-file scoping: each sent query can be restricted to only attached indexed docs
- Enter-to-send (`Shift+Enter` newline)
- Markdown rendering for answer/critique text formatting
- Per-answer feedback controls (`Helpful` / `Needs work`) wired to RLM
- Message utility actions (copy/reuse), quick prompt chips, and auto-scroll thread behavior
- Focus mode toggle hides non-chat panels for distraction-free querying
- Theme switcher toggles dark mode and beige light mode using `data-theme` styling
- Composer telemetry (word count + token estimate + scoped doc count)
- Metrics cards for model, index size, document count, embedder status, and cache size
- Loop timeline (`details` blocks) for transparent iteration-by-iteration inspection

## Source Alias Handling
- Deduplicated chunks are reused at vector level but now retain a multi-source alias list.
- This keeps attachment-scoped retrieval accurate even when the same content is uploaded again under a new file ID.

## System Flow Diagram
```text
Vite React UI
    |
    v
FastAPI /api
    |
    v
Controller (N loops)
    |
    +--> Retriever
    |     - lazy MiniLM load
    |     - query embedding cache
    |     - FAISS search
    |
    +--> Answerer (OpenAI-compatible provider)
    |
    +--> Critic
          - LLM JSON critique (full)
          - heuristic critique (fast mode)

Controller -> confidence scorer + early stop -> final answer + timing metadata
```
