# STATUS

## Current Features
- Vite + React frontend with dark-mode default UI
- Chat-centric central pane with composer-first workflow
- Complete UI/UX refresh with split workspace layout, sticky insights rail, improved spacing/typography, and subtle fade-in motion
- Tailwind CSS integrated (`@tailwind` layers + component `@apply` architecture) for scalable theming and faster visual iteration
- Markdown rendering for model outputs (bold, lists, inline code)
- Two-tab insights workflow (`Ask + Tune` and `Research Stats`) for low-clutter navigation
- Light/Dark theme switcher (light mode uses beige palette)
- Attachment chips in composer + chat, remove-before-send, and multi-file attach
- Enter-to-send (`Shift+Enter` for newline)
- Query-time attachment scoping so retrieval can target only the files attached to that prompt
- Source-alias aware dedupe so scoped retrieval still works when uploads are content-duplicates
- Per-answer feedback buttons wired to RLM (`Helpful` / `Needs work`)
- New productivity UX: focus mode, quick prompt chips, message copy/reuse actions, auto-scroll, and composer token estimate
- Session history panel with click-to-reuse previous questions
- FastAPI backend with recursive retrieval loop (`retrieve -> answer -> critique -> refine`)
- Configurable loop depth (1-3 iterations) with ablation comparison mode
- Broad document support (`pdf`, `docx`, markdown/text/data/code formats)
- Multi-file batch upload endpoint (`/api/upload/batch`)
- FAISS local persistence (`vectorstore/faiss.index` + `vectorstore/chunks.json`)
- MiniLM embeddings with lazy loading at first use
- Query embedding LRU cache for repeated query speedup
- Hybrid retrieval fusion (dense + lexical ranking)
- Provider-agnostic OpenAI-compatible LLM integration (`LLM_*`), with OpenRouter legacy compatibility
- Provider auto-defaults for model/base URL (OpenRouter, OpenAI, Groq) so key-only setup works faster
- `fast_mode` for lower latency (heuristic critique + early-stop behavior)
- Deterministic mode for reproducible runs
- Confidence + groundedness + source-coverage tracking per iteration
- Robustness audit metrics: challenge risk, support redundancy, single-source dependency, reliability grade
- Failure case logging and JSON export
- RLM feedback memory with source re-weighting (`/api/feedback`, `/api/rlm/stats`)
- Answer verbosity control (`short`/`normal`/`long`) for long-form responses
- Health and stats endpoints (`/api/health`, `/api/stats`)
- One-command startup script: `./start.sh`
- GitHub packaging hardening: expanded `.gitignore` for secrets/runtime artifacts and tracked `.gitkeep` placeholders for required runtime folders
- README rewritten for public release with full project narrative: problem, differentiation, pros/cons, architecture, setup, provider config, API, benchmarking, deployment, troubleshooting, and roadmap
- README onboarding flow improved: local setup moved to the top immediately after intro for faster first-run success
- README setup expanded with explicit API key acquisition steps (including OpenRouter free path) and provider alternatives

## Known Bugs
- Free-tier providers can still hit rate limits under burst traffic
- PDF extraction quality depends on text layer availability
- Session history is in-memory and resets when backend restarts
- First request after cold start can be slower due to initial model/provider warmup

## TODO
- Add Redis-backed persistent query history
- Add chunk-level citations in final answer rendering
- Add async background ingestion queue for large PDFs
- Add CI job for benchmark harness quality/latency tracking
- Add user-facing model selector with automatic fallback policy

## Next Milestones
- Hybrid retrieval (BM25 + dense vector) improvements for larger corpora
- Streaming answer tokens to frontend
- User auth and workspace-scoped indices
- Docker + CI pipelines for reproducible deploys
- Background job support for large document ingestion on free-tier constraints

## Open Research Questions
- How much quality delta remains between full critique and fast heuristic mode?
- Best early-stop policy for accuracy vs latency?
- Should loop depth be adaptive by retrieval confidence?
- Which chunking strategy is most robust across mixed PDF/TXT corpora?
- What confidence calibration best predicts factual correctness across domains?
