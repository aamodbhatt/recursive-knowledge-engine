# Recursive Knowledge Engine

A research-grade, installable retrieval-augmented generation (RAG) system that answers questions over your documents using a recursive self-refinement loop instead of one-shot retrieval.

Core loop:

`retrieve -> answer -> critique -> refine query -> retrieve`

Built for two goals at once:
- Practical daily use (clean chat UX, one-command startup, local indexing)
- Research-style evaluation (ablation mode, calibration tracking, failure logging, robustness audit)

---

## Quick Local Setup (One Command)

### Step A: Get an API key (OpenRouter free path recommended)

1. Create/sign in to an OpenRouter account: [https://openrouter.ai/](https://openrouter.ai/)
2. Generate an API key from: [https://openrouter.ai/keys](https://openrouter.ai/keys)
3. Use a free model route (for example `openrouter/free`) for local testing

Note:
- OpenRouter has free routes but they can rate-limit during peak usage.
- If free routes are busy, keep the same key and switch to another available model.

### Step B: Configure environment

```bash
cp .env.example .env
```

Edit `.env` with at least:

```env
LLM_PROVIDER=openrouter
LLM_API_KEY=sk-or-...your_key...
LLM_MODEL=openrouter/free
```

### Step C: Start the app

```bash
./start.sh
```

Open:
- App: `http://127.0.0.1:8000`
- OpenAPI docs: `http://127.0.0.1:8000/docs`

`start.sh` handles:
- venv creation
- backend deps install
- frontend deps install
- frontend production build
- FastAPI startup

### Alternative key providers

You can use any OpenAI-compatible API key/provider:

- OpenAI keys: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- Groq keys: [https://console.groq.com/keys](https://console.groq.com/keys)

Then set `LLM_PROVIDER`, `LLM_API_KEY`, `LLM_MODEL`, and optionally `LLM_BASE_URL` in `.env`.

---

## 1) What This Project Is

Recursive Knowledge Engine is a full-stack RAG application with:

- A FastAPI backend for document ingestion, vector retrieval, recursive reasoning control, and evaluation metrics
- A React + Tailwind frontend designed as a chat-first productivity workspace
- Local FAISS indexing and persistence for free-tier friendliness and easy local usage
- Provider-agnostic LLM integration via OpenAI-compatible chat completions

It is not an "AI demo page". It is a controlled retrieval/reasoning system where you can inspect how answers were formed, how confidence changed per iteration, and when recursion helped or hurt.

---

## 2) What It Does

### Document ingestion and indexing
- Upload one or many files
- Supports broad text-like formats (`pdf`, `docx`, `txt`, `md`, `csv`, `json`, `xml`, `html`, code files, logs, configs, etc.)
- Chunks content and embeds with MiniLM
- Stores vectors in FAISS (`vectorstore/`) and tracks chunk metadata

### Query and recursive reasoning
- Retrieves top-k chunks
- Generates grounded answer
- Critiques answer and refines retrieval query
- Repeats for configurable depth (`1-3` passes)
- Returns best answer + full iteration timeline

### Transparency and quality controls
- Confidence per iteration
- Groundedness and unsupported-claim signals
- Source coverage metric (answer-token overlap with retrieved evidence)
- Early-stop behavior in fast mode
- Robustness audit mode with challenge retrieval

### Research tooling
- Ablation mode: compare 1/2/3 pass outputs side-by-side
- Query evolution visualization (token-level changes)
- Failure case detection and export as JSON
- RLM feedback memory to learn source reliability over time

---

## 3) Why It Is Different

Most RAG projects stop after one retrieval and one answer. This project emphasizes *process observability* and *controlled recursion*.

### Differentiators
- Recursive retrieval with inspectable per-step outputs
- Ablation built into the product UX (not external scripts only)
- Calibration signals beyond a single "confidence" number
- Failure logger for degradation analysis
- Robustness audit (challenge evidence + source dependency)
- Feedback-driven source bias memory (RLM)

### In short
Instead of asking only "What is the answer?", this system asks:
- "How did the answer evolve?"
- "Did extra passes actually help?"
- "How grounded is this answer in retrieved evidence?"
- "Can this result survive challenge retrieval?"

---

## 4) Key Features

### Retrieval + indexing
- FAISS local vector index with disk persistence
- Sentence-transformers MiniLM embeddings
- Hybrid fusion retrieval (dense + lexical evidence)
- Query embedding cache for repeated prompt speedups
- Source alias handling for duplicate-content uploads

### Recursive controller
- Configurable `max_iterations` (1-3)
- Fast mode (`fast_mode`) for latency-sensitive use
- Deterministic mode (`deterministic_mode`) for reproducibility
- Early-stop policy under high confidence stability

### Research-grade evaluation
- Ablation outputs (`ablation_mode`, `ablation_depths`)
- Confidence trajectory + drop-point detection
- Source coverage and low-coverage warnings
- Groundedness and unsupported-claim ratio
- Failure-case logging + export endpoint

### Robustness audit
- Challenge retrieval query generation
- Challenge risk score
- Support redundancy metric
- Single-source dependency metric
- Reliability grade (`high`, `medium`, `low`)

### Feedback loop (RLM)
- Per-response thumbs up/down feedback
- Source-level bias updates from user feedback
- Stats for tracked source quality trends

### Frontend UX
- Chat-style centered workflow
- Multi-doc attachments with per-file chips/icons
- Remove file before send
- Enter-to-send (`Shift+Enter` newline)
- Bright `Tune` toggle near Send for discoverable controls
- Tabs: `Ask + Tune` and `Research Stats`

---

## 5) Pros and Cons

### Pros
- Better retrieval quality than one-pass baselines on hard queries
- Strong transparency: users can inspect every iteration
- Research-friendly metrics and failure collection
- Easy local install, free-tier deployability
- Provider-agnostic LLM config (not locked to a single vendor)

### Cons / Tradeoffs
- More passes increase latency and token usage
- Session/failure memory is in-memory by default (resets on restart)
- PDF extraction quality depends on text layer quality
- Free-tier models/providers may rate-limit under burst load
- Deterministic mode can reduce exploratory richness

---

## 6) System Architecture

```text
React + Vite + Tailwind Frontend
  -> FastAPI (/api)
      -> RecursiveController
         -> Retriever (MiniLM embeddings + FAISS + cache)
         -> Answerer (LLM, OpenAI-compatible)
         -> Critic (LLM critic or heuristic fast path)
      -> SessionHistoryStore
      -> FailureCaseStore
      -> RewardLearningMemory (RLM)
```

---

## 7) Repository Layout

```text
recursive-knowledge-engine/
├── app/
│   ├── main.py
│   ├── api.py
│   ├── controller.py
│   ├── retriever.py
│   ├── answerer.py
│   ├── critic.py
│   ├── rlm.py
│   ├── schemas.py
│   ├── config.py
│   └── llm_provider.py
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── styles.css
│   ├── package.json
│   └── vite.config.js
├── scripts/
│   └── benchmark_harness.py
├── docs/
│   ├── IMPLEMENTATION.md
│   └── STATUS.md
├── data/uploads/          # git-ignored runtime files
├── vectorstore/           # git-ignored runtime index
├── start.sh
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## 8) Prerequisites

- Python `3.10+`
- Node.js `18+`
- npm

---

## 9) LLM Configuration (Any OpenAI-Compatible API Key)

The backend uses provider-agnostic env vars:

- `LLM_PROVIDER=auto|openrouter|openai|groq|custom`
- `LLM_API_KEY=...`
- `LLM_MODEL=...` (optional)
- `LLM_BASE_URL=...` (optional, required for many custom providers)
- `LLM_FALLBACK_MODELS=...` (optional comma-separated)
- `LLM_TIMEOUT_SECONDS=45`
- `LLM_MAX_RETRIES=2`

Legacy compatibility remains:
- `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, etc. still work

### Auto defaults (if `LLM_MODEL` is empty)
- `openrouter`: `openrouter/free`
- `openai`: `gpt-4o-mini`
- `groq`: `llama-3.1-8b-instant`

### Example: OpenRouter
```env
LLM_PROVIDER=openrouter
LLM_API_KEY=sk-or-...
LLM_MODEL=openrouter/free
```

### Example: OpenAI
```env
LLM_PROVIDER=openai
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
LLM_BASE_URL=https://api.openai.com/v1/chat/completions
```

### Example: Groq
```env
LLM_PROVIDER=groq
LLM_API_KEY=gsk_...
LLM_MODEL=llama-3.3-70b-versatile
LLM_BASE_URL=https://api.groq.com/openai/v1/chat/completions
```

### Example: Custom OpenAI-compatible endpoint
```env
LLM_PROVIDER=custom
LLM_API_KEY=<your_key>
LLM_MODEL=<model_name>
LLM_BASE_URL=https://<provider-host>/v1/chat/completions
```

---

## 10) How To Use

1. Attach one or more documents using the paperclip
2. Ensure chips show indexed/ready status
3. Ask a question and press Enter
4. Use `Tune` button next to Send for advanced controls
5. Inspect `Research Stats` for coverage, evolution, failures, robustness

---

## 11) API Endpoints

- `GET /api/health`
- `GET /api/stats`
- `POST /api/upload`
- `POST /api/upload/batch`
- `POST /api/query`
- `POST /api/feedback`
- `GET /api/rlm/stats`
- `GET /api/history/{session_id}`
- `GET /api/failures`
- `GET /api/failures/export`
- `POST /api/index/clear`

### Query example

```bash
curl -X POST "http://127.0.0.1:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
        "question": "Summarize key findings and caveats.",
        "session_id": "demo",
        "top_k": 6,
        "max_iterations": 3,
        "fast_mode": false,
        "deterministic_mode": true,
        "ablation_mode": true,
        "ablation_depths": [1,2,3],
        "answer_verbosity": "long",
        "challenge_mode": true
      }'
```

---

## 12) Benchmarking / Hard Testing

Run the benchmark harness on many docs/queries:

```bash
python scripts/benchmark_harness.py \
  --api-base http://127.0.0.1:8000/api \
  --queries-file ./eval_queries.jsonl \
  --docs-dir ./eval_docs \
  --output ./benchmark_report.json \
  --max-iterations 3 \
  --answer-verbosity long \
  --deterministic-mode
```

`eval_queries.jsonl` sample:

```json
{"question":"What is the main claim?","expected_keywords":["claim","result"]}
{"question":"List major limitations.","expected_keywords":["limitation","risk"]}
```

---

## 13) Deployment

### Backend: Render (free tier)
1. Connect repo to Render
2. Build command: `pip install -r requirements.txt`
3. Start command: `uvicorn app.main:app --host 0.0.0.0 --port 10000`
4. Add env vars from `.env.example`

### Frontend: Cloudflare Pages (free tier)
1. Set `VITE_API_BASE_URL=https://<render-backend>/api`
2. Build command: `npm run build`
3. Output directory: `frontend/dist`

---

## 14) Security and Privacy Notes

- `.env` and secret env files are git-ignored
- Uploaded user docs are git-ignored (`data/uploads/`)
- Generated vector index is git-ignored (`vectorstore/`)
- Local runtime data is not automatically encrypted at rest
- For production, add auth + access control before multi-user exposure

---

## 15) Current Limitations

- In-memory session/failure stores reset on backend restart
- No built-in auth layer yet
- No distributed job queue for large-scale ingestion
- No persistent DB for long-term analytics by default

---

## 16) Roadmap Ideas

- Redis/Postgres-backed persistent session/failure stores
- Streaming token responses
- Citation-level inline grounding links
- Async ingestion pipeline for very large corpora
- CI benchmark regression gating

---

## 17) Troubleshooting

### "Using a local fallback summary because the LLM provider is unavailable"
- Verify `LLM_API_KEY`
- Verify provider-specific `LLM_MODEL` / `LLM_BASE_URL`
- Restart server after changing `.env`

### Weak answers
- Increase `top_k`
- Use `2-3` passes
- Disable `fast_mode` for full critic path
- Ensure relevant docs are attached/scoped

### Slow first request
- Expected on cold start (embedding + provider warmup)

---

## 18) GitHub-Ready Notes

This repo is prepared for open-source publishing:

- Runtime and sensitive files are ignored in `.gitignore`
- Runtime folders are preserved via `.gitkeep` placeholders
- README includes setup, architecture, usage, deployment, and limitations

---

## 19) License

MIT (recommended)
