# Recursive Knowledge Engine

Research-grade recursive RAG with a polished chat interface.

This project improves document QA quality by running a self-refining loop instead of a single retrieval pass:

`retrieve -> answer -> critique -> refine query -> retrieve`

It is built to be practical for local use, reproducible for experiments, and deployable on free tiers.

## Why This Is Different

Most RAG apps stop after one retrieval + one answer. This engine adds deeper evaluation and control:

- Recursive multi-pass retrieval (`1-3` passes)
- Ablation mode (`1 vs 2 vs 3` passes side-by-side)
- Confidence trajectory and calibration signals
- Query evolution trace (token-level diffs per iteration)
- Source coverage + groundedness checks
- Robustness audit (challenge retrieval, contradiction-risk signals)
- Failure case logging + JSON export for analysis
- RLM feedback memory that reweights source quality over time

## Feature Summary

### Retrieval + Answering
- FAISS local vector index (disk persistence)
- Sentence-transformers MiniLM embeddings
- Hybrid retrieval fusion (dense + lexical)
- Attachment-scoped retrieval (query only selected docs)
- Broad text document support (`pdf`, `docx`, `txt`, `md`, `csv`, `json`, code/text formats)

### Research Controls
- `fast_mode`: lower latency heuristic critique path
- `deterministic_mode`: reproducible generation path
- `answer_verbosity`: `short | normal | long`
- `ablation_mode`: compare answer quality/latency across recursion depths
- `challenge_mode`: reliability stress-check using challenge retrieval

### UX
- Chat-style UI with inline document chips
- Multi-file attach + remove-before-send
- Enter to send (`Shift+Enter` newline)
- Visible tuning toggle in composer
- Workspace tabs (`Ask + Tune`, `Research Stats`)

## Architecture

```text
React + Vite Frontend
  -> FastAPI Backend (/api)
      -> Recursive Controller
         -> Retriever (MiniLM + FAISS + cache)
         -> Answerer (OpenAI-compatible provider)
         -> Critic (LLM critic or fast heuristic)
      -> Session history + failure store + RLM memory
```

## Project Structure

```text
recursive-knowledge-engine/
├── app/                    # FastAPI backend + loop/controller/retrieval logic
├── frontend/               # React + Tailwind UI
├── scripts/                # Benchmark harness and utility scripts
├── docs/                   # IMPLEMENTATION.md + STATUS.md
├── data/uploads/           # Uploaded docs (ignored in git)
├── vectorstore/            # FAISS index metadata (ignored in git)
├── start.sh                # One-command local startup
├── requirements.txt
├── .env.example
└── .gitignore
```

## Prerequisites

- Python `3.10+`
- Node.js `18+` and npm
- macOS/Linux shell (or equivalent on Windows with minor command adaptation)

## Quickstart (Local, One Command)

1. Clone repo
2. Create env file (or let script create it):

```bash
cp .env.example .env
```

3. Set your key in `.env` (minimum required):

```env
LLM_API_KEY=your_key_here
```

4. Run:

```bash
./start.sh
```

5. Open:
- App: `http://127.0.0.1:8000`
- API docs: `http://127.0.0.1:8000/docs`

`start.sh` will:
- bootstrap `.venv` if missing
- install backend/frontend dependencies
- build frontend
- start FastAPI that serves the built UI

## LLM Provider Configuration (Any OpenAI-Compatible API Key)

The app supports provider-agnostic configuration through `LLM_*` env vars.

### Core env vars
- `LLM_PROVIDER=auto|openrouter|openai|groq|custom`
- `LLM_API_KEY=<your key>`
- `LLM_MODEL=<optional model id>`
- `LLM_BASE_URL=<optional full chat completions URL>`
- `LLM_FALLBACK_MODELS=<optional comma-separated models>`

Legacy `OPENROUTER_*` vars are still supported for backward compatibility.

### Auto defaults (if `LLM_MODEL` is blank)
- `openrouter` -> `openrouter/free`
- `openai` -> `gpt-4o-mini`
- `groq` -> `llama-3.1-8b-instant`

### Example configs

OpenRouter:
```env
LLM_PROVIDER=openrouter
LLM_API_KEY=sk-or-...
LLM_MODEL=openrouter/free
```

OpenAI:
```env
LLM_PROVIDER=openai
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
LLM_BASE_URL=https://api.openai.com/v1/chat/completions
```

Groq:
```env
LLM_PROVIDER=groq
LLM_API_KEY=gsk_...
LLM_MODEL=llama-3.3-70b-versatile
LLM_BASE_URL=https://api.groq.com/openai/v1/chat/completions
```

Custom OpenAI-compatible endpoint:
```env
LLM_PROVIDER=custom
LLM_API_KEY=<provider_key>
LLM_MODEL=<model_name>
LLM_BASE_URL=https://<host>/v1/chat/completions
```

## How To Use The App

1. Attach one or more docs from the paperclip in the composer.
2. Wait for each attachment to show indexed/ready.
3. Ask your question and send.
4. Use `Tune` button (next to Send) to open advanced controls.
5. Inspect:
- Confidence timeline
- Query evolution
- Retrieved sources
- Robustness audit
- Failure logs and export

## API Endpoints

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

### Sample query request

```bash
curl -X POST "http://127.0.0.1:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
        "question": "Summarize key findings with caveats.",
        "session_id": "demo-session",
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

## Benchmark / Hard Testing

Use the harness to evaluate behavior over larger sets:

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

`eval_queries.jsonl` example:
```json
{"question":"What is the main claim?","expected_keywords":["claim","result"]}
{"question":"List key limitations.","expected_keywords":["limitation","risk"]}
```

## Deployment (Free-Tier Friendly)

### Backend on Render
1. Create Render Web Service from repo
2. Build command: `pip install -r requirements.txt`
3. Start command: `uvicorn app.main:app --host 0.0.0.0 --port 10000`
4. Add env vars from `.env.example`

### Frontend on Cloudflare Pages
1. Set `VITE_API_BASE_URL=https://<your-render-backend>/api`
2. Build command: `npm run build`
3. Output directory: `frontend/dist`

## GitHub Readiness

This repo is configured to avoid committing sensitive/local runtime data:

- `.env` and secret env files are ignored
- Uploaded docs in `data/uploads/` are ignored
- Generated FAISS index in `vectorstore/` is ignored
- `node_modules`, build output, caches, logs are ignored

Tracked placeholders are kept via:
- `data/uploads/.gitkeep`
- `vectorstore/.gitkeep`

## Troubleshooting

### "Using a local fallback summary because the LLM provider is unavailable"
- Check `LLM_API_KEY` in `.env`
- If using non-default provider, set `LLM_PROVIDER`, `LLM_MODEL`, and `LLM_BASE_URL`
- Restart backend after env changes

### Empty or weak answers
- Increase `top_k`
- Disable `fast_mode` for full critic path
- Use `2` or `3` passes
- Ensure relevant documents are attached/in scope

### First query is slower
- Expected on cold start while embedding model and provider path warm up

## Limitations

- Session history/failure memory are in-memory (reset on restart)
- PDF extraction quality depends on the source text layer
- Free-tier providers can rate limit under burst traffic

## License

MIT (recommended)
