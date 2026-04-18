---
title: Autonomous Financial Analyst
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: LangGraph agent with Claude + RAG — AI investment research.
---

# Autonomous Financial Research Analyst

An institutional-grade investment research assistant. A LangGraph agent with
Claude Sonnet 4.6 calls eight tools (stock fundamentals, technicals, analyst
consensus, news, sentiment, and RAG over private analyst reports) to produce
comprehensive briefings on AI-sector companies.

Designed for **public Hugging Face Space** deployment with strict per-session
isolation — every visitor brings their own Anthropic + Tavily keys, uploads
their own PDFs, and is walled off from everyone else.

---

## Quick start (local, Windows)

```bash
# 1. Clone
git clone https://github.com/alanvaa06/Autonomous_Financial_Analyst.git
cd Autonomous_Financial_Analyst

# 2. Create a Python 3.12 virtual environment (required — see "Known pins")
py -3.12 -m venv .venv

# 3. Install dependencies (~1-2 GB, takes a few minutes)
.venv/Scripts/python.exe -m pip install -r requirements.txt

# 4. Run
.venv/Scripts/python.exe app.py
```

Then open **http://127.0.0.1:7860** and paste your API keys in the **API Keys** tab.

### Quick start (macOS / Linux)

```bash
git clone https://github.com/alanvaa06/Autonomous_Financial_Analyst.git
cd Autonomous_Financial_Analyst
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

### Getting API keys

1. **Anthropic** — https://console.anthropic.com/settings/keys
   You pay Anthropic directly for Claude Sonnet 4.6 and Haiku 4.5 usage.
2. **Tavily** — https://tavily.com
   Free tier covers casual use.

---

## What it does

Given a ticker (or a list of tickers), the agent proactively gathers:

- **Financial health** — live price + 3-year performance + full fundamentals
  (revenue, margins, FCF, P/E, EPS, growth) via yfinance
- **Technical picture** — RSI(14), MA50, MA200, 52-week high/low, golden/death
  cross signals
- **Analyst consensus** — Wall Street Buy/Hold/Sell counts + price targets
- **Market sentiment** — real-time news (Tavily) scored by Claude Haiku 4.5
- **AI research activity** — retrieved from a **hybrid RAG corpus**:
  - *Your* uploaded PDFs (per-session, in-memory, isolated)
  - *Bundled* base corpus of AI-initiative analyst reports
- **Risks** — 2-3 proactively identified
- **Recommendation** — Buy / Hold / Sell with confidence + full source citations

---

## Architecture

Three views of the system, short form:

**1. Agent loop**

```
HumanMessage ──► [agent_node: Claude Sonnet 4.6 + 8 tools + cached charter]
                       │
                should_continue
              ┌────────┴────────┐
              ▼                 ▼
        tool_calls?          no_calls
              │                 │
              ▼                 ▼
        [tool_node]           END
              │                 │
              └──► loop back ◄──┘
```

**2. Tool catalog**

| Family | Tool | Source | Key needed |
|---|---|---|---|
| Static (shared) | `get_stock_price` | yfinance | no |
| | `get_stock_history` | yfinance | no |
| | `get_analyst_ratings` | yfinance | no |
| | `get_financials` | yfinance | no |
| | `calculate_technical_indicators` | yfinance + pandas | no |
| Session (closure-scoped) | `search_financial_news` | Tavily | Tavily |
| | `analyze_sentiment` | Claude Haiku 4.5 | Anthropic |
| | `query_private_database` | dual Chroma + Claude Sonnet 4.6 | Anthropic |

**3. RAG dual-retriever merge**

```
    query ─► session_retriever (user PDFs, in-memory Chroma)  ──┐
          └► base_retriever     (bundled PDFs, persisted)      ──┤
                                                                 ▼
                                                 dedupe by content hash
                                                                 ▼
                                                 top 12 chunks → Claude
                                                                 ▼
                                                  grounded answer + cites
```

---

## Repository layout

```
.
├── app.py             # Gradio Blocks UI — gr.State for per-session isolation
├── agent.py           # build_agent_for_session(keys, retriever) factory
├── tools.py           # 5 static tools + build_session_tools factory
├── rag.py             # base (persisted) + session (in-memory) retrievers
├── requirements.txt   # pinned versions (see "Known pins" below)
├── README.md          # this file (also the HF Space card)
├── .env.example       # local-dev env var template
├── .gitignore
├── .claude/
│   └── launch.json    # dev-server config for Claude Code users
└── data/
    ├── README.md
    └── Companies-AI-Initiatives.zip   # base corpus (auto-extracted on first run)
```

At first run, `rag.py` extracts the zip into `data/Companies-AI-Initiatives/`
and builds a persisted Chroma index at `data/chroma_index/`. Both folders are
git-ignored — they regenerate from the committed zip.

---

## Public-Space design — BYO-key + per-session isolation

Running this as a public HF Space means many users share one Python process.
We do a few things to keep them isolated:

| Concern | How we address it |
|---|---|
| API-key leakage across users | Keys live ONLY in that session's tool closures. **Never** `os.environ`. |
| PDF leakage across users | Uploads go into a fresh in-memory Chroma collection with a UUID name. No `persist_directory`. Auto-GC'd when the Gradio session ends. |
| Owner's wallet exposure | Env-var keys are **ignored** unless `ALLOW_ENV_KEYS=1`. Each visitor pastes their own. |
| Abuse via huge uploads | ≤ 25 MB per PDF, ≤ 10 PDFs per session, encrypted/unreadable PDFs skipped. |
| Cross-session cost | The HuggingFace embedding model is shared (local, free). The base corpus is embedded once at boot. Sessions only pay for their own uploads + LLM calls. |

---

## Deploying to a public Hugging Face Space

1. Create a new Gradio Space on HF.
2. Push **this repo's contents** to the Space's git remote:
   ```bash
   git remote add hf https://huggingface.co/spaces/<you>/<space-name>
   git push hf main
   ```
3. **Do NOT** set `ANTHROPIC_API_KEY` or `TAVILY_API_KEY` as Space Secrets —
   every visitor will bring their own via the UI.
4. **Do NOT** set `ALLOW_ENV_KEYS=1`.

## Deploying to a private/team Space

If only trusted users access the Space, you can pre-fill keys:

1. Set `ANTHROPIC_API_KEY`, `TAVILY_API_KEY` as Space Secrets.
2. Set `ALLOW_ENV_KEYS=1` as a Space Variable.

The app auto-builds the agent on session start; the form still works as an override.

---

## Known pins (don't change without testing)

This stack has a few tight version constraints driven by ecosystem churn:

| Package | Pin | Why |
|---|---|---|
| Python | **3.12** | pydantic v1 shims used by LangChain 0.3 don't work on 3.14+ |
| `gradio` | `==4.44.1` | locks to `gradio-client 1.3.0` |
| `fastapi` | `>=0.110,<0.113` | starlette 0.38+ changed `TemplateResponse` signature (breaks gradio 4.44) |
| `starlette` | `>=0.37,<0.38` | same |
| `huggingface_hub` | `<1.0` | 1.x removed `HfFolder` (still imported by gradio 4.44) |
| `transformers` | `<5.0` | 5.x requires hf_hub≥1.5, conflicts with gradio pin |
| `yfinance` | `>=0.2.40,<0.2.50` | ≥0.2.50 requires `websockets≥13`, conflicts with gradio-client 1.3 |

Also: `app.py` top-of-file monkey-patches `gradio_client.utils._json_schema_to_python_type` to handle `bool` schemas — works around a bug when `gr.File` / `gr.State` introspection encounters `additionalProperties: True`.

---

## Environment variables

See `.env.example`. All are optional:

- `ANTHROPIC_API_KEY` — only used if `ALLOW_ENV_KEYS=1` (local dev)
- `TAVILY_API_KEY` — same
- `ALLOW_ENV_KEYS` — `1` to enable env-var bootstrap; unset on public Spaces
- `PDF_DIR` — override the base-corpus PDF folder (default: `data/Companies-AI-Initiatives/`)

---

## License

MIT.
