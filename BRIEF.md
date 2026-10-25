# Public-Sector Policy Assistant — Small RAG Demo

## Instructions for Claude Code / Codex

**Build this project now, end to end. Do not stop at a design or implementation plan.** Make reasonable decisions without asking me questions. Prioritize a small, working demonstration I can understand and describe in a Google Public Sector Forward Deployed Engineer recruiter call.

**Time budget:** Aim for a working MVP in roughly 45–60 minutes of implementation effort. Do not add features outside this brief. If blocked, deliver the simplest working version and clearly state what is blocked.

## What to build

A **Python command-line assistant** that answers questions over a small set of municipal-service policy documents and **cites the document(s)** supporting its answers.

Example: `python ask.py "Can I apply for a residential parking permit online?"`

The entire workflow should be:

```text
Local policy documents -> split into chunks -> Gemini embeddings -> Chroma DB

User question -> Gemini query embedding -> Chroma top-3 search
              -> relevant excerpts -> Gemini answer + source citations
```

This is a **personal proof of concept**, not a production government system.

## Technology (keep it minimal)

- Python 3.11+
- Official `google-genai` Python SDK for Gemini generation **and** embeddings
- Local persistent Chroma (`chromadb`) for similarity search
- `python-dotenv` for `GEMINI_API_KEY`
- `pytest` for a few fast offline tests

Use the currently supported Gemini generation model available to the API key and `gemini-embedding-001` for embeddings, unless the API indicates a supported replacement is needed. Don't use deprecated Google AI SDKs. Pass precomputed vectors to Chroma for both indexing and querying; don't accidentally use a second, unrelated embedding model.

## Requirements — only these

1. Create **five or six short Markdown sample policies** covering realistic municipal topics (e.g., parking permits, recycling, building permits, public records requests, water bills). **Label them conspicuously as fictional demo policies**, not official government guidance. Give each a clear title and distinct facts so answers are easy to verify.
2. `python ingest.py`: read `data/*.md`, split into small overlapping chunks, create Gemini embeddings, and upsert them to a **persistent** Chroma collection with deterministic IDs, filename/title metadata, and chunk text. Re-running ingestion should not duplicate chunks.
3. `python ask.py "QUESTION"`: embed the question; retrieve the top 3 most relevant chunks from Chroma; send only those chunks plus the question to Gemini; print a concise answer with **[document filename]** citations. Also show the source snippets used and total latency for transparency.
4. **Grounding:** Tell Gemini to use only the retrieved excerpts. If the answer is not supported, say **"I couldn't find that in the provided policies."** Never fabricate policies or citations. Every cited filename must correspond to a retrieved document.
5. Handle common errors clearly: missing API key, ingestion not yet run, empty collection, invalid model, and API failure. Never commit credentials.
6. Add at least **three answerable demo questions** and **one deliberately unanswerable question**. Put them in this README with expected answer facts, not fabricated model outputs.
7. Include a few simple **offline pytest tests** covering chunking, deterministic IDs, and the grounded-answer prompt or citation logic. Mock API calls where needed. Tests must not require a paid API request.
8. Actually run available tests and, **if I provide `GEMINI_API_KEY`**, run ingestion and at least two end-to-end questions. Fix errors discovered. If no key is set, do not pretend the external API demo succeeded; give exact commands for me to run.

## Expected file layout

```text
README.md
requirements.txt
.env.example
.gitignore
data/
  parking_permits.md
  recycling.md
  ...3–4 more short policies...
rag.py          # small shared helpers (chunking, Gemini, Chroma)
ingest.py       # indexing command
ask.py          # interactive question command
tests/
  test_rag.py
```

You may simplify this layout if it makes the code easier to follow. Write readable, straightforward Python; avoid excessive abstraction.

## Setup and demo experience

Make setup no harder than:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Put GEMINI_API_KEY in .env
python ingest.py
python ask.py "Can I get a residential parking permit?"
pytest -q
```

Document the equivalent Windows activation command only if convenient. `.env` and Chroma database files must be in `.gitignore`.

## Explicitly out of scope

**Do not build:** a web UI, multi-agent system, LangGraph, CrewAI, Google ADK, LangChain, FastAPI, Docker, Kubernetes, cloud deployment, auth/permissions, PDF scraping, scheduled ingestion, CI/CD, model training, or complex evaluation tooling. No architectural overengineering.

## What I need to understand for the recruiter call

After implementation, add a short **"How It Works"** section to this README explaining, in plain English:

- What embeddings are and why the vector database exists
- What RAG is and how it reduces unsupported answers
- How source citations are carried through the pipeline
- Two things needed for production government deployments: **document-level access control** and **evaluation/monitoring**

Add a **30-second truthful talking point** along these lines, updated to match what you actually completed:

> "I've been building a small personal RAG prototype in Python using Gemini and a vector database. It indexes sample municipal policy documents, retrieves relevant passages, and generates cited answers. My background is stronger in distributed systems and AI infrastructure, so this was a useful way to get hands-on with the application layer. I'd be excited to build on that in real customer deployments."

**Do not claim the project is production-grade, uses official government policies, or fulfills years of enterprise AI experience.**

## Definition of done

- I can run two commands—`python ingest.py` and `python ask.py "..."`—and see a grounded answer with named sources.
- An out-of-scope question produces a clear "not found" answer.
- Re-running ingestion doesn't create duplicate documents.
- Tests pass, or genuine blockers are documented.
- The whole implementation is small enough for me to read and explain in **10 minutes**.

At the end, tell me: **what you built, how to run it, test results, any actual blockers, and exactly three talking points for Alan Ward.**
