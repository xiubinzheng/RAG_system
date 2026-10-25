# Public-Sector Policy Assistant — Small RAG Demo

A small Python command-line assistant. It answers questions about a handful of **fictional** municipal policy documents and cites the document(s) behind each answer.

> **Personal proof of concept.** The policies in `data/` describe a made-up "City of Maplewood". They are not official guidance from any real government, and this is not a production system. The original project brief is kept in `BRIEF.md`.

```text
Local policy documents -> split into chunks -> Gemini embeddings -> Chroma DB

User question -> Gemini query embedding -> Chroma top-3 search
              -> relevant excerpts -> Gemini answer + source citations
```

## Files

| File | What it does |
|---|---|
| `data/*.md` | Fourteen short fictional policies (parking, recycling, building permits, public records, water bills, noise, facility rentals, surplus property, property tax, snow removal, library, business licenses, street trees, pools) |
| `rag.py` | Shared helpers: chunking, deterministic IDs, Gemini calls, Chroma access, grounded prompt, citation checks |
| `ingest.py` | Chunks `data/*.md`, embeds the chunks with `gemini-embedding-001`, and upserts them into a persistent Chroma DB (`chroma_db/`) |
| `ask.py` | Embeds the question, retrieves the top 3 chunks, asks Gemini for a cited answer, and prints the answer, sources, snippets, and latency |
| `tests/test_rag.py` | Offline pytest tests. Gemini is mocked, so no API calls are made |

## Setup and run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Put GEMINI_API_KEY in .env  (https://aistudio.google.com/apikey)
python ingest.py
python ask.py "Can I apply for a residential parking permit online?"
pytest -q
```

The generation model defaults to `gemini-3.6-flash`. To use a different model, set `GEMINI_MODEL` in `.env`. Embeddings always use `gemini-embedding-001`.

You can re-run `python ingest.py` safely. Chunk IDs are deterministic (`filename::index::content-hash`), so a re-run upserts the same IDs instead of adding duplicates. It also deletes chunks from files that were edited or removed.

## Demo questions

Expected facts come from the documents themselves. These are not recorded model outputs.

| Question | Expected answer facts | Source |
|---|---|---|
| `python ask.py "Can I apply for a residential parking permit online, and how much does it cost?"` | Yes, through the Maplewood Resident Portal (or in person at 120 Elm Street). $45/year; a second household permit is $75; residents 65+ pay $20. | `parking_permits.md` |
| `python ask.py "Can I put glass bottles in my recycling cart?"` | No. Glass is not accepted curbside; take it to the Glass Drop-Off Center at 900 Quarry Road. | `recycling.md` |
| `python ask.py "How long does the city have to respond to a public records request?"` | Acknowledgement within 5 business days; complex requests can be extended by up to 10 more business days with written notice. | `public_records.md` |
| `python ask.py "When is my water bill due and what happens if I pay late?"` | Due 21 days after the bill date; $10 late fee; disconnection possible after 60+ days past due, following a 10-day notice; $50 reconnection. | `water_bills.md` |
| `python ask.py "How much does it cost to rent a park pavilion?"` | $60 per 4-hour block for residents, $90 for non-residents; book via the Parks & Facilities Reservation Portal at least 10 days ahead. | `city_facility_rentals.md` |
| `python ask.py "Can city employees buy surplus city vehicles at auction?"` | No. City employees and their immediate family may not bid; auctions are online and quarterly. | `surplus_property.md` |
| **Unanswerable:** `python ask.py "How do I get a dog license?"` | `I couldn't find that in the provided policies.` | none |

## Error handling

`ask.py` and `ingest.py` print one clear line and exit non-zero for each of these cases:

- `GEMINI_API_KEY` missing or still the placeholder
- An invalid or rejected API key
- `ask.py` run before `ingest.py` (no DB, or the collection is missing)
- An empty collection
- An unknown model, which suggests setting `GEMINI_MODEL`
- Rate limits
- Other API or network failures

`.env` and `chroma_db/` are in `.gitignore`.

## How It Works

**Embeddings and the vector database.** An embedding is a list of numbers (here, 3,072 of them) that captures what a piece of text *means*. Texts about similar things get vectors that point in similar directions, even when they share no words. For example, "Can I park on my street?" lands near the parking-permit text. During ingestion, each policy is split into small overlapping chunks (about 80 words, with a 20-word overlap so facts aren't cut in half). Each chunk is embedded with Gemini, and the vectors are stored in Chroma. Chroma is a vector database. Its job is to quickly find the stored vectors closest to a new one. At question time, the question is embedded the same way, and Chroma returns the 3 nearest chunks. Both sides use the same Gemini embedding model. The code passes vectors to Chroma directly and never uses Chroma's built-in embedder.

**RAG (Retrieval-Augmented Generation).** Instead of asking the model what it "knows", we first *retrieve* the relevant excerpts. Then we ask the model to answer *only from those excerpts*. The prompt tells Gemini not to use outside knowledge, to cite a filename after each fact, and to reply with exactly "I couldn't find that in the provided policies." when the excerpts don't contain the answer. This keeps answers tied to real source text and gives the model an explicit way to say "I don't know", which reduces made-up answers. It reduces them but doesn't eliminate them.

**How citations are carried through.** Each chunk is stored in Chroma with metadata: source filename, document title, and chunk index. Retrieved chunks are placed in the prompt with labels like `[parking_permits.md]`, and the model cites using those labels. After generation, `ask.py` checks every `[*.md]` citation against the filenames that were actually retrieved. A citation to any other file is removed with a warning, and an answer with no citations is flagged as unverified. The retrieved snippets and their similarity distances are printed so a person can check the answer.

**What a production government deployment would also need:**

1. **Document-level access control.** Many agency documents are restricted, for example internal procedures, personnel records, or law-enforcement material. Each chunk would carry permission metadata such as classification, department, and allowed groups. Retrieval would filter by the signed-in user's entitlements *before* anything reaches the model, so the assistant can never quote a document the user couldn't open directly. Access would be audit-logged.
2. **Evaluation and monitoring.** This would need a test set of real questions with known correct answers and sources, measured before every model, prompt, or document change. The metrics would be retrieval hit rate, answer accuracy, citation correctness, and the rate of correct "not found" replies. In production, latency, cost, refusal rates, and user feedback would be tracked, along with sampled answers for human review. Policy documents change, so the index also needs a re-ingestion and versioning process.

## 30-second talking point

> "I've built a small personal RAG prototype in Python using Gemini and the Chroma vector database. It indexes a handful of fictional municipal policy documents with Gemini embeddings, retrieves the top three relevant passages for a question, and has Gemini answer only from those passages with a citation to the source file. It checks those citations against what was actually retrieved and says 'not found' when the policies don't cover the question. My background is stronger in distributed systems and AI infrastructure, so this was a useful way to get hands-on with the application layer. I'd be excited to build on that in real customer deployments, where things like document-level access control and evaluation matter."
