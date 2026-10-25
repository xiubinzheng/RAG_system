"""Index data/*.md into a persistent Chroma collection using Gemini embeddings.

Usage: python ingest.py
"""

import sys
import time

import rag


def main() -> int:
    start = time.perf_counter()
    try:
        records = rag.load_chunks()
        if not records:
            raise rag.RAGError(f"No Markdown files found in {rag.DATA_DIR}.")
        client = rag.get_client()

        # Prefix each chunk with its document title so the embedding knows the topic.
        texts = [f"{r['metadata']['title']}\n\n{r['text']}" for r in records]
        print(f"Embedding {len(records)} chunks with {rag.EMBED_MODEL}...")
        vectors = rag.embed(client, texts, task_type="RETRIEVAL_DOCUMENT")

        collection = rag.get_collection(create=True)
        ids = [r["id"] for r in records]
        collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=[r["text"] for r in records],
            metadatas=[r["metadata"] for r in records],
        )

        # Remove chunks from edited or deleted files so the index matches data/ exactly.
        stale = set(collection.get(include=[])["ids"]) - set(ids)
        if stale:
            collection.delete(ids=sorted(stale))
    except rag.RAGError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    files = sorted({r["metadata"]["source"] for r in records})
    print(f"Indexed {len(records)} chunks from {len(files)} files: {', '.join(files)}")
    if stale:
        print(f"Removed {len(stale)} stale chunks.")
    print(f"Collection now holds {collection.count()} chunks. ({time.perf_counter() - start:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
