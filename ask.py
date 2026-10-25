"""Ask a question over the indexed policies and get a cited answer.

Usage: python ask.py "Can I apply for a residential parking permit online?"
"""

import sys
import textwrap
import time

import rag


def main() -> int:
    if len(sys.argv) < 2 or not " ".join(sys.argv[1:]).strip():
        print('Usage: python ask.py "YOUR QUESTION"', file=sys.stderr)
        return 2
    question = " ".join(sys.argv[1:]).strip()
    start = time.perf_counter()

    try:
        collection = rag.get_collection()
        client = rag.get_client()
        query_vector = rag.embed(client, [question], task_type="RETRIEVAL_QUERY")[0]
        chunks = rag.search(collection, query_vector, k=rag.TOP_K)
        answer = rag.generate(client, rag.build_prompt(question, chunks))
    except rag.RAGError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    retrieved = {c["source"] for c in chunks}
    answer, valid, invalid = rag.check_citations(answer, retrieved)
    if not answer:
        answer = rag.NOT_FOUND

    print(f"\nQuestion: {question}\n")
    print("Answer:")
    print(textwrap.fill(answer, width=90))
    if invalid:
        print(f"\n[warning] Removed citations to documents that were not retrieved: {', '.join(invalid)}")
    if not valid and not rag.is_not_found(answer):
        print("\n[warning] The answer contains no citations; treat it as unverified.")
    if valid:
        print(f"\nCited sources: {', '.join(valid)}")

    print("\nRetrieved excerpts (top 3, lower distance = more similar):")
    for i, c in enumerate(chunks, 1):
        snippet = c["text"][:220] + ("..." if len(c["text"]) > 220 else "")
        print(f"  {i}. [{c['source']}] distance={c['distance']:.3f}")
        print(textwrap.indent(textwrap.fill(snippet, width=86), "     "))

    print(f"\nModel: {rag.gen_model()} | Total latency: {time.perf_counter() - start:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
