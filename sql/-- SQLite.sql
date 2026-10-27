-- SQLite


--Paste the ID from the ask.py output into this query:

SELECT e.embedding_id, e.created_at, m.string_value AS full_text
FROM embeddings e
JOIN embedding_metadata m ON m.id = e.id
WHERE m.key = 'chroma:document'
  AND e.embedding_id = 'sample_processing_test.md::0::93bdd2b94e72';



--To see everything indexed for that file, which shows any stale leftovers (step 5):
SELECT e.embedding_id, e.created_at, m.string_value AS full_text
FROM embeddings e
JOIN embedding_metadata m ON m.id = e.id
WHERE m.key = 'chroma:document'
  AND e.embedding_id LIKE 'sample_processing_test.md::%';