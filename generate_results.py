"""
Script to test the FastAPI endpoint and generate results.json with REAL query results.
Validates the retrieval pipeline end-to-end.
"""

import json
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_validation():
    print("Testing API validations...")
    # 1. Test empty query
    res = client.post("/query", json={"query": "", "top_k": 3})
    assert res.status_code == 422, f"Expected 422 for empty query, got {res.status_code}"
    print("[PASS] Empty query validation passed (422)")

    # 2. Test whitespace-only query
    res = client.post("/query", json={"query": "   ", "top_k": 3})
    assert res.status_code == 422, f"Expected 422 for whitespace query, got {res.status_code}"
    print("[PASS] Whitespace query validation passed (422)")

    # 3. Test top_k <= 0
    res = client.post("/query", json={"query": "machine learning", "top_k": 0})
    assert res.status_code == 422, f"Expected 422 for top_k=0, got {res.status_code}"
    print("[PASS] Non-positive top_k validation passed (422)")

    # 4. Test top_k > 20
    res = client.post("/query", json={"query": "machine learning", "top_k": 25})
    assert res.status_code == 422, f"Expected 422 for top_k>20, got {res.status_code}"
    print("[PASS] Maximum top_k limit validation passed (422)")


def run_queries_and_save_results():
    queries = [
        {
            "query": "What is the difference between AI, Machine Learning, and Deep Learning?",
            "top_k": 3
        },
        {
            "query": "How does linear regression work?",
            "top_k": 3
        },
        {
            "query": "What are model evaluation metrics?",
            "top_k": 3
        },
        {
            "query": "What is the difference between supervised and unsupervised learning?",
            "top_k": 3
        }
    ]

    all_results = []
    print("\nExecuting real queries against vector store via POST /query...")

    for item in queries:
        q_text = item["query"]
        k_val = item["top_k"]
        print(f"\nQuerying: '{q_text}' (top_k={k_val})")
        response = client.post("/query", json={"query": q_text, "top_k": k_val})
        assert response.status_code == 200, f"Query failed: {response.text}"
        data = response.json()
        assert isinstance(data, list), "Response must be a list of results"
        assert len(data) > 0, "Expected at least one matching chunk"
        for chunk in data:
            assert "chunk_text" in chunk, "Result missing chunk_text"
            assert "page_number" in chunk, "Result missing page_number"
            assert "score" in chunk, "Result missing score"
            assert isinstance(chunk["page_number"], int), "page_number must be int"
            assert isinstance(chunk["score"], (float, int)), "score must be numeric"

        print(f"  -> Retrieved {len(data)} chunks. Top score: {data[0]['score']}, Page: {data[0]['page_number']}")
        all_results.append({
            "query": q_text,
            "top_k": k_val,
            "results": data
        })

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n[PASS] Successfully written results to results.json")


if __name__ == "__main__":
    test_validation()
    run_queries_and_save_results()
