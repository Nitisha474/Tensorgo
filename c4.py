dummy_document = "This is a sample document used to demonstrate the query capability of RAG."
print(f"Dummy Document: {dummy_document}")

response = query_rag(dummy_document)
print(f"Query Response: {response}")