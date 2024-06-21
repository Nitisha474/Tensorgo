from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

rag_tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
rag_model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")
rag_retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq")