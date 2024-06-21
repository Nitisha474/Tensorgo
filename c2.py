def query_rag(transcription):
    inputs = rag_tokenizer(transcription, return_tensors="pt")
    generated = rag_model.generate(input_ids=inputs["input_ids"])
    response = rag_tokenizer.batch_decode(generated, skip_special_tokens=True)
    return response[0]

dummy_transcription = transcriptions[0]
response = query_rag(dummy_transcription)
print(f"RAG Response: {response}")