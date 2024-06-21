def transcribe(batch):
    inputs = processor(batch["input_values"], return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    batch["transcription"] = transcription[0]
    return batch

transcribed_dataset = dataset.map(transcribe, batched=True, batch_size=16)