rom transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

model_name = "openai/whisper-large-v2"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

dataset = load_dataset("common_voice", "en", split="test[:1%]")
dataset = dataset.map(lambda batch: processor(batch["audio"]["array"], sampling_rate=16000), remove_columns=["audio"])

def transcribe(batch):
    inputs = processor(batch["input_values"], return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    batch["transcription"] = transcription[0]
    return batch

transcribed_dataset = dataset.map(transcribe, batched=True, batch_size=16)
transcriptions = transcribed_dataset["transcription"]
print(transcriptions)
evaluate.py

