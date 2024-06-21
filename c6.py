import evaluate

wer = evaluate.load("wer")
transcriptions = transcribed_dataset["transcription"]
references = transcribed_dataset["sentence"]
error_rate = wer.compute(predictions=transcriptions, references=references)
print(f"Word Error Rate (WER): {error_rate}")