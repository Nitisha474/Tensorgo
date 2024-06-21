from datasets import load_dataset

# Example dataset
dataset = load_dataset("common_voice", "en", split="test[:1%]")  # Loading a small portion for testing
dataset = dataset.map(lambda batch: processor(batch["audio"]["array"], sampling_rate=16000), remove_columns=["audio"])