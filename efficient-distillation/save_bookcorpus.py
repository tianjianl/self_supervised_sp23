
from datasets import load_dataset

dataset = load_dataset("bookcorpus")

dataset.save_to_disk("/scratch4/cs601/tli104/bookcorpus")

