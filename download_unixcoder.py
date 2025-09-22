from transformers import AutoModel, AutoTokenizer

# Download the model files
tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base-nine")
model = AutoModel.from_pretrained("microsoft/unixcoder-base-nine")

# Save them to your local path
model.save_pretrained('./unixcoder-nine')
tokenizer.save_pretrained('./unixcoder-nine')