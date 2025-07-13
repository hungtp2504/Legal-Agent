from sentence_transformers import SentenceTransformer

model_name = "bkai-foundation-models/vietnamese-bi-encoder"

print(f"Downloading and caching model: {model_name}")

SentenceTransformer(model_name)

print("Model cached successfully.")
