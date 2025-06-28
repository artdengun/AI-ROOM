def preprocess(text, tokenizer):
    encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"]
    }