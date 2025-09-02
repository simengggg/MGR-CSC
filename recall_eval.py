import torch
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import json
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "checkpoint-770000"  
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name).to(device)

validation_data_path = "dev.jsonl"
validation_data = []
with open(validation_data_path, "r", encoding="utf-8") as fd:
    for line in fd:
        validation_data.append(json.loads(line))
    #validation_data = json.load(fd)

def generate_predictions_with_beam_search(texts, beam_size=10):
    predictions = {}
    for data in tqdm(texts, desc="Generating predictions"):
        text_id = data["text_id"]
        input_text = data["text"]


        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                inputs["input_ids"],
                num_beams=beam_size,  
                num_return_sequences=beam_size,  
                max_length=50,  
                early_stopping=True 
            )

 
        predicted_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]


        predictions[text_id] = predicted_texts

    return predictions


def save_predictions_to_file(predictions, expected_text_ids, output_file_path):
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        for text_id, predicted_texts in predictions.items():
            expected_id = str(expected_text_ids[text_id])

            if expected_id in predicted_texts[:1]:
                output_file.write(json.dumps({
                    "text_id": text_id,
                    "recall_type": "recall@1",
                    "predicted_text": predicted_texts[0]
                }, ensure_ascii=False) + "\n")

            if expected_id in predicted_texts[:10] and expected_id not in predicted_texts[:1]:
                output_file.write(json.dumps({
                    "text_id": text_id,
                    "recall_type": "recall@10",
                    "predicted_texts": predicted_texts[:10]
                }, ensure_ascii=False) + "\n")
    print(f"recall@1 and recall@10 result save as: {output_file_path}")


expected_text_ids = {item["text_id"]: item["text_id"] for item in validation_data}
predictions = generate_predictions_with_beam_search(validation_data, beam_size=10)
output_file_path = "recall_predictions.jsonl"
save_predictions_to_file(predictions, expected_text_ids, output_file_path)

def calculate_hits_at_k(predictions, k, expected_text_ids):
    hits = 0
    total = len(predictions)

    for text_id, predicted_texts in predictions.items():
        if text_id in expected_text_ids and str(expected_text_ids[text_id]) in predicted_texts[:k]:
            hits += 1

    hits_at_k = hits / total if total > 0 else 0
    return hits_at_k

hits_at_1 = calculate_hits_at_k(predictions, k=1, expected_text_ids=expected_text_ids)
hits_at_10 = calculate_hits_at_k(predictions, k=10, expected_text_ids=expected_text_ids)

print(f"recall@1: {hits_at_1:.4f}")
print(f"recall10: {hits_at_10:.4f}")

