from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
import json

device = "cuda"  # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "llama3-8b-instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("llama3-8b-instruct")

def truncate_after_word(input_string, word):
    position = input_string.find(word)
    if position != -1:
        return input_string[position:]
    else:
        return ""

with open("data/train.json", "r", encoding="utf-8") as f_zh:
    data_zh = json.load(f_zh) 

for text_content in data_zh:
    try:
        messages = [
            {
                "role": "system",
                "content": "Please generate 10 Chinese questions based on the following text. Use \";\" to separate the questions. Only generate questions, do not generate other content."
            },
            {
                "role": "user",
                "content": text_content
            }
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=64,
            temperature = 0.7
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        result = {
            "text": text_content,
            "keywords": response.strip()
        }

        with open("query_train_zh.jsonl", "a", encoding="utf-8") as zh:
            json.dump(result, zh, ensure_ascii=False)
            zh.write("\n")

    except Exception as e:
        print(f"An unexpected error occurred while processing entry: {text_content[:30]}... Error: {e}")
