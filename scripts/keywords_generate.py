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

with open("msmarco_DSI_train_data.json", "r", encoding="utf-8") as f_zh:
    data_zh = json.load(f_zh)

for text in data_zh:
    try:
        content = text 
        messages = [
            {
                "role": "system",
                "content": "Please generate 3 keywords based on this article, separate the keywords with \";\", answer in article language, and only generate keywords without other content."
            },
            {"role": "user", "content": content}
        ]

        text_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text_prompt], return_tensors="pt").to(device)
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=32,
            temperature=0 
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        result = {
            "text": text,
            "keywords": response.strip()
        }

        with open("msmarco_keyword_100K.jsonl", "a", encoding="utf-8") as zh:
            json.dump(result, zh, ensure_ascii=False)
            zh.write("\n")
    except Exception as e:
        print(f"Error processing text: {text[:30]}... Error: {e}")
