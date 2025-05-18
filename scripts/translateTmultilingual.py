import os
import concurrent.futures
from deep_translator import GoogleTranslator
import time


TARGET_LANGUAGES = ["af", "ar", "fr", "hi", "ja", "mk", "sv", "vi"]


def translate_text(text, target_lang, max_retries=4):
    for attempt in range(max_retries):
        try:
            translated_text = GoogleTranslator(source="en", target=target_lang).translate(text)
            return translated_text 
        except Exception as e:
            print(f"⚠️ translate failing (try {attempt + 1}/{max_retries}, lanuguage: {target_lang}), error: {e}")
            time.sleep(2)  
    return None  

def load_translated_lines(output_path):
    translated_ids = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    translated_ids.add(parts[0])  
    return translated_ids


def translate_and_save(input_path, output_path, max_threads=12):
    translated_ids = load_translated_lines(output_path)  

    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "a", encoding="utf-8") as outfile:
        infile = infile.readlines()
        with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
            futures = {}

            for line in infile:
                parts = line.strip().split("\t", 1)
                if len(parts) != 2:
                    continue  
                
                index, text = parts
                if index in translated_ids:
                    print(f"✅ index {index} translated, skipped...")
                    continue  

                for target_lang in TARGET_LANGUAGES:
                    future = executor.submit(translate_text, text, target_lang)
                    futures[future] = (index, text, target_lang)

            for future in concurrent.futures.as_completed(futures):
                index, original_text, target_lang = futures[future]
                translated_text = future.result()

                if translated_text is None:  
                    translated_text = original_text  
                    print(f"⚠️  index {index} translate to {target_lang} failed, save original document")

                outfile.write(f"{index}\t{translated_text}\n")
                outfile.flush()  

    print(f"✅ translate complete, result save as {output_path}")

# 运行
input_file = "original_data.tsv"   
output_file = "translate_multilingual_data.tsv" 
translate_and_save(input_file, output_file)