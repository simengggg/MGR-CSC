import json
import random
import time
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor, as_completed

# 输入输出文件
input_file = r"C:\Users\86137\Desktop\random_queries.jsonl"
output_file = 'translated_queries.jsonl'

# 语言映射
language_map = {
    "中文": "zh-CN",
    "英语": "en",
    "俄语": "ru",
    "泰语": "th",
    "西班牙语": "es",
    "孟加拉语": "bn",
    "印地语": "hi"
}
TARGET_LANGUAGES = list(language_map.values())

# 加载数据
with open(input_file, 'r', encoding='utf-8') as f:
    queries = [json.loads(line) for line in f]

# 翻译函数，带重试
def translate_item(item, max_retries=3):
    target_lang = random.choice(TARGET_LANGUAGES)
    for attempt in range(max_retries):
        try:
            translated_query = GoogleTranslator(source='auto', target=target_lang).translate(item['text'])
            return {
                "text_id": item['text_id'],
                "text": translated_query
            }
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)  # 稍等再重试
            else:
                print(f"[失败] query_id={item['text_id']} 翻译失败，保留原文。错误信息: {e}")
                return {
                    "text_id": item['text'],
                    "text": item['text']
                }

# 多线程执行
translated = []
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(translate_item, item) for item in queries]
    for future in as_completed(futures):
        translated.append(future.result())

# 保存到文件
with open(output_file, 'w', encoding='utf-8') as f:
    for item in translated:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"共处理 {len(translated)} 条 query，结果已保存到 {output_file}")
