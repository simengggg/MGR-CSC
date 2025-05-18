import json
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from langdetect import detect
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LaBSEEncoder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("LaBSE")
        self.model = AutoModel.from_pretrained("LaBSE").to(device)
        self.model.eval()

    def encode(self, texts, batch_size=32):
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
                batch = texts[i:i+batch_size]
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                outputs = self.model(**inputs)
                pooled = outputs.last_hidden_state[:, 0]  # CLS token
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                embeddings.append(pooled.cpu())
        return torch.cat(embeddings, dim=0)

def load_data(train_path, dev_path):
    docs, doc_ids = [], []
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            docs.append(item['text'])          
            doc_ids.append(item['text_id'])

    queries, query_ids = [], []
    with open(dev_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            queries.append(item['text'])
            query_ids.append(item['text_id'])

    return docs, doc_ids, queries, query_ids

def evaluate(docs, doc_ids, queries, query_ids, encoder, lang_list):
    doc_embeds = encoder.encode(docs)
    query_embeds = encoder.encode(queries)

    scores = cosine_similarity(query_embeds, doc_embeds)
    top10 = np.argsort(scores, axis=1)[:, -10:][:, ::-1]

    lang_result = {lang: {'correct@1': 0, 'correct@10': 0, 'total': 0} for lang in lang_list}

    for i, (qid, qtext) in enumerate(zip(query_ids, queries)):
        try:
            lang = detect(qtext)
        except:
            continue
        if lang not in lang_list:
            continue

        lang_result[lang]['total'] += 1
        top10_doc_ids = [doc_ids[j] for j in top10[i]]
        if qid == top10_doc_ids[0]:
            lang_result[lang]['correct@1'] += 1
        if qid in top10_doc_ids:
            lang_result[lang]['correct@10'] += 1

    r1_total, r10_total, count = 0, 0, 0
    for lang in lang_list:
        total = lang_result[lang]['total']
        r1 = lang_result[lang]['correct@1'] / total if total > 0 else 0
        r10 = lang_result[lang]['correct@10'] / total if total > 0 else 0
        print(f"{lang}: recall@1 = {r1:.4f}, recall@10 = {r10:.4f}")
        r1_total += r1
        r10_total += r10
        count += 1

    print(f"avg: recall@1 = {r1_total/count:.4f}, recall@10 = {r10_total/count:.4f}")

def main():
    train_path = "mmarco_train_dpr.json"
    dev_path = "mmarco_dev.jsonl"
    lang_list = ['en', 'fr', 'de', 'it', 'es', 'ja','zh-cn']
    encoder = LaBSEEncoder()
    docs, doc_ids, queries, query_ids = load_data(train_path, dev_path)
    evaluate(docs, doc_ids, queries, query_ids, encoder, lang_list)

if __name__ == "__main__":
    main()
