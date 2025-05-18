import json
from rank_bm25 import BM25Okapi
from langdetect import detect
from collections import defaultdict
from tqdm import tqdm
import numpy as np

def tokenize(text):
    return text.lower().split()

def load_data(train_path, dev_path):
    corpus, doc_ids = [], []
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            corpus.append(item['text'])
            doc_ids.append(item['text_id'])

    queries, query_ids = [], []
    with open(dev_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            queries.append(item['text'])
            query_ids.append(item['text_id'])

    return corpus, doc_ids, queries, query_ids

def evaluate(corpus, doc_ids, queries, query_ids, lang_list):
    tokenized_corpus = [tokenize(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    lang_result = {lang: {'correct@1': 0, 'correct@10': 0, 'total': 0} for lang in lang_list}

    for i, (query, qid) in enumerate(tqdm(zip(queries, query_ids), total=len(queries))):
        try:
            lang = detect(query)
        except:
            continue
        if lang not in lang_list:
            continue

        scores = bm25.get_scores(tokenize(query))
        top10_idx = np.argsort(scores)[-10:][::-1]
        top10_doc_ids = [doc_ids[j] for j in top10_idx]

        lang_result[lang]['total'] += 1
        if qid == top10_doc_ids[0]:
            lang_result[lang]['correct@1'] += 1
        if qid in top10_doc_ids:
            lang_result[lang]['correct@10'] += 1

    r1_total, r10_total, count = 0, 0, 0
    for lang in lang_list:
        total = lang_result[lang]['total']
        r1 = lang_result[lang]['correct@1'] / total if total else 0
        r10 = lang_result[lang]['correct@10'] / total if total else 0
        print(f"{lang}: recall@1 = {r1:.4f}, recall@10 = {r10:.4f}")
        r1_total += r1
        r10_total += r10
        count += 1

    print(f"avg: recall@1 = {r1_total/count:.4f}, recall@10 = {r10_total/count:.4f}")

def main():
    train_path = "mmarco_train_10.json"
    dev_path = "mmarco_dev.jsonl"
    lang_list = ['en', 'fr', 'de', 'it', 'es', 'ja','zh-cn']
    corpus, doc_ids, queries, query_ids = load_data(train_path, dev_path)
    evaluate(corpus, doc_ids, queries, query_ids, lang_list)

if __name__ == "__main__":
    main()
