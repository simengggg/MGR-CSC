import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from torch.multiprocessing import Pool, set_start_method

model = SentenceTransformer('/paraphrase-multilingual-MiniLM-L12-v2')
#device_count = torch.cuda.device_count()
device_count = 1
keywords = []

with open("/data_msmarco/cache_data/msmarco_keywords_100K_list.jsonl", "r") as f:
    for lines in f.readlines():
        lines = json.loads(lines)
        for line in lines["keywords"]:
            keywords.append(line)
keywords = list(set(keywords))  
    


print(f"Total unique keywords: {len(keywords)}")


def compute_embeddings(keywords, device_id):
    torch.cuda.set_device(device_id)
    local_model = model.to(f"cuda:{device_id}")
    
    
    embeddings = local_model.encode(keywords, show_progress_bar=True, convert_to_tensor=True, device=f"cuda:{device_id}")
    return embeddings.cpu().numpy()


def parallel_embeddings(keywords, num_gpus):
    chunks = np.array_split(keywords, num_gpus)
    with Pool(processes=num_gpus) as pool:
        embeddings = pool.starmap(compute_embeddings, [(chunk, i) for i, chunk in enumerate(chunks)])
    return np.vstack(embeddings)


def compute_similarity(embeddings, threshold=0.8):
    n = len(embeddings)
    clusters = []
    visited = np.zeros(n, dtype=bool)

    for i in range(n):
        if not visited[i]:
            cluster = [i]
            visited[i] = True
            for j in range(i + 1, n):
                if not visited[j]:
                    similarity = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                    if similarity > threshold:
                        cluster.append(j)
                        visited[j] = True
            clusters.append(cluster)
    return clusters


def save_clusters(clusters, keywords, output_file):
    with open(output_file, "w") as f:
        for cluster in clusters:
            cluster_keywords = [keywords[i] for i in cluster]
            f.write(json.dumps(cluster_keywords, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    try:
        set_start_method('spawn')  
    except RuntimeError:
        pass

    print(f"Using {device_count} GPUs for computation...")

 
    embeddings = parallel_embeddings(keywords, device_count)

  
    print("Clustering keywords...")
    clusters = compute_similarity(embeddings, threshold=0.9)


    output_file = "mnq320k_keywords_6keywords.json"
    save_clusters(clusters, keywords, output_file)

    print(f"Keyword clusters saved to {output_file}")
