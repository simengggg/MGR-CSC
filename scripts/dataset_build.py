import json

def process_all_files(cluster_file_path, text_file_path, file3_path, file4_path, output1_path, output2_path):
    clusters = []
    with open(cluster_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                cluster = json.loads(line)
                clusters.append(cluster)
            except json.JSONDecodeError as e:
                print(f"error cluster: {line}")
                raise e
    
    text_data = []
    with open(text_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                text_data.append(item)
            except json.JSONDecodeError as e:
                print(f"error text_1: {line}")
                raise e
    
    file3_data = []
    with open(file3_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                file3_data.append(item)
            except json.JSONDecodeError as e:
                print(f"error text_3: {line}")
                raise e
    
    file4_data = []
    with open(file4_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                file4_data.append(item)
            except json.JSONDecodeError as e:
                print(f"error text_4: {line}")
                raise e
    
    text_id_to_cluster = {}
    for cluster_index, cluster in enumerate(clusters, 1):  
        for text_id in cluster:
            text_id_to_cluster[text_id] = cluster_index
    
    results1 = []
    for item in text_data:
        if "text_id" not in item or "text" not in item:
            continue
            
        text_ids = item["text_id"]
        if not isinstance(text_ids, list):
            if isinstance(text_ids, str):
                text_ids = [text_ids]
            else:
                continue
                
        cluster_numbers = []
        for text_id in text_ids:
            if text_id in text_id_to_cluster:
                cluster_numbers.append(str(text_id_to_cluster[text_id]))
            else:
                cluster_numbers.append("0")
        

        new_item = {
            "text_id": " ".join(cluster_numbers),
            "text": item["text"]
        }
        results1.append(new_item)
    
    with open(output1_path, 'w', encoding='utf-8') as f:
        for item in results1:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    text_to_cluster_numbers = {}
    
    for item in text_data:
        if "text_id" not in item or "text" not in item:
            continue
            
        text_ids = item["text_id"]
        if not isinstance(text_ids, list):
            if isinstance(text_ids, str):
                text_ids = [text_ids]
            else:
                continue
                
        cluster_numbers = []
        for text_id in text_ids:
            if text_id in text_id_to_cluster:
                cluster_numbers.append(str(text_id_to_cluster[text_id]))
            else:
                cluster_numbers.append("0")
        
        text_content = item["text"]
        text_to_cluster_numbers[text_content] = " ".join(cluster_numbers)
    
    file3_id_to_cluster = {}
    
    for item in file3_data:
        if "text_id" not in item or "text" not in item:
            continue
            
        text_content = item["text"]
        text_id = item["text_id"]
        
        if text_content in text_to_cluster_numbers:
            file3_id_to_cluster[text_id] = text_to_cluster_numbers[text_content]

    results2 = []
    for item in file4_data:
        if "text_id" not in item or "text" not in item:
            continue
            
        text_id = item["text_id"]
        text_content = item["text"]
        
        cluster_numbers = file3_id_to_cluster.get(text_id, "0")
        
        new_item = {
            "text_id": cluster_numbers,
            "text": text_content
        }
        results2.append(new_item)

    with open(output2_path, 'w', encoding='utf-8') as f:
        for item in results2:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return results1, results2


cluster_file_path = "cluster_result.jsonl" 
text_file_path = "mnq320k_queriers_6keywords.json"     
file3_path = "trainTquery.json"         
file4_path = "dev.json"         
output1_path = "train_dataset_other_similarity_or_keywords.jsonl"    
output2_path = "dev_dataset_other_similarity_or_keywords.jsonl"     

try:
    results1, results2 = process_all_files(cluster_file_path, text_file_path, file3_path, file4_path, output1_path, output2_path)
    for i in range(min(3, len(results1))):
        print(results1[i])
        
    for i in range(min(3, len(results2))):
        print(results2[i])
except Exception as e:
    import traceback
    print(traceback.format_exc())
