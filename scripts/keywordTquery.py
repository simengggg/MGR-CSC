import json


with open('/mnq320k_keywords_6keywords.json', 'r', encoding='utf-8') as f1, open('/queries.doct5.tsv', 'r', encoding='utf-8') as f2:

    file1_lines = [line.strip() for line in f1.readlines() if line.strip()]
    file2_lines = [line.strip() for line in f2.readlines() if line.strip()]

result = []

for i, file1_line in enumerate(file1_lines):
    text_id = json.loads(file1_line)
    start_index = i * 10
    end_index = (i + 1) * 10
    file2_chunk = file2_lines[start_index:end_index]
    

    for file2_line in file2_chunk:
        json_obj = {
            "text_id": text_id,
            "text": file2_line.split('\t')[1]  
        }
        result.append(json_obj)

with open('mnq320k_queriers_6keywords.json', 'w', encoding='utf-8') as output_file:
    for json_obj in result:
        json.dump(json_obj, output_file, ensure_ascii=False)                                                                                                                                 
        output_file.write('\n')

print("complete!")