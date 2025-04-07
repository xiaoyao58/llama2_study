import json

# def view_jsonl_head(file_path, n=5):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for i, line in enumerate(f):
#             if i >= n:
#                 break
#             print(line.strip())  # 去除换行符并打印

# # 使用示例
# view_jsonl_head('data/sft_mini_512.jsonl', 5)


def view_json_head(file_path, n=5):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
        if isinstance(data, list):
            # 如果是JSON数组，打印前n项
            print(json.dumps(data[:n], indent=2, ensure_ascii=False))
        elif isinstance(data, dict):
            # 如果是JSON对象，打印前n个键值对
            items = list(data.items())[:n]
            print(json.dumps(dict(items), indent=2, ensure_ascii=False))
        else:
            # 其他类型直接打印
            print(data)

# 使用示例
view_json_head('data/pretrain_hq.json', 3)


