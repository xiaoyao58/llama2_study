import json
import re

def extract_conversations_to_json(input_path, output_path):
    all_conversations = []
    
    # 读取并处理JSONL文件
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            messages_object = json.loads(line.strip())
            text = messages_object["text"]
            
            # 使用正则表达式提取<s>和</s>之间的内容
            # 这种方法比简单的split更准确，能处理嵌套标签
            pattern = r'<s>(.*?)</s>'
            conversations = re.findall(pattern, text, re.DOTALL)
            
            # 将提取的对话添加到总列表中
            for conv in conversations:
                if conv.strip():  # 只添加非空内容
                    all_conversations.append(conv.strip())
    
    # 将结果保存为JSON文件
    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(all_conversations, output_file, ensure_ascii=False, indent=4)
    
    print(f"处理完成！共提取了{len(all_conversations)}条对话，并保存到{output_path}")
    
    # 返回提取的对话列表
    return all_conversations

if __name__ == "__main__":
    input_path = "pretrain_hq.jsonl"
    output_path = "pretrain_hq.json"
    conversations = extract_conversations_to_json(input_path, output_path)
    
    # 打印前几个对话作为示例
    print("\n示例对话:")
    for i, conv in enumerate(conversations[:3]):
        print(f"对话 {i+1}: {conv[:100]}..." if len(conv) > 100 else f"对话 {i+1}: {conv}")