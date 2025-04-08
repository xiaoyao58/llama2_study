import json
import glob
import numpy as np
from tqdm import tqdm
from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer
import pandas as pd
import multiprocessing as mp
from functools import partial
import os,time
#from zhconv import convert
def process_wiki_clean():
    with open('./data/wikipedia_cn_20230720/wikipedia-cn-20230720-filtered.json','r',encoding='utf-8') as f:
        data=json.load(f)
    doc_ids=[]
    for line in tqdm(data):
        text=line['completion']
        text_id=tokenizer.encode(text,add_special_tokens=False)
        text_id.append(tokenizer.special_tokens['<eos>'])
        if len(text_id)>5:
            doc_ids+=text_id
    arr = np.array(doc_ids,dtype=np.uint16)
    with open('./data/wiki.bin','wb') as f:
        f.write(arr.tobytes())

def process_text(text, tokenizer):
    """处理单个文本，返回处理后的token IDs"""
    text_id = tokenizer.encode(text, add_special_tokens=False)
    text_id.append(tokenizer.special_tokens['<eos>'])
    if len(text_id) > 5:
        return text_id
    return []

def process_batch(batch_data, tokenizer):
    """处理一批文本"""
    batch_idx, batch = batch_data
    results = []
    for text in batch:
        result = process_text(text, tokenizer)
        results.extend(result)
    return batch_idx, results

def process_pretrain_hq(input_file="pretrain_hq.json", output_file="./data/pretrain_hq.bin", num_processes=None, batch_size=512):
    """
    使用多进程处理pretrain_hq.json文件
    
    参数:
    - input_file: 输入的JSON文件路径
    - output_file: 输出的二进制文件路径
    - num_processes: 进程数量（默认为CPU核心数）
    - batch_size: 每批处理的文本数量
    """
    start_time = time.time()
    
    # 如果未指定进程数，使用CPU核心数
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"加载数据: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    
    total_texts = len(data_list)
    print(f"总文本数量: {total_texts}")
    
    # 将数据分成多个批次
    batches = []
    for i in range(0, total_texts, batch_size):
        batches.append((i // batch_size, data_list[i:min(i+batch_size, total_texts)]))
    
    print(f"分割为 {len(batches)} 批次进行处理")
    print(f"使用 {num_processes} 个进程进行并行处理")
    
    # 使用multiprocessing进行处理
    all_ids = []
    
    # 为每个进程创建tokenizer的副本（如果tokenizer不是全局可序列化的）
    # 注意：这需要根据您的tokenizer实现进行调整
    # 此处假设tokenizer可以在多进程环境中使用
    
    with mp.Pool(processes=num_processes) as pool:
        # 创建部分函数，固定tokenizer参数
        process_fn = partial(process_batch, tokenizer=tokenizer)
        
        # 使用imap处理批次，保持顺序
        results = []
        for result in tqdm(pool.imap(process_fn, batches), total=len(batches), desc="处理批次"):
            results.append(result)
    
    # 按批次索引排序结果，确保顺序正确
    results.sort(key=lambda x: x[0])
    
    # 合并所有结果
    for _, batch_results in results:
        all_ids.extend(batch_results)
    
    print(f"处理完成，共 {len(all_ids)} 个token")
    
    # 转换为numpy数组并保存
    print("将结果转换为numpy数组...")
    arr = np.array(all_ids, dtype=np.uint16)
    
    print(f"保存结果到: {output_file}")
    with open(output_file, 'wb') as f:
        f.write(arr.tobytes())
    
    elapsed_time = time.time() - start_time
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    
    print(f"完成! 文件大小: {file_size_mb:.2f} MB")
    print(f"总处理时间: {elapsed_time:.2f} 秒")
    print(f"处理速度: {total_texts / elapsed_time:.2f} 文本/秒")

def process_medical(data_path,name):
    f=open(data_path,'r',encoding='utf-8')
    doc_ids=[]
    while True:
        line=f.readline()
        if not line:
            break
        line=json.loads(line)
        text=line['text']
        text_id=tokenizer.encode(text,add_special_tokens=False)
        text_id.append(tokenizer.special_tokens['<eos>'])
        if len(text_id)>5:
            doc_ids+=text_id
    arr = np.array(doc_ids,dtype=np.uint16)
    with open('./data/medical_{}.bin'.format(name),'wb') as f:
        f.write(arr.tobytes()) 

def sft_to_pretrain():
    doc_ids=[]

    '''
    df=pd.read_csv('./data/medical_qa_144w.csv')
    for _,q,a in tqdm(df.itertuples()):
        q_id = tokenizer.encode(q,add_special_tokens=False)
        a_id = tokenizer.encode(a,add_special_tokens=False)
        #
        print(q)
        print(a)
        print('-----')
        text_id=q_id+a_id+[tokenizer.special_tokens['<eos>']]
        if len(text_id)>5:
            doc_ids+=text_id
    '''

    with open('./data/shibing624_medical/finetune/train_en_1.json','r',encoding='utf-8') as f:
        for row in f:
            line=json.loads(row)
            q=line['input']
            a=line['output']
            q_id=tokenizer.encode(q,add_special_tokens=False)
            a_id=tokenizer.encode(a,add_special_tokens=False)
            text_id=q_id+a_id+[tokenizer.special_tokens['<eos>']]
            if len(text_id)>5:
                doc_ids+=text_id
    with open('./data/shibing624_medical/finetune/test_en_1.json','r',encoding='utf-8') as f:
        for row in f:
            line=json.loads(row)
            q=line['input']
            a=line['output']
            q_id=tokenizer.encode(q,add_special_tokens=False)
            a_id=tokenizer.encode(a,add_special_tokens=False)
            text_id=q_id+a_id+[tokenizer.special_tokens['<eos>']]
            if len(text_id)>5:
                doc_ids+=text_id
    with open('./data/shibing624_medical/finetune/valid_en_1.json','r',encoding='utf-8') as f:
        for row in f:
            line=json.loads(row)
            q=line['input']
            a=line['output']
            q_id=tokenizer.encode(q,add_special_tokens=False)
            a_id=tokenizer.encode(a,add_special_tokens=False)
            text_id=q_id+a_id+[tokenizer.special_tokens['<eos>']]
            if len(text_id)>5:
                doc_ids+=text_id

    with open('./data/shibing624_medical/finetune/train_zh_0.json','r',encoding='utf-8') as f:
        for row in f:
            line=json.loads(row)
            q=line['instruction']+line['input']
            a=line['output']
            q_id=tokenizer.encode(q,add_special_tokens=False)
            a_id=tokenizer.encode(a,add_special_tokens=False)
            text_id=q_id+a_id+[tokenizer.special_tokens['<eos>']]
            if len(text_id)>5:
                doc_ids+=text_id
    with open('./data/shibing624_medical/finetune/test_zh_0.json','r',encoding='utf-8') as f:
        for row in f:
            line=json.loads(row)
            q=line['instruction']+line['input']
            a=line['output']
            q_id=tokenizer.encode(q,add_special_tokens=False)
            a_id=tokenizer.encode(a,add_special_tokens=False)
            text_id=q_id+a_id+[tokenizer.special_tokens['<eos>']]
            if len(text_id)>5:
                doc_ids+=text_id
    with open('./data/shibing624_medical/finetune/valid_zh_0.json','r',encoding='utf-8') as f:
        for row in f:
            line=json.loads(row)
            q=line['instruction']+line['input']
            a=line['output']
            q_id=tokenizer.encode(q,add_special_tokens=False)
            a_id=tokenizer.encode(a,add_special_tokens=False)
            text_id=q_id+a_id+[tokenizer.special_tokens['<eos>']]
            if len(text_id)>5:
                doc_ids+=text_id

    arr = np.array(doc_ids,dtype=np.uint16)
    print(arr.shape)
    with open('./data/medical_qa.bin','wb') as f:
        f.write(arr.tobytes())

def process_baidu():
    BATCH_SIZE = 1000000

    cnt=0
    batch_cnt=0
    token=0
    doc_ids=[]

    f1=open('./data/563w_baidubaike/563w_baidubaike.json','r',encoding='utf-8')
    
    while True:
        line = f1.readline()
        if not line:
            break
        line=json.loads(line)
        text=''
        try:
            text+=line['title']+'：'+line['summary']
        except:
            pass
        for per in line['sections']:
            text+=per['title']+'：'+per['content']+'。'
        text_id=tokenizer.encode(text,add_special_tokens=False)
        text_id.append(tokenizer.special_tokens['<eos>'])
        if len(text_id)>5:
            doc_ids+=text_id
        cnt+=1
        if cnt%BATCH_SIZE==0:
            batch_cnt+=1
            arr = np.array(doc_ids,dtype=np.uint16)
            doc_ids=[]
            print('cnt:',cnt,'arr_shape:',arr.shape)
            with open('./data/baidubaike_563w_{}.bin'.format(batch_cnt),'wb') as f2:
                f2.write(arr.tobytes())
            del arr

    if not doc_ids:
        batch_cnt+=1
        arr = np.array(doc_ids,dtype=np.uint16)
        print('cnt:',cnt,'arr_shape:',arr.shape)
        with open('./data/baidubaike_563w_{}.bin'.format(batch_cnt),'wb') as f:
            f.write(arr.tobytes())
    
def process_c4():
    c4_zh_paths = glob.glob('./data/c4_zh/*')
    c4_zh_paths=sorted(c4_zh_paths)
    print(len(c4_zh_paths))
    cnt=0
    token=0
    doc_ids=[]
    for per in tqdm(c4_zh_paths):
        with open(per,'r') as f:
            for line in f:
                text = json.loads(line)
                text = text['text']
                text_id=tokenizer.encode(text,add_special_tokens=False)
                text_id.append(tokenizer.special_tokens['<eos>'])
                if len(text_id)>5:
                    doc_ids+=text_id
                cnt+=1

    arr = np.array(doc_ids,dtype=np.uint16)
    with open('./data/c4_zh.bin','wb') as f:
        f.write(arr.tobytes())
    print(arr.shape)

def process_wudao():
    wudao_zh_paths = glob.glob('./data/WuDaoCorpus2.0_base_200G/*')
    wudao_zh_paths=sorted(wudao_zh_paths)
    print(len(wudao_zh_paths))#很多子文件
    cnt=0
    token=0
    doc_ids=[]
    for per in tqdm(wudao_zh_paths[320:]):#wudao_zh_paths[i:j]手动分片，一片片处理，不然太大一次性处理不完
        with open(per,'r') as f:
            data=json.load(f)
            for text in data:
                text = text['title'] + text['content']
                text_id=tokenizer.encode(text,add_special_tokens=False)
                text_id.append(tokenizer.special_tokens['<eos>'])
                if len(text_id)>5:
                    doc_ids+=text_id
                #
                # if cnt%10000==0:
                #     print(cnt)
                cnt+=1
                #token+=len(text_id)
                #break
        #
        # arr = np.array(doc_ids,dtype=np.uint16)
        # with open('./data/c4-zh/{}.bin'.format(per.split('/')[-1].split('.')[0]),'wb') as f:
        #     f.write(arr.tobytes())
        # print(arr.shape)
    arr = np.array(doc_ids,dtype=np.uint16)
    with open('./data/wudaocorpus_zh_16.bin','wb') as f:
        f.write(arr.tobytes())
    print(arr.shape)

if __name__=="__main__":
    
    tokenizer = ChatGLMTokenizer(vocab_file='./chatglm_tokenizer/tokenizer.model')
    # 数据预处理-如果下载分词处理后的数据，可以不用执行以下函数
    # process_wiki_clean()
    # process_medical('./data/shibing624_medical/pretrain/medical_book_zh.json','book')
    # process_medical('./data/shibing624_medical/pretrain/train_encyclopedia.json','encyclopedia')
    # process_baidu()
    # process_c4()
    # process_wudao()
    process_pretrain_hq()

    print('data processing finished!')

    # # 分词处理后的文件列表
    data_path_list=[
        # './data/baidubaike_563w_1.bin',
        # './data/baidubaike_563w_2.bin',
        # './data/baidubaike_563w_3.bin',
        # './data/baidubaike_563w_4.bin',
        # './data/baidubaike_563w_5.bin',
        # './data/medical_book.bin',
        # './data/medical_encyclopedia.bin',
        # './data/wiki.bin',
        # './data/c4_zh_0.bin',
        # './data/c4_zh_1.bin',
        # './data/c4_zh_2.bin',
        # './data/c4_zh_3.bin',
        # './data/c4_zh_4.bin',
        # './data/c4_zh_5.bin',
        # './data/c4_zh_6.bin',
        # './data/c4_zh_7.bin',
        # './data/c4_zh_8.bin',
        # './data/wudaocorpus_zh_0.bin',
        # './data/wudaocorpus_zh_1.bin',
        # './data/wudaocorpus_zh_2.bin',
        # './data/wudaocorpus_zh_3.bin',
        # './data/wudaocorpus_zh_4.bin',
        # './data/wudaocorpus_zh_5.bin',
        # './data/wudaocorpus_zh_6.bin',
        # './data/wudaocorpus_zh_7.bin',
        # './data/wudaocorpus_zh_8.bin',
        # './data/wudaocorpus_zh_9.bin',
        # './data/wudaocorpus_zh_10.bin',
        # './data/wudaocorpus_zh_11.bin',
        # './data/wudaocorpus_zh_12.bin',
        # './data/wudaocorpus_zh_13.bin',
        # './data/wudaocorpus_zh_14.bin',
        # './data/wudaocorpus_zh_15.bin',
        # './data/wudaocorpus_zh_16.bin',
        # './data/pretrain_hf.bin',
    ]
    data_lst=[]
    for data_path in tqdm(data_path_list):
        with open(data_path,'rb') as f:
            data=np.fromfile(f,dtype=np.uint16)
            data_lst.append(data)
    arr = np.concatenate(data_lst)
    print(arr.shape)
    with open('./data/pretrain_data.bin','wb') as f:
        f.write(arr.tobytes())
