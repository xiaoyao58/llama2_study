
import random
import pandas as pd
import numpy as np
import json
import pickle
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer
from typing import List, Dict, Tuple
import threading
from multiprocessing import Pool,Manager,cpu_count
from queue import Queue
import logging
from tqdm import tqdm

class SFTDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512, prompt_max_len=256, answer_max_len=256):
        """
        初始化SFT数据集类
        
        参数:
            df: 包含训练数据的DataFrame，应有'prompt'和'answer'列
            tokenizer: 用于文本编码的分词器
            max_length: 输入序列的最大长度
            prompt_max_len: prompt部分的最大长度
            answer_max_len: answer部分的最大长度
        """
        super().__init__()
        self.df = df
        self.max_length = max_length  # 输入序列总最大长度
        self.prompt_max_len = prompt_max_len  # prompt部分最大长度
        self.answer_max_len = answer_max_len  # answer部分最大长度
        
        # 初始化分词器和特殊token
        self.tokenizer = tokenizer
        self.bos = self.tokenizer.special_tokens['<bos>']  # 开始token
        self.eos = self.tokenizer.special_tokens['<eos>']  # 结束token
        self.pad = 0  # 填充token，这里设为0
        
    def __len__(self):
        """返回数据集中的样本数量"""
        return self.df.shape[0]
    
    def __getitem__(self, index: int):
        """
        获取单个样本并进行预处理
        
        返回:
            X: 输入序列（去掉最后一个token）
            Y: 目标序列（去掉第一个token）
            loss_mask: 损失掩码，决定哪些位置参与损失计算
        """
        # 获取当前样本
        sample = self.df.iloc[index]
        
        # 编码prompt和answer部分
        prompt = self.tokenizer.encode(sample['prompt'], add_special_tokens=False)
        answer = self.tokenizer.encode(sample['answer'], add_special_tokens=False)
        
        # 截断过长的prompt和answer
        if len(prompt) > self.prompt_max_len:
            prompt = prompt[:self.prompt_max_len-2]  # 保留空间给特殊token
        if len(answer) > self.answer_max_len:
            answer = answer[:self.answer_max_len-2]
        
        # 构建完整输入序列: prompt + <bos> + answer + <eos>
        input_id = prompt + [self.bos] + answer + [self.eos]
        
        # 确定上下文长度（<bos>之前的部分）
        context_length = input_id.index(self.bos)
        # 掩码位置是<bos>前一个位置
        mask_position = context_length - 1
        
        # 填充到最大长度
        pad_len = self.max_length - len(input_id)
        input_id = input_id + [self.pad] * pad_len
        
        # 创建损失掩码:
        # - prompt部分不计算损失（0）
        # - answer部分计算损失（1）
        # - 填充部分不计算损失（0）
        if pad_len == 0:
            loss_mask = [0]*context_length + [1]*(len(input_id[mask_position+1:])) + [0]*pad_len
        else:
            loss_mask = [0]*context_length + [1]*(len(input_id[mask_position+1:-pad_len])) + [0]*pad_len
        
        # 转换为numpy数组
        input_id = np.array(input_id)
        # X是输入序列（去掉最后一个token）
        X = np.array(input_id[:-1]).astype(np.int64)
        # Y是目标序列（去掉第一个token，实现位移）
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[:-1])  # 损失掩码对齐X的长度
        
        # 转换为PyTorch张量
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SFTDataProcessor:
    def __init__(self, tokenizer_path: str="./chatglm_tokenizer/tokenizer.model", num_workers: int = 4):
        """
        初始化数据处理器（多进程版本）
        
        :param tokenizer_path: 分词器模型路径
        :param max_length: 最大序列长度
        :param num_workers: 并行进程数（默认使用CPU核心数）
        """
        self.tokenizer_path = tokenizer_path
        self.num_workers = num_workers if num_workers else cpu_count()
        
        # 预加载特殊token（主进程）
        temp_tokenizer = ChatGLMTokenizer(vocab_file=tokenizer_path)
        self.bos = temp_tokenizer.special_tokens['<bos>']
        self.eos = temp_tokenizer.special_tokens['<eos>']
        self.pad = 0

    def process_jsonl(self, jsonl_path: str) -> List[Dict[str, str]]:
        """
        从JSONL文件读取并预处理数据（单进程）
        """
        processed_data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc='读取JSONL文件', unit='lines'):
                try:
                    data = json.loads(line)
                    conversations = data.get('conversations', [])
                    
                    user_content = next(
                        (conv['content'] for conv in conversations if conv['role'] == 'user'), 
                        ""
                    )
                    assistant_content = next(
                        (conv['content'] for conv in conversations if conv['role'] == 'assistant'), 
                        ""
                    )
                    
                    if user_content and assistant_content:
                        processed_data.append({
                            'prompt': user_content,
                            'answer': assistant_content
                        })
                except Exception as e:
                    print(data)
                    logger.warning(f"跳过错误行: {e}")
        return processed_data

    def _init_process(self, *args):
        """每个进程初始化时创建独立的分词器实例"""
        global _process_tokenizer
        _process_tokenizer = ChatGLMTokenizer(vocab_file=self.tokenizer_path)

    def _process_sample(self, sample: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """处理单个样本（在每个进程中执行）"""
        try:
            # 使用进程内独立的分词器
            prompt_ids = _process_tokenizer.encode(sample['prompt'], add_special_tokens=False)
            answer_ids = _process_tokenizer.encode(sample['answer'], add_special_tokens=False)
            return prompt_ids,answer_ids
        except Exception as e:
            logger.error(f"处理样本出错: {e}")
            return None

    def save_to_bin(self, processed_data: List[Dict[str, str]], output_path: str):
        """
        多进程处理数据并保存为二进制文件
        
        :param processed_data: 处理后的数据
        :param output_path: 输出文件路径
        """
        total_samples = len(processed_data)
        logger.info(f"开始处理 {total_samples} 个样本（{self.num_workers}进程）")
        
        # 初始化结果数组
        prompt_ids_list = []
        answer_ids_list = []
        original_indices = []
        
        # 使用多进程池
        with Pool(
            processes=self.num_workers,
            initializer=self._init_process,
            initargs=(self.tokenizer_path,)
        ) as pool:
            # 使用imap_unordered获取结果（更快）
            results = list(tqdm(
                pool.imap_unordered(self._process_sample, processed_data),
                total=total_samples,
                desc='处理样本',
                unit='samples',
                dynamic_ncols=True
            ))
        
        # 收集结果（注意：imap_unordered顺序可能打乱，需要按原始顺序重组）
        for idx, result in enumerate(results):
            if result is not None:
                prompt_ids,answer_ids = result
                prompt_ids_list.append(prompt_ids)
                answer_ids_list.append(answer_ids)
                original_indices.append(idx)

        prompt_concatenated = np.concatenate(prompt_ids_list) if prompt_ids_list else np.array([], dtype=np.uint16)
        answer_concatenated = np.concatenate(answer_ids_list) if answer_ids_list else np.array([],dtype=np.uint16)

        lengths_prompt = np.array([len(p) for p in prompt_ids_list],dtype=np.uint16)
        lengths_answer = np.array([len(a) for a in answer_ids_list],dtype=np.uint16)

        num_samples = len(prompt_ids_list)
        if len(answer_ids_list) != num_samples:
            raise ValueError("Number of questions and answers must match!")
        
        # 保存结果
        data_to_save = {
            'prompt_concatenated': prompt_concatenated,
            'answer_concatenated': answer_concatenated,
            'lengths_prompt':lengths_prompt,
            'lengths_answer':lengths_answer,
            'num_samples':num_samples,
            'bos':self.bos,
            'eos':self.eos,
        }
        np.savez_compressed(output_path, **data_to_save)
        print(f"Data saved to {output_path}")

class SFTBinDataset(Dataset):
    """从二进制文件加载数据的Dataset（保持不变）"""
    def __init__(self, npz_path: str,max_length: int=512,debug=False):
        super().__init__()
        loaded_data = np.load(npz_path)
        self.debug = debug
        self.bos = loaded_data['bos']
        self.eos = loaded_data['eos']
        self.pad = 0
        self.prompt_concatenated = loaded_data['prompt_concatenated']
        self.answer_concatenated = loaded_data['answer_concatenated']
        self.lengths_prompt = loaded_data['lengths_prompt']
        self.lengths_answer = loaded_data['lengths_answer']
        self.num_samples = loaded_data['num_samples']
        self.max_length = max_length
        self.tokenizer=ChatGLMTokenizer(vocab_file='./chatglm_tokenizer/tokenizer.model')

        def calculate_indices(lengths):
            end_indices = np.cumsum(lengths)
            start_indices = np.zeros_like(end_indices)
            start_indices[1:] = end_indices[:-1] # 开始索引是前一个的结束索引
            return start_indices, end_indices

        self.start_indices_prompt, self.end_indices_prompt = calculate_indices(self.lengths_prompt)
        self.start_indices_answer, self.end_indices_answer = calculate_indices(self.lengths_answer)

    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index: int):
        prompt_ids = self.prompt_concatenated[self.start_indices_prompt[index]:self.end_indices_prompt[index]].tolist()
        answer_ids = self.answer_concatenated[self.start_indices_answer[index]:self.end_indices_answer[index]].tolist()

        if len(prompt_ids)+len(answer_ids)>self.max_length-2:
                prompt = self.tokenizer.decode(prompt_ids)
                answer = self.tokenizer.decode(answer_ids)
                print("prompt: "+prompt)
                print("answer: "+answer)
                print("="*100)
                half_max_length = int(self.max_length/2)
                if len(prompt_ids)>half_max_length:
                    prompt_ids = prompt_ids[:half_max_length-1]
                if len(answer_ids)>half_max_length:
                    answer_ids = answer_ids[:half_max_length-1]
        if self.debug:
            prompt = self.tokenizer.decode(prompt_ids)
            answer = self.tokenizer.decode(answer_ids)
            print("prompt: "+prompt)
            print("answer: "+answer)
            print("="*100)
        # 构建完整序列
        input_ids = prompt_ids + [self.bos] + answer_ids + [self.eos]
        # 上下文处理
        context_length = input_ids.index(self.bos)
        mask_position = context_length - 1
        
        # 填充
        pad_len = self.max_length - len(input_ids)
        input_ids = input_ids + [self.pad] * pad_len
        
        # 损失掩码
        loss_mask = (
            [0]*context_length + 
            [1]*(len(input_ids[mask_position+1:-pad_len] if pad_len > 0 else input_ids[mask_position+1:])) + 
            [0]*pad_len
        )
        
        # 转换为numpy
        input_ids = np.array(input_ids, dtype=np.int64)
        X = input_ids[:-1]
        Y = input_ids[1:]
        loss_mask = np.array(loss_mask[:-1])
        return X,Y,loss_mask
    
def ConvertJsonlToBin(input_path,output_path):
     # 1. 初始化处理器
    processor = SFTDataProcessor()
    
    # 2. 处理JSONL文件
    processed_data = processor.process_jsonl(input_path)
    
    # 3. 保存为二进制文件
    processor.save_to_bin(processed_data, output_path)

if __name__=="__main__":
    input_path = "data/sft_mini_512.jsonl"
    output_path = "data/sft_mini_512.npz"
    ConvertJsonlToBin(input_path,output_path)

    # 测试创建数据集实例
    train_ds = SFTBinDataset("data/sft_mini_512.npz",debug=False)
    for idx,ds in enumerate(train_ds):
        # print(idx)
        pass
    # train_ds[0]
    # train_ds[1]
    # train_ds[2]
    # # 创建数据加载器
    # train_loader = torch.utils.data.DataLoader(
    #     train_ds,
    #     batch_size=1,
    #     pin_memory=False,
    #     drop_last=False,
    #     shuffle=False,        
    #     num_workers=0,
    # )
    
    # # 测试数据加载
    # for i, (X, Y, loss_mask) in enumerate(train_loader):
    #     print("输入和目标形状:", X.shape, Y.shape)
    #     print("第一个样本的输入:", X[0])
    #     print("第一个样本的目标:", Y[0])
    #     print("第一个样本的损失掩码:", loss_mask[0])
    #     break  # 只查看第一个样本

