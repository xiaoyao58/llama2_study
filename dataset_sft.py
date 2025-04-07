
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
    def __init__(self, df, tokenizer, max_length=256, prompt_max_len=128, answer_max_len=128):
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

# if __name__ == "__main__":
#     # 测试代码
#     df = pd.read_csv('./data/sft_data.csv')  # 读取训练数据
#     tokenizer = ChatGLMTokenizer(vocab_file='./chatglm_tokenizer/tokenizer.model')  # 初始化分词器
    
#     # 创建数据集实例
#     train_ds = SFTDataset(df, tokenizer, max_length=256)
    
#     # 创建数据加载器
#     train_loader = torch.utils.data.DataLoader(
#         train_ds,
#         batch_size=1,
#         pin_memory=False,
#         drop_last=False,
#         shuffle=False,        
#         num_workers=0,
#     )
    
#     # 测试数据加载
#     for i, (X, Y, loss_mask) in enumerate(train_loader):
#         print("输入和目标形状:", X.shape, Y.shape)
#         print("第一个样本的输入:", X[0])
#         print("第一个样本的目标:", Y[0])
#         print("第一个样本的损失掩码:", loss_mask[0])
#         break  # 只查看第一个样本


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SFTDataProcessor:
    def __init__(self, tokenizer_path: str="./chatglm_tokenizer/tokenizer.model", max_length: int = 256, num_workers: int = 4):
        """
        初始化数据处理器（多进程版本）
        
        :param tokenizer_path: 分词器模型路径
        :param max_length: 最大序列长度
        :param num_workers: 并行进程数（默认使用CPU核心数）
        """
        self.tokenizer_path = tokenizer_path
        self.max_length = max_length
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
            
            # 构建完整序列
            input_ids = prompt_ids + [self.bos] + answer_ids + [self.eos]
            
            # 截断
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length-1] + [self.eos]
            
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
            loss_mask = np.array(loss_mask[:-1], dtype=np.int64)
            
            return X, Y, loss_mask
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
        all_X = np.zeros((total_samples, self.max_length-1), dtype=np.int64)
        all_Y = np.zeros((total_samples, self.max_length-1), dtype=np.int64)
        all_loss_mask = np.zeros((total_samples, self.max_length-1), dtype=np.int64)
        
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
                all_X[idx], all_Y[idx], all_loss_mask[idx] = result
        
        # 保存结果
        with open(output_path, 'wb') as f:
            pickle.dump({
                'X': all_X,
                'Y': all_Y,
                'loss_mask': all_loss_mask,
                'max_length': self.max_length
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"数据已保存到 {output_path}")

class SFTBinDataset(Dataset):
    """从二进制文件加载数据的Dataset（保持不变）"""
    def __init__(self, bin_path: str):
        super().__init__()
        with open(bin_path, 'rb') as f:
            data = pickle.load(f)
        
        self.X = torch.from_numpy(data['X'])
        self.Y = torch.from_numpy(data['Y'])
        self.loss_mask = torch.from_numpy(data['loss_mask'])
        self.max_length = data['max_length']
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index: int):
        return self.X[index], self.Y[index], self.loss_mask[index]
    
def ConvertJsonlToBin(input_path,output_path):
     # 1. 初始化处理器
    processor = SFTDataProcessor(
        max_length=256
    )
    
    # 2. 处理JSONL文件
    processed_data = processor.process_jsonl(input_path)
    
    # 3. 保存为二进制文件
    processor.save_to_bin(processed_data, output_path)

if __name__=="__main__":
    # input_path = "data/sft_mini_512.jsonl"
    # output_path = "data/sft_mini_512.bin"
    # ConvertJsonlToBin(input_path,output_path)

    #     # 创建数据集实例
    train_ds = SFTBinDataset("data/sft_mini_512.bin")
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=1,
        pin_memory=False,
        drop_last=False,
        shuffle=False,        
        num_workers=0,
    )
    
    # 测试数据加载
    for i, (X, Y, loss_mask) in enumerate(train_loader):
        print("输入和目标形状:", X.shape, Y.shape)
        print("第一个样本的输入:", X[0])
        print("第一个样本的目标:", Y[0])
        print("第一个样本的损失掩码:", loss_mask[0])
        break  # 只查看第一个样本

