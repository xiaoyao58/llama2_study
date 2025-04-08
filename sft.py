# 监督微调(SFT)脚本，用于训练Transformer模型
# 该脚本经过优化以提高多GPU训练效率

import os
import time
import math
import pickle
from contextlib import nullcontext
import numpy as np
import torch
from model import Transformer, ModelArgs
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import pandas as pd
from dataset_sft import SFTDataset, SFTBinDataset
import logging
import json
import torch.nn.functional as F
from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
import psutil

def get_logger(filename, verbosity=1, name=None):
    """
    创建并配置训练进度和指标的日志记录器
    
    参数:
        filename (str): 日志文件保存路径
        verbosity (int): 日志级别 (0: DEBUG, 1: INFO, 2: WARNING)
        name (str): 日志记录器名称
    
    返回:
        logging.Logger: 配置好的日志记录器实例
    """
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    # 文件处理器
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # 控制台处理器
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

def get_lr(it):
    """
    根据当前迭代次数计算学习率，使用带线性预热的余弦衰减调度
    
    参数:
        it (int): 当前迭代次数
    
    返回:
        float: 当前迭代的学习率
    """
    # 1) 线性预热阶段
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) 如果超过衰减迭代次数，返回最小学习率
    if it > lr_decay_iters:
        return min_lr
    # 3) 在预热和衰减之间，使用余弦衰减到最小学习率
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 系数范围0到1
    return min_lr + coeff * (learning_rate - min_lr)

def train_epoch(epoch, train_loader, train_sampler=None):
    """
    训练模型一个epoch
    
    参数:
        epoch (int): 当前epoch数
        train_loader (DataLoader): 训练数据加载器
        train_sampler (DistributedSampler, optional): 分布式采样器
    """
    # 如果是分布式训练，设置当前epoch
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
        
    model.train()
    start_time = time.time()
    total_loss = 0
    total_steps = len(train_loader)
    
    # 创建进度条 - 仅主进程显示完整进度条
    if master_process:
        pbar = tqdm(enumerate(train_loader), total=total_steps, desc=f'Epoch {epoch+1}/{max_epoch}')
    else:
        pbar = enumerate(train_loader)
    
    for step, (X, Y, loss_mask) in pbar:
        step_start = time.time()
        
        # 将数据移动到指定设备
        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)
        loss_mask = loss_mask.to(device, non_blocking=True)
        
        # 更新学习率
        lr = get_lr(epoch*iter_per_epoch+step) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # 处理分布式训练设置
        if ddp:
            model.require_backward_grad_sync = (step % gradient_accumulation_steps == gradient_accumulation_steps - 1)
            
        # 使用自动混合精度进行前向传播
        with ctx:
            logits = model(X, Y)
            # 计算带掩码的损失
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=0, reduction='none')
            loss_mask = loss_mask.view(-1)
            loss = torch.sum(loss*loss_mask)/torch.clamp(loss_mask.sum(), min=1.0)
            
        # 梯度累积实现
        loss = loss / gradient_accumulation_steps
            
        # 使用梯度缩放进行反向传播(适用于fp16训练)
        scaler.scale(loss).backward()
        
        # 完成梯度累积后更新模型
        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == total_steps:
            # 梯度裁剪
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
            # 更新模型参数
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        # 收集所有进程的损失并计算平均值
        if ddp:
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
            loss = loss / ddp_world_size
            
        # 更新进度条 - 仅主进程
        if master_process:
            step_time = time.time() - step_start
            total_loss += loss.item() * gradient_accumulation_steps  # 恢复原始损失大小
            avg_loss = total_loss / (step + 1)
            
            # 更新进度条信息
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{lr:.2e}',
                'time/step': f'{step_time:.3f}s',
                'time/epoch': f'{(time.time() - start_time)/60:.1f}min',
                'mem': f'{torch.cuda.max_memory_allocated(device) / 1e9:.2f}GB'
            })
            
            # 定期记录性能指标
            # if step % log_interval == 0:
            #     gpu_utilization = torch.cuda.utilization(ddp_local_rank if ddp else 0)
            #     cpu_utilization = psutil.cpu_percent()
            #     logger.info(f"Step {step}: GPU使用率 = {gpu_utilization}%, CPU使用率 = {cpu_utilization}%")
    
    # 返回平均损失 - 所有进程同步
    avg_loss = total_loss / total_steps
    if ddp:
        torch.distributed.all_reduce(torch.tensor([avg_loss], device=device))
        avg_loss = avg_loss / ddp_world_size
        
    return avg_loss

@torch.no_grad()
def valid_epoch(epoch, val_loader):
    """
    在验证集上评估模型
    
    参数:
        epoch (int): 当前epoch数
        val_loader (DataLoader): 验证数据加载器
    
    返回:
        float: 平均验证损失
    """
    global best_val_loss
    losses = []
    model.eval()
    
    # 仅主进程显示验证进度条
    if master_process:
        pbar = tqdm(val_loader, desc=f'Validating Epoch {epoch+1}/{max_epoch}')
    else:
        pbar = val_loader
    
    for X, Y, loss_mask in pbar:
        X = X.to(device, non_blocking=True)
        Y = Y.to(device, non_blocking=True)
        loss_mask = loss_mask.to(device, non_blocking=True)
        
        with ctx:
            logits = model(X, Y)
            # 计算带掩码的损失
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=0, reduction='none')
            loss_mask = loss_mask.view(-1)
            loss = torch.sum(loss*loss_mask)/torch.clamp(loss_mask.sum(), min=1.0)
            
        losses.append(loss.item())
        
        # 仅主进程更新进度条
        if master_process:
            pbar.set_postfix({'val_loss': f'{np.mean(losses):.4f}'})
    
    model.train()
    val_loss = np.mean(losses)
    
    # 收集所有进程的验证损失并计算平均值
    if ddp:
        torch.distributed.all_reduce(torch.tensor([val_loss], device=device))
        val_loss = val_loss / ddp_world_size
    
    # 记录验证结果并保存最佳模型 - 仅主进程
    if master_process:
        logger.info('验证损失 = {:.4f}'.format(val_loss))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info('最佳验证损失: {} 最佳epoch: {} '.format(best_val_loss, epoch))
            torch.save(raw_model.state_dict(), '{}/best.pth'.format(save_dir))
    
    return val_loss

def init_model():
    """
    初始化Transformer模型，可以从头开始或从检查点加载
    
    返回:
        torch.nn.Module: 初始化后的模型
    """
    # 模型配置
    model_args = dict(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_heads,
        vocab_size=64793,  # 词汇表大小
        multiple_of=multiple_of,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )
    
    if init_from == "scratch":
        # 从头开始初始化新模型
        if master_process:
            print("从头开始初始化新模型")
        gptconf = ModelArgs(**model_args)
        model = Transformer(gptconf)
    elif init_from == "resume":
        if master_process:
            print(f"从 {out_dir} 恢复训练")
        # 从检查点恢复训练
        ckpt_path = os.path.join(out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint["model_args"]
        # 确保这些配置属性一致
        for k in ["dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "multiple_of", "max_seq_len"]:
            model_args[k] = checkpoint_model_args[k]
        gptconf = ModelArgs(**model_args)
        model = Transformer(gptconf)
        state_dict = checkpoint["model"]
        # 处理状态字典中的前缀
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    return model

def load_pretrained_weights(model, path):
    """
    加载预训练权重并处理前缀问题
    
    参数:
        model: 要加载权重的模型
        path: 权重文件路径
    """
    state_dict = torch.load(path, map_location='cpu')
    new_state_dict = {}
    for k, v in state_dict.items():
        # 处理_orig_mod前缀
        if k.startswith('_orig_mod.'):
            new_state_dict[k[10:]] = v  # 移除'_orig_mod.'前缀
        else:
            new_state_dict[k] = v
            
    # 检查是否有权重形状不匹配
    model_state = model.state_dict()
    mismatch_keys = []
    for k, v in new_state_dict.items():
        if k in model_state and v.shape != model_state[k].shape:
            mismatch_keys.append(f"{k}: model={model_state[k].shape}, ckpt={v.shape}")
    
    if mismatch_keys and master_process:
        logger.warning(f"发现形状不匹配的权重: {mismatch_keys}")
    
    # 加载权重到模型
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    
    if master_process:
        if missing_keys:
            logger.warning(f"未从权重文件加载的密钥: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"权重文件中未使用的密钥: {unexpected_keys}")
    
    return model

def save_checkpoint(model, optimizer, epoch, save_path):
    """
    保存模型和优化器状态
    
    参数:
        model: 要保存的模型
        optimizer: 优化器
        epoch: 当前epoch
        save_path: 保存路径
    """
    # 确保只有主进程保存检查点
    if not master_process:
        return
        
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'scaler': scaler.state_dict() if scaler else None,
    }
    torch.save(checkpoint, save_path)
    logger.info(f"检查点已保存到 {save_path}")

def print_gpu_memory_stats():
    """打印所有GPU的内存统计信息"""
    if not torch.cuda.is_available() or not master_process:
        return
        
    logger.info("----- GPU内存使用情况 -----")
    for i in range(torch.cuda.device_count()):
        reserved = torch.cuda.memory_reserved(i) / 1e9
        allocated = torch.cuda.memory_allocated(i) / 1e9
        free = reserved - allocated
        logger.info(f"GPU {i}: 已分配={allocated:.2f}GB, 已保留={reserved:.2f}GB, 空闲={free:.2f}GB")
    logger.info("---------------------------")

if __name__ == "__main__":
    # 输出目录和训练配置
    out_dir = 'out/cauchy_pretrain_v21'
    max_epoch = 1
    eval_interval = 1
    log_interval = 50
    eval_iters = 200
    eval_only = False  # 如果为True，第一次评估后脚本退出
    always_save_checkpoint = True  # 如果为True，每次评估后都保存检查点
    init_from = 'scratch'  # 'scratch'从头开始或'resume'恢复训练
    
    # 训练参数
    gradient_accumulation_steps = 4  # 用于模拟更大的批大小
    per_gpu_batch_size = 16  # 每个GPU的批大小
    
    # 初始化分布式训练
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        # 设置设备
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(ddp_local_rank)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            torch.cuda.set_device(0)
    
    # 计算全局批大小
    global_batch_size = per_gpu_batch_size * ddp_world_size
    
    # 使用输入/输出预取提高效率
    prefetch_factor = 2
    num_workers = 4
    
    # 模型参数
    max_seq_len = 512
    dim = 512
    n_layers = 8
    n_heads = 8
    multiple_of = 32
    dropout = 0.1  # 预训练时0较好，微调时可尝试0.1+
    bias = False  # 是否在LayerNorm和Linear层中使用偏置
    
    # 优化器参数
    learning_rate = 1e-5  # 最大学习率
    weight_decay = 1e-4
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0  # 梯度裁剪值，0.0表示禁用
    
    # 学习率衰减设置
    decay_lr = True  # 是否衰减学习率
    warmup_iters = 1000  # 预热步数
    lr_decay_iters = 50000  # 应该约等于Chinchilla的最大迭代次数
    min_lr = 1e-6  # 最小学习率，应约等于learning_rate/10
    
    # DDP设置
    backend = 'nccl'  # 'nccl', 'gloo'等
    
    # 系统设置
    dtype = 'float16'  # 'float32', 'bfloat16'或'float16'
    compile = True  # 使用PyTorch 2.0编译模型以加速
    
    # 保存目录和日志设置
    save_dir = os.path.join(out_dir, 'sft')
    if master_process:
        os.makedirs(save_dir, exist_ok=True)
        logger = get_logger(os.path.join(save_dir, f'log_rank{ddp_rank if ddp else 0}.log'))
    else:
        # 非主进程也需要logger，但可以禁用输出
        logger = logging.getLogger('dummy')
        logger.addHandler(logging.NullHandler())
    
    # 调整批大小和梯度累积以保持全局批大小不变
    batch_size = per_gpu_batch_size  # 直接使用每个GPU的批大小
    if master_process:
        logger.info(f"每个GPU批大小: {batch_size}, 全局批大小: {global_batch_size}")
    
    # 计算每次迭代处理的token数量
    tokens_per_iter = global_batch_size * max_seq_len
    if master_process:
        logger.info(f"每次迭代处理的token数量: {tokens_per_iter:,}")
        logger.info(f"分解为: {ddp_world_size}进程 * {batch_size}批大小 * {max_seq_len}最大序列长度")

    # 设置随机种子和CUDA配置
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # 允许在矩阵乘法中使用tf32
    torch.backends.cudnn.allow_tf32 = True  # 允许在cudnn中使用tf32
    
    # 设备类型和数据类型设置
    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    
    # 初始化最佳验证损失
    best_val_loss = 1e9
    
    # 加载数据集
    train_ds = SFTBinDataset("data/sft_mini_512.npz")
    
    # 创建数据分发采样器
    if ddp:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=ddp_world_size,
            rank=ddp_rank,
            shuffle=True,
            seed=1337
        )
    else:
        train_sampler = None
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        pin_memory=True,  # 使用固定内存加速数据传输到GPU
        drop_last=False,
        shuffle=(train_sampler is None),  # 如果使用sampler就不shuffle
        num_workers=num_workers,  # 使用多个工作进程
        sampler=train_sampler,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    # 可选：创建验证加载器
    # val_ds = SFTBinDataset("data/val_data.bin") 
    # val_loader = torch.utils.data.DataLoader(
    #     val_ds,
    #     batch_size=batch_size,
    #     pin_memory=True,
    #     shuffle=False,
    #     num_workers=num_workers,
    #     prefetch_factor=prefetch_factor if num_workers > 0 else None,
    # )

    # 初始化模型
    model = init_model()
    
    # 加载预训练权重
    if master_process:
        logger.info("加载预训练权重...")
    model = load_pretrained_weights(model, './out/cauchy_pretrain_v21/epoch_19.pth')
    
    # 将模型移到GPU
    model = model.to(device)
    
    # 打印GPU内存使用情况
    print_gpu_memory_stats()
    
    # 初始化梯度缩放器(如果dtype='float16'则启用)
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    
    # 初始化优化器
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    
    # 计算每个epoch的迭代次数
    iter_per_epoch = len(train_loader)
    
    # 编译模型(可选)
    if compile:
        if master_process:
            logger.info("编译模型中... (大约需要一分钟)")
        unoptimized_model = model
        model = torch.compile(model)  # 需要PyTorch 2.0
    
    # 将模型包装到DDP容器中
    if ddp:
        # 检查是否需要设置bucket_cap_mb
        # 默认25MB可能太小，导致通信开销增加
        # 增加bucket大小可减少通信次数
        bucket_cap_mb = 50  # 增大buckets大小可能提高性能
        
        # 忽略`freqs_cis`缓冲区，因为NCCL不支持`ComplexFloat`
        prefix = "_orig_mod." if compile else ""
        model._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cis"}
        model = DDP(
            model, 
            device_ids=[ddp_local_rank],
            bucket_cap_mb=bucket_cap_mb,
            find_unused_parameters=False,  # 提高DDP效率
            gradient_as_bucket_view=True,  # 减少内存使用
        )
    
    # 获取原始模型(如果需要解包DDP容器)
    raw_model = model.module if ddp else model
    
    # 设置CUDA事件记录器，用于性能分析
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    if master_process:
        logger.info("开始训练循环")
        logger.info(f"训练设备: {device}, 数据类型: {dtype}")
        logger.info(f"总GPU数: {ddp_world_size if ddp else 1}, 每个GPU的批大小: {batch_size}")
    
    try:
        # 训练循环
        for epoch in range(max_epoch):
            # 同步所有进程开始时间
            if ddp:
                torch.distributed.barrier()
                
            start_event.record()
            train_loss = train_epoch(epoch, train_loader, train_sampler)
            end_event.record()
            
            torch.cuda.synchronize()
            epoch_time = start_event.elapsed_time(end_event) / 1000.0  # 转换为秒
            
            if master_process:
                logger.info(f'Epoch {epoch+1}/{max_epoch} 完成. 平均训练损失: {train_loss:.4f}, 耗时: {epoch_time:.2f}秒')
                print_gpu_memory_stats()
            
            # 验证（如果有验证集）
            # if (epoch + 1) % eval_interval == 0:
            #     val_loss = valid_epoch(epoch, val_loader)
            
            # 保存检查点
            if master_process:
                save_checkpoint(
                    raw_model,
                    optimizer,
                    epoch,
                    os.path.join(save_dir, f'epoch_{epoch}.pth')
                )
                
    except Exception as e:
        if master_process:
            logger.error(f"训练过程中发生错误: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    finally:
        # 清理分布式训练环境
        if ddp:
            destroy_process_group()
            
        if master_process:
            logger.info("训练完成")