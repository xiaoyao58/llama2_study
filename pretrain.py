import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,6,7'
import time
import math
import pickle
from contextlib import nullcontext
import numpy as np
import torch
from model import Transformer, ModelArgs  # 导入前面定义的Transformer模型和参数类
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from dataset import PretrainDataset  # 导入数据集类
from train_config import pretrain_config
import logging
from tqdm import tqdm
import wandb

# 打印GPU信息
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of CUDA devices: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")

# 分布式训练启动命令示例
# torchrun --standalone --nproc_per_node=4 pretrain.py 或 python -m torch.distributed.launch --nproc_per_node=4 pretrain.py
        
def get_logger(filename, verbosity=1, name=None):
    """
    创建日志记录器
    
    Args:
        filename: 日志文件名
        verbosity: 日志级别（0=DEBUG, 1=INFO, 2=WARNING）
        name: 记录器名称
        
    Returns:
        配置好的日志记录器
    """
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    # 添加文件处理器
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # 添加控制台处理器
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

# -----------------------------------------------------------------------------
def get_lr(it):
    """
    根据当前迭代步数计算学习率（学习率调度）
    
    Args:
        it: 当前迭代步数
    
    Returns:
        当前学习率
    """
    # 1) 线性预热阶段
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) 如果超过衰减步数，返回最小学习率
    if it > lr_decay_iters:
        return min_lr
    # 3) 中间阶段，使用余弦衰减
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 系数范围从0到1
    return min_lr + coeff * (learning_rate - min_lr)

def train_epoch(epoch):
    """
    训练一个完整的epoch
    
    Args:
        epoch: 当前epoch索引
    """
    start_time = time.time()
    train_loader.sampler.set_epoch(epoch)  # 重要：确保分布式训练中数据正确打乱
    
    # 仅在主进程创建进度条
    if master_process:
        pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{max_epoch}", 
                   ncols=100, leave=True, position=0)
        
    running_loss = 0.0
    for step, (X, Y) in enumerate(train_loader):
        X = X.to(device)  # 输入序列
        Y = Y.to(device)  # 目标序列
        
        # 获取当前学习率（如果启用衰减）
        lr = get_lr(epoch*iter_per_epoch+step) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        if ddp:
            # 在DDP训练中，只有在最后一个微批次中同步梯度
            model.require_backward_grad_sync = 0 == gradient_accumulation_steps - 1
            
        with ctx:  # 自动混合精度上下文（如果启用）
            logits = model(X, Y)  # 前向传播
            loss = raw_model.last_loss  # 获取上一次前向传播的损失
            loss = loss / gradient_accumulation_steps  # 梯度累积时缩放损失
            
        # 带梯度缩放的反向传播（如果使用fp16）
        scaler.scale(loss).backward()
        
        # 梯度累积完成后更新参数
        if (step + 1) % gradient_accumulation_steps == 0:
            # 梯度裁剪
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)  # 在裁剪前取消缩放
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
            # 更新优化器和缩放器
            scaler.step(optimizer)
            scaler.update()
            
            # 清零梯度
            optimizer.zero_grad(set_to_none=True)
        
        # 更新主进程的进度条
        if master_process:
            running_loss += loss.item() * gradient_accumulation_steps
            avg_loss = running_loss / (step + 1)
            
            # 用当前统计数据更新进度条
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}', 
                'lr': f'{optimizer.param_groups[-1]["lr"]:.7f}'
            })
            pbar.update(1)

            # 记录到wandb
            if use_wandb and (step + 1) % log_interval == 0:
                wandb.log({
                    "train/loss": avg_loss,
                    "train/learning_rate": optimizer.param_groups[-1]["lr"],
                    "train/epoch": epoch + (step + 1) / len(train_loader),
                    "train/step": step + epoch * len(train_loader)
                })
        
        # 定期保存模型
        if step % save_interval == 0 and step > 0:
            if ddp:
                # 在DDP中，只有主进程保存模型
                if torch.distributed.get_rank() == 0:
                    model.eval()
                    torch.save(model.module.state_dict(),'{}/iter_{}.pth'.format(save_dir, int(step+epoch*iter_per_epoch)))
                    model.train()
            else:
                model.eval()
                torch.save(model.state_dict(),'{}/iter_{}.pth'.format(save_dir, int(step+epoch*iter_per_epoch)))
                model.train()
    
    # 关闭进度条并记录摘要
    if master_process:
        total_time = time.time() - start_time
        avg_loss = running_loss / len(train_loader)
        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'lr': f'{optimizer.param_groups[-1]["lr"]:.7f}',
            'time': f'{total_time:.1f}s'
        })
        pbar.close()
        
        # 记录到日志
        logger.info(f'Epoch {epoch+1}/{max_epoch} completed. Avg loss: {avg_loss:.4f}, Time: {total_time:.1f}s')
        # 记录epoch摘要到wandb
        if use_wandb:
            wandb.log({
                "train/epoch_loss": avg_loss,
                "train/epoch": epoch + 1,
                "train/epoch_time": total_time
            })


def init_model():
    """
    初始化或加载模型
    
    Returns:
        初始化好的模型
    """
    # 模型初始化
    model_args = dict(
        dim=dim,  # 隐藏维度
        n_layers=n_layers,  # 层数
        n_heads=n_heads,  # 注意力头数
        n_kv_heads=n_heads,  # 键值头数（这里等于注意力头数）
        vocab_size=64793,  # 词表大小
        multiple_of=multiple_of,  # SwiGLU隐藏层的倍数
        max_seq_len=max_seq_len,  # 最大序列长度
        dropout=dropout,  # dropout比例
    )  # 从命令行开始的model_args
    
    if init_from == "scratch":
        # 从头初始化新模型
        print("Initializing a new model from scratch")
        gptconf = ModelArgs(**model_args)
        model = Transformer(gptconf)
    elif init_from == "resume":
        # 从检查点恢复训练
        print(f"Resuming training from {out_dir}")
        ckpt_path = os.path.join(out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint["model_args"]
        
        # 强制这些配置属性相等，否则无法恢复训练
        # 其他属性（如dropout）可以按需从命令行设置
        for k in ["dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "multiple_of", "max_seq_len"]:
            model_args[k] = checkpoint_model_args[k]
            
        # 创建模型
        gptconf = ModelArgs(**model_args)
        model = Transformer(gptconf)
        
        # 加载状态字典
        state_dict = checkpoint["model"]
        # 修复状态字典的键（解决前缀问题）
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
        
    return model

# I/O
if __name__=="__main__":
    # 配置参数
    out_dir = pretrain_config['out_dir']  # 输出目录
    max_epoch = pretrain_config['max_epoch']  # 最大训练轮次
    eval_interval = pretrain_config['eval_interval']  # 评估间隔
    log_interval = pretrain_config['log_interval']  # 日志记录间隔
    save_interval = pretrain_config['save_interval']  # 保存间隔
    eval_iters = pretrain_config['eval_iters']  # 评估迭代次数
    eval_only = pretrain_config['eval_only']  # 如果为True，脚本在第一次评估后就退出
    always_save_checkpoint = pretrain_config['always_save_checkpoint']  # 如果为True，每次评估后都保存检查点
    init_from = pretrain_config['init_from']  # 初始化模式：'scratch'（从头开始）, 'resume'（恢复训练）或 'gpt2*'

    # 梯度累积相关设置
    gradient_accumulation_steps = pretrain_config['gradient_accumulation_steps']  # 用于模拟更大的批量大小
    batch_size = pretrain_config['batch_size']  # 如果gradient_accumulation_steps > 1，这是微批量大小

    # 模型参数 - 根据需要更改
    max_seq_len = pretrain_config['max_seq_len']  # 最大序列长度
    dim = pretrain_config['dim']  # 模型维度
    n_layers = pretrain_config['n_layers']  # 层数
    n_heads = pretrain_config['n_heads']  # 注意力头数
    multiple_of = pretrain_config['multiple_of']  # 隐藏层大小的倍数
    dropout = pretrain_config['dropout']  # 预训练时0较好，微调时尝试0.1+
    bias = pretrain_config['bias']  # 在LayerNorm和Linear层中是否使用偏置

    # AdamW优化器设置
    learning_rate = pretrain_config['learning_rate']  # 最大学习率
    weight_decay = pretrain_config['weight_decay']  # 权重衰减率
    beta1 = pretrain_config['beta1']  # Adam的beta1参数
    beta2 = pretrain_config['beta2']  # Adam的beta2参数
    grad_clip = pretrain_config['grad_clip']  # 梯度裁剪值，如果==0.0则禁用

    # 学习率衰减设置
    decay_lr = pretrain_config['decay_lr']  # 是否使用学习率衰减
    warmup_iters = pretrain_config['warmup_iters']  # 预热步数
    lr_decay_iters = pretrain_config['lr_decay_iters']  # 衰减总步数，根据Chinchilla缩放规则应接近最大迭代次数
    min_lr = pretrain_config['min_lr']  # 最小学习率，根据Chinchilla规则应约为learning_rate/10

    # DDP（分布式数据并行）设置
    backend = pretrain_config['backend']  # 分布式后端：'nccl', 'gloo'等

    # 系统设置
    device = pretrain_config['device']  # 设备：'cpu', 'cuda', 'cuda:0', 'cuda:1'等，或在Mac上尝试'mps'
    dtype = pretrain_config['dtype']  # 数据类型：'float32', 'bfloat16'或'float16'，后者会自动实现GradScaler
    compile = pretrain_config['compile']  # 是否使用PyTorch 2.0编译模型以加速

     # 从配置中获取wandb参数
    use_wandb = pretrain_config.get('use_wandb', False)
    wandb_project = pretrain_config.get('wandb_project', 'llm-training')
    wandb_entity = pretrain_config.get('wandb_entity', None)
    wandb_run_name = pretrain_config.get('wandb_run_name', 'llm-run')
    # -----------------------------------------------------------------------------
    # 收集配置键
    config_keys = [
        k
        for k, v in globals().items()
        if not k.startswith("_") and isinstance(v, (int, float, bool, str))
    ]

    #登陆wandb
    wandb.login(key="99216b7efd01227fa458aeff0fddfaeed59f32c7")

    # 初始化wandb
    if use_wandb:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            config={k: globals()[k] for k in config_keys}
        )

    # 创建保存目录
    # save_dir = os.path.join(out_dir, 'pretrain')
    save_dir = out_dir
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir, exist_ok=True)
    logger = get_logger(os.path.join(save_dir,'log.log'))  # 初始化日志记录器
    
    # 各种初始化、派生属性、I/O设置
    ddp = int(os.environ.get("RANK", -1)) != -1  # 是否为DDP运行？
    
    if ddp:
        try:
            # 检查操作系统是否为Windows并选择适当的后端
            backend = "gloo" if os.name == 'nt' else "nccl"  # Windows上使用gloo，其他使用nccl
            init_process_group(backend=backend)  # 初始化进程组
            ddp_rank = int(os.environ["RANK"])  # 全局进程排名
            ddp_local_rank = int(os.environ["LOCAL_RANK"])  # 本地进程排名
            ddp_world_size = int(os.environ["WORLD_SIZE"])  # 全球进程总数
            device = f"cuda:{ddp_local_rank}"  # 设置设备为本地GPU
            torch.cuda.set_device(device)  # 设置CUDA设备
            master_process = ddp_rank == 0  # 此进程将执行记录、检查点等
            seed_offset = ddp_rank  # 每个进程获得不同的种子
            print(f"DDP initialized. Rank: {ddp_rank}, Local Rank: {ddp_local_rank}, World Size: {ddp_world_size}")
        except Exception as e:
            print(f"DDP initialization failed: {e}")
            # 回退到单GPU
            ddp = False
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            master_process = True
            seed_offset = 0
            ddp_world_size = 1
            print(f"Falling back to single device mode: {device}")
    else:
        # 如果不是ddp，我们在单GPU上运行，并且一个进程
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
    # 计算每次迭代处理的token数
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * max_seq_len
    if master_process:
        print(f"tokens per iteration will be: {tokens_per_iter:,}")
        print(f"breaks down as: {gradient_accumulation_steps} grad accum steps * {ddp_world_size} processes * {batch_size} batch size * {max_seq_len} max seq len")
        print(f"Using device: {device}")

    # 创建输出目录和设置随机种子
    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    
    # 允许TensorFloat32精度优化
    torch.backends.cuda.matmul.allow_tf32 = True  # 允许在矩阵乘法中使用tf32
    torch.backends.cudnn.allow_tf32 = True  # 允许在cudnn中使用tf32
    
    # 定义设备类型和数据类型
    device_type = "cuda" if "cuda" in device else "cpu"  # 用于后续torch.autocast
    # 注意：float16数据类型将自动使用GradScaler
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    
    # 定义自动混合精度上下文
    ctx = (
        nullcontext()  # CPU不需要自动混合精度
        if device_type == "cpu"
        else torch.cuda.amp.autocast()  # GPU启用自动混合精度
    )
    
    # 初始化最佳验证损失
    best_val_loss = 1e9
    
    # 初始化数据加载器
    data_path_list=[
        './data/pretrain_data.bin'  # 预训练数据路径
        # 以下是其他可选的数据源
        #'./data/baidubaike_563w.bin',
        #'./data/medical_book.bin',
        # './data/medical_encyclopedia.bin',
        # './data/medical_qa.bin',
        # './data/wiki.bin'
    ]
    
    # 创建数据集
    train_ds = PretrainDataset(data_path_list, max_length=max_seq_len, memmap=True)
    
    # 为分布式训练配置采样器
    if ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_ds)
    
    # 简化的DataLoader，初始设置更少的工作线程进行调试
    num_workers = 0  # 开始时设为0，等一切正常工作后可以增加
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        pin_memory=False,  # 不使用pin_memory
        drop_last=False,   # 保留最后不完整的批次
        shuffle=False,     # 有采样器时不需要shuffle
        num_workers=num_workers,  # 工作线程数
        sampler=train_sampler      # 使用前面定义的采样器
    )
    
    # 初始化模型
    model = init_model()
    model.to(device)  # 将模型移至设备
    
    # 初始化梯度缩放器。如果enabled=False，则缩放器不执行任何操作
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    
    # 优化器
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    
    # 编译模型（如果enabled）
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)  # 需要PyTorch 2.0
    
    # 将模型包装到DDP容器中
    if ddp:
        # 忽略`freqs_cis`缓冲区，使DDP在构建时不广播它，因为NCCL不支持`ComplexFloat`
        prefix = "_orig_mod." if compile else ""
        model._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cis"}
        model = DDP(model, device_ids=[ddp_local_rank])
        
    # 如果需要，解包DDP容器
    raw_model = model.module if ddp else model
    
    # 训练循环
    iter_per_epoch = len(train_loader)  # 每个epoch的迭代次数
    
    try:
        for epoch in range(max_epoch):
            if master_process:
                print(f"\nStarting epoch {epoch+1}/{max_epoch}")
            train_epoch(epoch)  # 训练一个epoch
            
            # 保存epoch结束检查点
            if master_process:
                torch.save(raw_model.state_dict(),'{}/epoch_{}.pth'.format(save_dir, epoch))
                print(f"Saved checkpoint for epoch {epoch+1}")
                
    except Exception as e:
        # 捕获训练错误
        print(f"Training error: {e}")
        # 崩溃时保存
        if master_process and 'raw_model' in locals():
            torch.save(raw_model.state_dict(),'{}/crash_recovery.pth'.format(save_dir))
            print("Saved crash recovery checkpoint")
    
    finally:
        # 清理DDP资源
        if ddp:
            destroy_process_group()
            print("DDP resources cleaned up")
            
        # 完成wandb运行
        if master_process and use_wandb:
            wandb.finish()
        
        if master_process:
            print("Training completed")