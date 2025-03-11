pretrain_config = {
    "out_dir" : 'out',  # 输出目录
    "max_epoch" : 10,  # 最大训练轮次
    "eval_interval" : 1,  # 评估间隔
    "log_interval" : 100,  # 日志记录间隔
    "save_interval" : 10000,  # 保存间隔
    "eval_iters" : 200,  # 评估迭代次数
    "eval_only" : False,  # 如果为True，脚本在第一次评估后就退出
    "always_save_checkpoint" : True,  # 如果为True，每次评估后都保存检查点
    "init_from" : 'scratch',  # 初始化模式：'scratch'（从头开始）, 'resume'（恢复训练）或 'gpt2*'
    
    # 梯度累积相关设置
    "gradient_accumulation_steps" : 1,  # 用于模拟更大的批量大小
    "batch_size" : 32,  # 如果gradient_accumulation_steps > 1，这是微批量大小
    
    # 模型参数 - 根据需要更改
    "max_seq_len" : 512,  # 最大序列长度
    "dim" : 512,  # 模型维度
    "n_layers" : 8,  # 层数
    "n_heads" : 8,  # 注意力头数
    "multiple_of" : 32,  # 隐藏层大小的倍数
    "dropout" : 0.0,  # 预训练时0较好，微调时尝试0.1+
    "bias" : False,  # 在LayerNorm和Linear层中是否使用偏置
    
    # AdamW优化器设置
    "learning_rate" : 3e-4,  # 最大学习率
    "weight_decay" : 1e-1,  # 权重衰减率
    "beta1" : 0.9,  # Adam的beta1参数
    "beta2" : 0.95,  # Adam的beta2参数
    "grad_clip" : 1.0,  # 梯度裁剪值，如果::0.0则禁用
    
    # 学习率衰减设置
    "decay_lr" : True,  # 是否使用学习率衰减
    "warmup_iters" : 1000,  # 预热步数
    "lr_decay_iters" : 80000,  # 衰减总步数，根据Chinchilla缩放规则应接近最大迭代次数
    "min_lr" : 1e-5,  # 最小学习率，根据Chinchilla规则应约为learning_rate/10
    
    # DDP（分布式数据并行）设置
    "backend" : 'nccl',  # 分布式后端：'nccl', 'gloo'等
    
    # 系统设置
    "device" : 'cuda',  # 设备：'cpu', 'cuda', 'cuda:0', 'cuda:1'等，或在Mac上尝试'mps'
    "dtype" : 'float16',  # 数据类型：'float32', 'bfloat16'或'float16'，后者会自动实现GradScaler
    "compile" : False,  # 是否使用PyTorch 2.0编译模型以加速

    #wandb
    "use_wandb": True,  # 是否使用wandb
    "wandb_project": "llama2_pretrain",  # wandb项目名
    "wandb_entity": None,  # wandb实体（用户名或团队名）
    "wandb_run_name": "'llama2_pretrain_run",  # wandb运行名
}