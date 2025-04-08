import math
import struct
import inspect
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from activation import *

@dataclass
class ModelArgs:
    """模型配置参数类"""
    dim: int = 4096                     # 模型的隐藏层维度
    n_layers: int = 32                  # Transformer层数
    n_heads: int = 32                   # 注意力头数
    n_kv_heads: Optional[int] = None    # KV注意力头数（用于分组查询注意力机制，若为None则等于n_heads）
    vocab_size: int = -1                # 词表大小（由分词器稍后定义）
    multiple_of: int = 256              # SwiGLU隐藏层大小的倍数，保证是2的幂次方的倍数
    norm_eps: float = 1e-5              # LayerNorm的epsilon值
    max_seq_len: int = 2048             # 最大序列长度
    dropout: float = 0.0                # Dropout比例

class RMSNorm(torch.nn.Module):
    """均方根层归一化（Root Mean Square Layer Normalization）"""
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps                         # 避免除零的小常数
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习的缩放参数

    def _norm(self, x):
        """计算RMS归一化"""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """前向传播"""
        output = self._norm(x.float()).type_as(x)  # 先转为float计算，再转回原类型
        return output * self.weight  # 应用可学习的缩放参数

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    预计算RoPE（旋转位置编码）的频率
    
    Args:
        dim: 头维度
        end: 序列最大长度
        theta: 频率基数
    
    Returns:
        freqs_cos: 余弦值
        freqs_sin: 正弦值
    """
    # 计算不同维度的频率
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # 位置索引
    freqs = torch.outer(t, freqs).float()  # 计算外积，得到(seq_len, dim/2)的张量
    freqs_cos = torch.cos(freqs)  # 余弦部分（实部）
    freqs_sin = torch.sin(freqs)  # 正弦部分（虚部）
    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    重塑频率张量，用于广播机制
    
    Args:
        freqs_cis: 频率张量
        x: 输入张量
    
    Returns:
        重塑后的频率张量，便于与输入张量进行广播
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    应用旋转位置编码（Rotary Position Embeddings，RoPE）到查询和键向量
    
    Args:
        xq: 查询向量
        xk: 键向量
        freqs_cos: 预计算的余弦频率
        freqs_sin: 预计算的正弦频率
    
    Returns:
        应用RoPE后的查询和键向量
    """
    # 将查询和键分解为实部和虚部（将最后一维每两个值分为一对）
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # 重塑频率张量以便于广播
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # 使用复数旋转公式应用旋转
    # (a+bi)(cos θ+i sin θ) = (a cos θ - b sin θ) + (a sin θ + b cos θ)i
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin  # 实部
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos  # 虚部
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin  # 实部
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos  # 虚部

    # 将实部和虚部重新组合并扁平化最后两维
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    # 转回原来的数据类型
    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    在GQA（分组查询注意力）中重复键值头，实现类似于torch.repeat_interleave
    
    Args:
        x: 输入张量 [batch, seq_len, n_kv_heads, head_dim]
        n_rep: 重复次数
    
    Returns:
        重复后的张量 [batch, seq_len, n_kv_heads*n_rep, head_dim]
    """
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:  # 如果不需要重复，直接返回
        return x
    # 在第四维添加一个维度，扩展，再重塑
    return (
        x[:, :, :, None, :]  # [bs, slen, n_kv_heads, 1, head_dim]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)  # [bs, slen, n_kv_heads, n_rep, head_dim]
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)  # [bs, slen, n_kv_heads*n_rep, head_dim]
    )


class Attention(nn.Module):
    """
    多头注意力机制，支持分组查询注意力（GQA）
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads  # 键值头数
        model_parallel_size = 1  # 模型并行大小（此实现中为1）
        self.n_local_heads = args.n_heads // model_parallel_size  # 本地头数
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size  # 本地键值头数
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 每个键值头对应的查询头数
        self.head_dim = args.dim // args.n_heads  # 每个头的维度
        
        # 线性投影层
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)  # 查询投影
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)  # 键投影
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)  # 值投影
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)  # 输出投影
        
        # Dropout层
        self.attn_dropout = nn.Dropout(args.dropout)  # 注意力权重的dropout
        self.resid_dropout = nn.Dropout(args.dropout)  # 残差连接的dropout
        self.dropout = args.dropout

        # 检查是否可以使用FlashAttention（PyTorch>=2.0提供的优化实现）
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # 如果不能使用FlashAttention，创建掩码用于因果注意力
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)  # 上三角部分为-inf，形成因果掩码
            self.register_buffer("mask", mask)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ):
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, dim]
            freqs_cos: 预计算的余弦频率
            freqs_sin: 预计算的正弦频率
            
        Returns:
            注意力机制的输出 [batch_size, seq_len, dim]
        """
        bsz, seqlen, _ = x.shape

        # QKV投影
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # 重塑为多头形状
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # 应用旋转位置编码
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # 分组查询注意力：扩展键和值
        xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        # 将头维度调整为批次维度，便于矩阵乘法
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 使用FlashAttention实现（如可用）
        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(
                xq, xk, xv, 
                attn_mask=None,  # 使用is_causal替代显式掩码
                dropout_p=self.dropout if self.training else 0.0, 
                is_causal=True  # 启用因果注意力
            )
        else:
            # 手动实现注意力机制
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)  # 缩放点积注意力
            assert hasattr(self, 'mask')
            # 应用因果掩码
            scores = scores + self.mask[:, :, :seqlen, :seqlen]   # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)  # softmax归一化
            scores = self.attn_dropout(scores)  # 应用dropout
            output = torch.matmul(scores, xv)  # 计算加权和 (bs, n_local_heads, seqlen, head_dim)

        # 转置并连接所有头的输出
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # 最终线性投影回原始维度空间
        output = self.wo(output)
        output = self.resid_dropout(output)  # 应用dropout
        return output

class FeedForward(nn.Module):
    """
    前馈神经网络，使用SwiGLU激活函数
    """
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        # 调整隐藏层维度，保证是multiple_of的倍数
        hidden_dim = int(2 * hidden_dim / 3)  # SwiGLU减少隐藏维度的2/3
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)  # 向上取整为multiple_of的倍数
        
        # 三个线性变换层
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # 第一个投影
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # 输出投影
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # 用于门控机制的投影
        self.activation= CauchyActivationV8(neurons=hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """SwiGLU激活：swish(w1(x))*w3(x)"""
        # return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
        return self.dropout(self.w2(self.activation(self.w1(x)) * self.w3(x)))

class TransformerBlock(nn.Module):
    """
    Transformer块，结合了注意力机制和前馈网络
    """
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,  # 隐藏层维度是输入维度的4倍
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        self.layer_id = layer_id
        # 层归一化
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)  # 注意力前的归一化
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)  # 前馈网络前的归一化

    def forward(self, x, freqs_cos, freqs_sin):
        """
        前向传播，应用预归一化的残差连接
        """
        # 注意力层
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
        # 前馈层
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    """
    完整的Transformer模型
    """
    last_loss: Optional[torch.Tensor]  # 记录最后一次损失值

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        # 词嵌入
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        
        # Transformer层
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
            
        # 输出层
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)  # 最终的层归一化
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)  # 输出投影

        # 共享词嵌入和输出层的权重（权重绑定）
        self.tok_embeddings.weight = self.output.weight  # https://paperswithcode.com/method/weight-tying

        # 预计算RoPE的频率，并注册为缓冲区（不是模型参数）
        freqs_cos, freqs_sin = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # 初始化模型权重
        self.apply(self._init_weights)
        # 对残差投影特殊缩放初始化，基于GPT-2论文
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * params.n_layers))

        # 初始化最后一次前向传播的损失属性
        self.last_loss = None

    def _init_weights(self, module):
        """
        初始化模型权重
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            tokens: 输入token序列 [batch_size, seq_len]
            targets: 目标token序列 [batch_size, seq_len]，用于计算损失，可选
            
        Returns:
            如果训练，返回整个序列的logits；如果推理，只返回最后位置的logits
        """
        _bsz, seqlen = tokens.shape
        
        # 应用词嵌入
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        
        # 获取当前序列长度的位置编码
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        # 顺序通过所有Transformer层
        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
        
        # 最终的归一化
        h = self.norm(h)

        if targets is not None:
            # 如果提供了目标，计算所有位置的logits和损失
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # 推理时优化：只计算最后位置的logits
            logits = self.output(h[:, [-1], :])  # 注意：使用list [-1]保留时间维度
            self.last_loss = None

        return logits

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        配置优化器，区分权重衰减和非权重衰减参数
        
        Args:
            weight_decay: 权重衰减率
            learning_rate: 学习率
            betas: Adam优化器的beta参数
            device_type: 设备类型（cuda或cpu）
            
        Returns:
            配置好的AdamW优化器
        """
        # 获取所有参数
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # 过滤掉不需要梯度的参数
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        # 创建优化器组。所有二维参数应用权重衰减，其他不应用。
        # 即所有矩阵乘法的权重和嵌入使用权重衰减，所有偏置和层归一化不使用。
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        # 打印参数统计
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        # 创建AdamW优化器，如果可用使用融合版本
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """
        估计模型FLOP利用率（MFU），以A100 bfloat16峰值FLOPS为单位
        
        Args:
            fwdbwd_per_iter: 每次迭代的前向和后向传播次数
            dt: 时间间隔
            
        Returns:
            MFU值
        """
        # 首先估计每次迭代的FLOP数
        # 参考PaLM论文附录B: https://arxiv.org/abs/2204.02311
        N = sum(p.numel() for p in self.parameters())  # 参数总数
        cfg = self.params
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.dim//cfg.n_heads, cfg.max_seq_len
        flops_per_token = 6*N + 12*L*H*Q*T  # 每个token的FLOP
        flops_per_fwdbwd = flops_per_token * T  # 每次前向后向传播的FLOP
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter  # 每次迭代的FLOP
        
        # 将FLOP吞吐量表示为A100 bfloat16峰值FLOP的比率
        flops_achieved = flops_per_iter * (1.0/dt)  # 每秒FLOP
        flops_promised = 312e12  # A100 GPU bfloat16峰值FLOP是312 TFLOPS
        mfu = flops_achieved / flops_promised  # 模型FLOP利用率
        return mfu

    #@torch.inference_mode()
    @torch.no_grad()
    def generate(self, idx, eos, max_new_tokens, temperature=1.0, top_k=None):
        """
        自回归生成文本
        
        Args:
            idx: 条件序列索引张量 [batch_size, seq_len]
            eos: 结束符token
            max_new_tokens: 最多生成的新token数
            temperature: 采样温度，控制分布平滑度（0表示贪婪，>0表示采样）
            top_k: 只从top_k个概率最高的token中采样（可选）
            
        Returns:
            生成的token序列
        """
        for _ in range(max_new_tokens):
            # 如果序列上下文过长，裁剪到max_seq_len
            idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
            
            # 向前传播模型获取序列中的logits
            logits = self(idx_cond)
            logits = logits[:, -1, :]  # 只取最后一个时间步
            
            if temperature == 0.0:
                # 贪婪解码：选择概率最高的单个索引
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # 按温度缩放logits
                logits = logits / temperature
                
                # 可选地将logits裁剪到仅保留top_k个选项
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                # 应用softmax转换logits为（归一化的）概率
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)  # 多项分布采样
            
            # 将采样的索引附加到序列并继续
            idx = torch.cat((idx, idx_next), dim=1)
            
            # 如果生成了结束符，提前结束
            if idx_next==eos:
                break

        return idx

    def export(self, filepath='model.bin'):
        """
        将模型权重以fp32格式导出到.bin文件，供C语言读取
        
        Args:
            filepath: 导出文件路径
        """
        f = open(filepath, 'wb')

        def serialize(t):
            """将张量序列化为二进制格式"""
            d = t.detach().cpu().view(-1).numpy().astype(np.float32)
            b = struct.pack(f'{len(d)}f', *d)
            f.write(b)

        # 首先写入头部信息
        hidden_dim = self.layers[0].feed_forward.w1.weight.shape[0]
        p = self.params
        n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
        header = struct.pack('iiiiiii', p.dim, hidden_dim, p.n_layers, p.n_heads,
                                       n_kv_heads, p.vocab_size, p.max_seq_len)
        f.write(header)

        # 写入嵌入权重
        serialize(self.tok_embeddings.weight)

        # 写入所有Transformer层的权重
        # 注意力权重
        for layer in self.layers:
            serialize(layer.attention_norm.weight)
        for layer in self.layers:
            serialize(layer.attention.wq.weight)
        for layer in self.layers:
            serialize(layer.attention.wk.weight)
        for layer in self.layers:
            serialize(layer.attention.wv.weight)
        for layer in self.layers:
            serialize(layer.attention.wo.weight)
            
        # 前馈网络权重
        for layer in self.layers:
            serialize(layer.ffn_norm.weight)
        for layer in self.layers:
            serialize(layer.feed_forward.w1.weight)
        for layer in self.layers:
            serialize(layer.feed_forward.w2.weight)
        for layer in self.layers:
            serialize(layer.feed_forward.w3.weight)
            
        # 最终的层归一化权重
        serialize(self.norm.weight)
        # 注意：因为权重共享所以不需要写入最终分类器权重
        
        # 写入位置编码的频率
        serialize(self.freqs_cos[:p.max_seq_len])
        serialize(self.freqs_sin[:p.max_seq_len])

        # 关闭文件
        f.close()
        print(f"wrote {filepath}")