import torch
from torch import nn
import math

class CauchyActivation(nn.Module):
    def __init__(self, neurons=768):
        super(CauchyActivation, self).__init__()
        self.lambda_1 = nn.Parameter(0.01 * torch.ones(neurons))
        self.lambda_2 = nn.Parameter(1 * torch.ones(neurons))
        self.d = nn.Parameter(torch.ones(neurons))
        self.neurons = neurons

    def forward(self, x):
        # x.size() = (..., neurons)
        denomintor = x**2 + self.d ** 2     # size = (..., neurons)
        term_1 = 10*self.lambda_1 * x  / denomintor
        term_2 = 10*self.lambda_2 / denomintor
        return (term_1 + term_2)


class CauchyActivation2(nn.Module):
    """增强型Cauchy激活函数，专注于加速训练和梯度流动"""
    def __init__(self, dim: int, train_param: bool = True):
        super(CauchyActivation2, self).__init__()
        # 使用更大的初始值
        self.lambda1 = nn.Parameter(torch.ones(1) * 4.0, requires_grad=train_param)
        self.lambda2 = nn.Parameter(torch.ones(1) * 2.0, requires_grad=train_param)
        self.d = nn.Parameter(torch.ones(1) * 0.1, requires_grad=train_param)  # 更小的d值使曲线更陡峭
        
        # 增加初始缩放因子
        self.scale = nn.Parameter(torch.ones(1) * 5.0, requires_grad=train_param)
        
        # 添加截断参数以避免饱和
        self.clip_min = nn.Parameter(torch.ones(1) * -0.2, requires_grad=False)
        self.clip_max = nn.Parameter(torch.ones(1) * 0.2, requires_grad=False)

    def forward(self, x):
        # 首先裁剪输入范围以避免极端值
        x = torch.clamp(x, min=-20.0, max=20.0)
        
        lambda1 = torch.abs(self.lambda1) + 1e-6
        lambda2 = torch.abs(self.lambda2) + 1e-6
        d = torch.abs(self.d) + 1e-6
        scale = torch.abs(self.scale) + 1e-6

        # 使用平方项而不是完全二次项，提供更温和的非线性
        denominator = 1.0 + (x**2) / d
        
        # 组合两个项
        result = (lambda1 * x / denominator + lambda2 / denominator) * scale
        
        # 添加残差连接，让一些线性信号通过
        result = result + 0.1 * x
        
        # 裁剪输出范围
        return torch.clamp(result, min=self.clip_min, max=self.clip_max)

class CauchyActivationV3(nn.Module):
    def __init__(self, neurons=768, gamma_init=1.0, beta=0.1, alpha=10.0):
        super().__init__()
        # 可学习参数（带约束初始化）
        self.gamma = nn.Parameter(torch.full((neurons,), gamma_init))  # 正半轴尺度参数
        self.beta = nn.Parameter(torch.full((neurons,), beta))        # 负半轴泄漏斜率
        self.alpha = alpha                                            # 平滑过渡系数
        
        # 原参数重命名并约束（避免分母接近零）
        self.lambda_1 = nn.Parameter(0.01 * torch.ones(neurons))      # 线性项系数
        self.lambda_2 = nn.Parameter(1.0 * torch.ones(neurons))        # 非线性项系数
        self.d = nn.Parameter(torch.ones(neurons))                    # 分母偏移量
        
        # 确保初始值合理
        with torch.no_grad():
            self.d.clamp_(min=0.1)
            self.beta.clamp_(min=0.01, max=0.5)

    def forward(self, x):
        # 参数约束（训练时保持合理性）
        gamma = self.gamma.abs() + 0.1      # |gamma| >= 0.1
        d = self.d.abs() + 1e-5             # 避免分母为零
        beta = self.beta.clamp(min=0.01)     # 0.01 <= beta <= 0.5
        
        # 原Cauchy项计算
        denominator = x.pow(2) + d.pow(2)
        term_1 = self.lambda_1 * x / denominator
        term_2 = self.lambda_2 / denominator
        
        # 柯西-SiLU混合项（负半轴平滑过渡）
        cauchy_cdf = 0.5 + (1 / math.pi) * torch.atan(x / gamma)
        neg_part = (beta * x) * cauchy_cdf
        
        # 平滑门控（Sigmoid混合）
        gate = torch.sigmoid(self.alpha * x)
        output = gate * (term_1 + term_2) + (1 - gate) * neg_part
        
        return output
    
class CauchyActivationV4(nn.Module):
    def __init__(self, neurons=768, gamma_init=1.0, beta_init=0.1):
        super().__init__()
        # 可学习参数
        self.gamma = nn.Parameter(torch.full((neurons,), gamma_init))  # 控制正/负半轴尺度
        self.beta = nn.Parameter(torch.full((neurons,), beta_init))    # 负半轴泄漏斜率
        
        # 初始化约束
        with torch.no_grad():
            self.gamma.clamp_(min=0.1)      # 避免gamma过小导致梯度爆炸
            self.beta.clamp_(min=0.01, max=0.5)  # 限制泄漏范围

    def forward(self, x):
        # 确保所有张量使用相同的数据类型
        dtype = x.dtype
        device = x.device
        
        gamma = (self.gamma.abs() + 1e-5).to(dtype)     # 确保gamma > 0，并使用正确的数据类型
        beta = self.beta.clamp(min=0.01, max=0.5).to(dtype)
        
        # 计算正负半轴
        pos_mask = (x >= 0).to(dtype)
        neg_mask = 1 - pos_mask
        
        # 正半轴: 柯西阈值
        pos_part = x * (1 - 1 / (1 + (x / gamma).pow(2)))
        
        # 负半轴: 柯西-SiLU混合
        neg_part = (beta * x) * (0.5 + (1 / math.pi) * torch.atan(x / gamma))
        
        # 合并输出，确保使用正确的数据类型
        output = pos_mask * pos_part + neg_mask * neg_part
        return output.to(dtype)
    
class CauchyActivationV5(nn.Module):
    def __init__(self, neurons=768, a_init=1.0, b_init=1.0):
        super().__init__()
        # 可学习参数
        self.a = nn.Parameter(torch.full((neurons,), a_init))  # 线性缩放因子
        self.b = nn.Parameter(torch.full((neurons,), b_init))  # 分母缩放因子
        
        # 固定参数
        self.transition_scale = 10.0  # tanh的缩放因子，控制过渡的陡峭程度
        
        # 初始化约束
        with torch.no_grad():
            self.b.clamp_(min=0.1)  # 避免分母接近零

    def forward(self, x):
        # 确保参数为正值
        a = self.a.abs()
        b = self.b.abs() + 0.1  # 添加小偏移量避免除零
        
        # 计算正半轴部分
        pos_gate = (torch.tanh(self.transition_scale * x) + 1) / 2
        cauchy_cdf = 0.5 + (1 / math.pi) * torch.atan(x / b)
        pos_part = a * x * cauchy_cdf
        
        # 计算负半轴部分
        neg_gate = (torch.tanh(-self.transition_scale * x) + 1) / 2
        neg_part = (a * x) / (2 + 2 * (x.pow(2) / b))
        
        # 组合两部分
        output = pos_gate * pos_part + neg_gate * neg_part
        
        return output
    
class CauchyActivationV6(nn.Module):
    def __init__(self, neurons=768, a_init=0.5, b_init=1.0):
        super().__init__()
        self.a = nn.Parameter(torch.full((neurons,), a_init))
        self.b = nn.Parameter(torch.full((neurons,), b_init))
        self.transition_scale = nn.Parameter(torch.tensor(10.0))  # 可学习过渡
        
    def forward(self, x):
        a = self.a.abs()
        b = self.b.abs() + 0.01  # 避免除零
        
        # 正半轴（Cauchy CDF）
        pos_gate = torch.sigmoid(self.transition_scale * x)  # 改用 sigmoid
        cauchy_cdf = 0.5 + (1 / math.pi) * torch.atan(x / b)
        pos_part = a * x * cauchy_cdf
        
        # 负半轴（改进形式）
        neg_gate = torch.sigmoid(-self.transition_scale * x)
        neg_part = (a * x) / (1 + torch.abs(x) / b)  # 更平滑的负半轴
        
        # 组合
        output = pos_gate * pos_part + neg_gate * neg_part
        return output

# 增加系数
class CauchyActivationV7(nn.Module):
    def __init__(self, neurons=768):
        super(CauchyActivationV7, self).__init__()
        self.coeff = nn.Parameter(torch.ones(neurons))
        self.lambda_1 = nn.Parameter(0.01 * torch.ones(neurons))
        self.lambda_2 = nn.Parameter(1 * torch.ones(neurons))
        self.d = nn.Parameter(torch.ones(neurons))
        self.neurons = neurons

    def forward(self, x):
        # x.size() = (..., neurons)
        denomintor = x**2 + self.d ** 2     # size = (..., neurons)
        term_1 = self.coeff*self.lambda_1 * x  / denomintor
        term_2 = self.coeff*self.lambda_2 / denomintor
        return (term_1 + term_2)
    
# 冻结llambda_1
class CauchyActivationV8(nn.Module):
    def __init__(self, neurons=768):
        super(CauchyActivationV8, self).__init__()
        self.lambda_1 = nn.Parameter(0.01 * torch.ones(neurons),requires_grad=False)
        self.lambda_2 = nn.Parameter(1 * torch.ones(neurons))
        self.d = nn.Parameter(torch.ones(neurons))
        self.neurons = neurons

    def forward(self, x):
        # x.size() = (..., neurons)
        denomintor = x**2 + self.d ** 2     # size = (..., neurons)
        term_1 = self.lambda_1 * x  / denomintor
        term_2 = self.lambda_2 / denomintor
        return (term_1 + term_2)

# 冻结d
class CauchyActivationV9(nn.Module):
    def __init__(self, neurons=768):
        super(CauchyActivationV9, self).__init__()
        self.lambda_1 = nn.Parameter(0.01 * torch.ones(neurons))
        self.lambda_2 = nn.Parameter(1 * torch.ones(neurons))
        self.d = nn.Parameter(torch.ones(neurons),requires_grad=False)
        self.neurons = neurons

    def forward(self, x):
        # x.size() = (..., neurons)
        denomintor = x**2 + self.d ** 2     # size = (..., neurons)
        term_1 = self.lambda_1 * x  / denomintor
        term_2 = self.lambda_2 / denomintor
        return (term_1 + term_2)
    

class CauchyActivationV10(nn.Module):
    """增强型Cauchy激活函数，专注于加速训练和梯度流动"""
    def __init__(self, neurons: int, train_param: bool = True):
        super(CauchyActivationV10, self).__init__()
        # 使用更大的初始值
        self.coeff = nn.Parameter(torch.ones(neurons))
        self.lambda1 = nn.Parameter(torch.ones(neurons) * 4.0, requires_grad=train_param)
        self.lambda2 = nn.Parameter(torch.ones(neurons) * 2.0, requires_grad=train_param)
        self.d = nn.Parameter(torch.ones(neurons) * 0.1, requires_grad=train_param)  # 更小的d值使曲线更陡峭
        
        # 增加初始缩放因子
        self.scale = nn.Parameter(torch.ones(neurons) * 5.0, requires_grad=train_param)
        
        # 添加截断参数以避免饱和
        self.clip_min = nn.Parameter(torch.ones(neurons) * -0.2, requires_grad=False)
        self.clip_max = nn.Parameter(torch.ones(neurons) * 0.2, requires_grad=False)

    def forward(self, x):
        # 首先裁剪输入范围以避免极端值
        x = torch.clamp(x, min=-20.0, max=20.0)
        
        lambda1 = torch.abs(self.lambda1) + 1e-6
        lambda2 = torch.abs(self.lambda2) + 1e-6
        d = torch.abs(self.d) + 1e-6
        scale = torch.abs(self.scale) + 1e-6

        # 使用平方项而不是完全二次项，提供更温和的非线性
        denominator = 1.0 + (x**2) / d
        
        # 组合两个项
        result = (self.coeff*lambda1 * x / denominator + self.coeff*lambda2 / denominator) * scale
        
        # 添加残差连接，让一些线性信号通过
        result = result + 0.1 * x
        
        # 裁剪输出范围
        return torch.clamp(result, min=self.clip_min, max=self.clip_max)