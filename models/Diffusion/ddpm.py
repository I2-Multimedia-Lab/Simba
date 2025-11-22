import torch.nn as nn
from functools import partial
import torch
import random
import numpy as np
import sys
import torch.nn.functional as F


def set_seed(seed=None):
    """设置随机种子确保可复现性"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def exists(x):
    """检查变量是否存在"""
    return x is not None


def isfunction(func):
    """检查对象是否可调用"""
    return callable(func)


def default(val, d):
    """返回val或默认值d"""
    if exists(val):
        return val
    return d() if isfunction(d) else d


def noise_like(shape, device, repeat=False):
    """生成指定形状的噪声张量"""
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def extract_into_tensor(schedule_array, timesteps, target_shape):
    """从调度数组中提取时间步对应的值并重塑为目标形状"""
    batch_size, *_ = timesteps.shape
    extracted_values = schedule_array.gather(-1, timesteps)
    return extracted_values.reshape(batch_size, *((1,) * (len(target_shape) - 1)))


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    """创建不同类型的beta调度"""
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )
    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)
    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    
    return betas.numpy() if isinstance(betas, torch.Tensor) else betas


class DDPM(nn.Module):
    """DDPM (Denoising Diffusion Probabilistic Models) 实现"""
    
    def __init__(self,
                 denoise_model,          # 去噪网络模型
                 condition_model,        # 条件处理模型
                 timesteps=1000,         # 扩散步数
                 beta_schedule="linear", # beta调度策略
                 clip_denoised=False,    # 是否裁剪去噪结果
                 linear_start=1e-4,      # 线性调度起始值
                 linear_end=2e-2,        # 线性调度结束值
                 cosine_s=8e-3,          # 余弦调度参数
                 given_betas=None,       # 预定义beta值
                 v_posterior=0.,         # 后验方差调节参数
                 parameterization="eps"  # 参数化方式: "eps"或"x0"
                 ):

        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        
        # 模型配置
        self.parameterization = parameterization
        self.clip_denoised = clip_denoised
        self.denoise_model = denoise_model      # 标准化：去噪网络
        self.condition_model = condition_model  # 标准化：条件模型
        self.v_posterior = v_posterior

        # 注册扩散调度参数
        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                                 linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        """注册扩散过程的调度参数"""
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, 
                                     linear_end=linear_end, cosine_s=cosine_s)
        
        # 计算alpha相关参数
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        # 注册基础参数
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # 扩散过程q(x_t | x_{t-1})相关参数
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # 后验分布q(x_{t-1} | x_t, x_0)相关参数
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(1. - betas) / (1. - alphas_cumprod)))

    def q_mean_variance(self, x_start, timesteps):
        """计算前向过程q(x_t|x_0)的均值和方差"""
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, timesteps, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, timesteps, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, timesteps, predicted_noise):
        """从噪声预测中恢复x_0"""
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, timesteps, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, timesteps, x_t.shape) * predicted_noise
        )

    def q_posterior(self, x_start, x_t, timesteps):
        """计算后验分布q(x_{t-1}|x_t,x_0)的均值和方差"""
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, timesteps, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, timesteps, x_t.shape) * x_t
        )
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, timesteps, x_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x_noisy, timesteps, condition_features, clip_denoised: bool):
        """计算反向过程p_θ(x_{t-1}|x_t)的均值和方差"""
        # 调用去噪模型预测
        model_output = self.denoise_model(x_noisy, timesteps, condition_features)
        
        if self.parameterization == "eps":
            x_reconstructed = self.predict_start_from_noise(x_noisy, timesteps=timesteps, predicted_noise=model_output)
        elif self.parameterization == "x0":
            x_reconstructed = model_output
        
        if clip_denoised:
            x_reconstructed.clamp_(-1., 1.)

        # 计算后验均值和方差
        model_mean, posterior_log_variance = self.q_posterior(x_start=x_reconstructed, x_t=x_noisy, timesteps=timesteps)
        return model_mean, posterior_log_variance, x_reconstructed

    def p_sample(self, x_current, timesteps, condition_features, clip_denoised=True, repeat_noise=False):
        """执行一步反向采样"""
        batch_size, *_, device = *x_current.shape, x_current.device
        
        # 当t=0时不添加噪声
        nonzero_mask = (timesteps != 0).float().view(batch_size, *((1,) * (len(x_current.shape) - 1)))
        
        model_mean, model_log_variance, _ = self.p_mean_variance(x_noisy=x_current, timesteps=timesteps, 
                                                               condition_features=condition_features, 
                                                               clip_denoised=clip_denoised)
        noise = noise_like(x_current.shape, device, repeat_noise)
        
        # 只在t>0时添加噪声
        sampled_x = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return sampled_x

    def q_sample(self, x_start, timesteps, noise=None):
        """前向扩散过程：从x_0采样x_t"""
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape) * noise)

    def p_losses(self, x_start, timesteps, condition_features, noise=None):
        """只关注核心训练目标"""
        noise = default(noise, lambda: torch.randn_like(x_start))
        
        # 前向加噪
        x_noisy = self.q_sample(x_start=x_start, timesteps=timesteps, noise=noise)
        
        # 模型预测
        model_output = self.denoise_model(x_noisy, timesteps, condition_features)
        
        if self.parameterization == "eps":
            target = noise
            # 关键修改：无梯度完整逆向去噪生成x0
            with torch.no_grad():
                x_current = torch.randn_like(x_start)
            
                # 从t_start到0逆向去噪
                for step_idx in reversed(range(0, self.num_timesteps)):
                    current_t = torch.full((x_current.shape[0],), step_idx, device=x_noisy.device, dtype=torch.long)
                    x_current = self.p_sample(
                        x_current,
                        current_t,
                        condition_features=condition_features,
                        clip_denoised=self.clip_denoised
                    )
                pred_x0 = x_current  # 去噪完成后得到x0
                if self.clip_denoised:
                    pred_x0 = torch.clamp(pred_x0, -1., 1.)
            return model_output, target, pred_x0  # model_output仍为预测噪声，用于MSE损失
        else:
            target = x_start
            return model_output, target


        # 4. 返回预测结果和目标（不计算损失）
        

    def sample(self, shape, condition_input=None):
        """从噪声生成样本（推理过程）"""
        device = self.betas.device
        batch_size = shape[0]
        
        # 从纯噪声开始
        generated_sample = torch.randn(shape, device=device)
        
        # 处理条件输入
        condition_features = self.condition_model(condition_input) if exists(self.condition_model) else condition_input

        # 逆向去噪循环
        for step_idx in reversed(range(0, self.num_timesteps)):
            generated_sample = self.p_sample(generated_sample,
                                torch.full((batch_size,), step_idx, device=device, dtype=torch.long),
                                condition_features=condition_features,
                                clip_denoised=self.clip_denoised)
        return generated_sample

    def forward(self, x_original=None, condition_input=None, num_pred=512, dim=12, training_mode='standard'):
        """
        主入口函数
        
        Args:
            x_original: 原始数据（训练时必需）
            condition_input: 条件输入
            num_pred: 预测点数量
            dim: 每个点的维度
            training_mode: 训练模式 ('standard' 或 'full_reverse')
        
        Returns:
            训练时返回预测结果和目标，推理时返回生成结果
        """
        device = self.betas.device
        
        if not exists(condition_input):
            raise ValueError("condition_input must be provided.")
            
        batch_size = condition_input.shape[0]

        if self.training:
            if not exists(x_original):
                raise ValueError("x_original must be provided during training.")
            
            if training_mode == 'standard':
                # 关键修改：生成batch内统一的时间步t（所有样本使用相同t）
                t = torch.randint(0, self.num_timesteps, (1,), device=device).long().item()  # 随机选一个t
                timesteps = torch.full((batch_size,), t, device=device).long()  # 所有样本共享t
                condition_features = self.condition_model(condition_input) if exists(self.condition_model) else condition_input
                if self.parameterization == "eps":
                    model_output, target, pred_x0 = self.p_losses(x_start=x_original, timesteps=timesteps, condition_features=condition_features)
                    return model_output, target, pred_x0
                else:
                    model_output, target = self.p_losses(x_start=x_original, timesteps=timesteps, condition_features=condition_features)
                    return model_output, target

            elif training_mode == 'full_reverse':
                # 完整去噪训练模式
                t_start = torch.full((batch_size,), self.num_timesteps - 1, device=device, dtype=torch.long)
                noise = torch.randn_like(x_original)
                x_prediction = self.q_sample(x_start=x_original, timesteps=t_start, noise=noise)
                condition_features = self.condition_model(condition_input) if exists(self.condition_model) else condition_input
                
                for step_idx in reversed(range(0, self.num_timesteps)):
                    x_prediction = self.p_sample(x_prediction,
                                        torch.full((batch_size,), step_idx, device=device, dtype=torch.long),
                                        condition_features=condition_features,
                                        clip_denoised=self.clip_denoised)
                return x_prediction, x_original

            else:
                raise ValueError(f"Unknown training_mode: {training_mode}")

        else:  # 推理模式
            shape = (batch_size, num_pred, dim)
            return self.sample(shape=shape, condition_input=condition_input)
