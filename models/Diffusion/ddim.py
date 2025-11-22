import torch.nn as nn
from functools import partial
import torch
import torch.nn.functional as F
import random
import numpy as np
import sys
from typing import Optional, Union, List
from utils.logger import *
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

def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    """
    生成DDIM采样的时间步序列
    Args:
        ddim_discr_method: 时间步选择方法 ('uniform', 'quad')
        num_ddim_timesteps: DDIM采样步数
        num_ddpm_timesteps: 原始DDPM总步数
    """
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps # c是DDIM时间步与DDPM时间步的间隔，也就是DDIM的跳步步长 delta T
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    # 确保最后一个时间步是num_ddpm_timesteps-1
    ddim_timesteps[-1] = num_ddpm_timesteps - 1
    if verbose:
        print(f'Selected timesteps for ddim sampler: {ddim_timesteps}')
    
    return ddim_timesteps

def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    """
    计算DDIM采样所需的参数
    Args:
        alphacums: 累积alpha值 ᾱ_t = ∏_{s=1}^t α_s
        ddim_timesteps: DDIM时间步序列
        eta: 随机性控制参数，0表示完全确定性，1表示与DDPM相同
    """
    # 选择对应时间步的alpha累积值
    alphas_cumprod_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())
    
    # 计算DDIM采样参数
    # σ_t = η * sqrt((1 - ᾱ_{t-1}) / (1 - ᾱ_t)) * sqrt(1 - ᾱ_t / ᾱ_{t-1})
    sigmas = eta * np.sqrt((1 - alphas_cumprod_prev) / (1 - alphacums[ddim_timesteps])) * np.sqrt(
        1 - alphacums[ddim_timesteps] / alphas_cumprod_prev)
    
    if verbose:
        print(f'Selected alphas for ddim sampler: a_t: {alphacums[ddim_timesteps]}; a_(t-1): {alphas_cumprod_prev}')
        print(f'For the chosen value of eta, which is {eta}, '
              f'this results in the following sigma_t schedule for ddim sampler {sigmas}')
    
    return sigmas, alphas_cumprod_prev

class DDIM(nn.Module):
    """
    DDIM (Denoising Diffusion Implicit Models) 实现
    
    DDIM是DDPM的确定性变体，主要特点：
    1. 非马尔可夫过程：x_{t-1}的生成依赖于x_t和x_0的预测
    2. 确定性采样：当eta=0时为完全确定性
    3. 加速采样：可以跳过时间步，大幅减少采样步数
    4. 保持训练兼容性：使用相同的训练目标和网络
    
    核心DDIM采样公式：
    x_{t-1} = sqrt(ᾱ_{t-1}) * pred_x0 + sqrt(1 - ᾱ_{t-1} - σ_t²) * pred_eps + σ_t * eps
    """
    
    def __init__(self,
                 denoise_model=None,                # 去噪网络模型
                 condition_model=None,              # 条件处理模型
                 timesteps=1000,                   # 训练时的扩散步数
                 beta_schedule="linear",           # beta调度策略
                 clip_denoised=False,              # 是否裁剪去噪结果
                 linear_start=1e-4,                # 线性调度起始值
                 linear_end=2e-2,                  # 线性调度结束值
                 cosine_s=8e-3,                    # 余弦调度参数
                 given_betas=None,                 # 预定义beta值
                 v_posterior=0.,                   # 后验方差调节参数
                 parameterization="eps",           # 参数化方式："eps"或"x0"
                 # DDIM特有参数
                 ddim_num_steps=50,                # DDIM采样步数
                 ddim_discretize="uniform",        # 时间步选择方法
                 ddim_eta=0.0,                    # 随机性参数，0为完全确定性
                 ddim_clip_denoised=False,         # DDIM采样时是否裁剪
                 ):

        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        
        # 模型配置
        self.parameterization = parameterization
        self.clip_denoised = clip_denoised
        self.denoise_model = denoise_model      # 标准化：去噪网络
        self.condition_model = condition_model  # 标准化：条件模型
        self.v_posterior = v_posterior

        # DDIM特有参数
        self.ddim_num_steps = ddim_num_steps
        self.ddim_discretize = ddim_discretize
        self.ddim_eta = ddim_eta
        self.ddim_clip_denoised = ddim_clip_denoised
        
        # 注册扩散调度参数
        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, 
                               timesteps=timesteps, linear_start=linear_start, 
                               linear_end=linear_end, cosine_s=cosine_s)

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

        # 后验分布q(x_{t-1} | x_t, x_0)相关参数（DDPM兼容）
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', 
                           to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(1. - betas) / (1. - alphas_cumprod)))

        # 准备DDIM采样参数
        self.make_ddim_schedule()

    def make_ddim_schedule(self, verbose=True):
        """
        准备DDIM采样调度 (最终版)
        新增：预计算所有步骤的 alpha_cumprod_prev
        """
        # 生成DDIM时间步序列 (这部分不变)
        ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=self.ddim_discretize,
            num_ddim_timesteps=self.ddim_num_steps,
            num_ddpm_timesteps=self.num_timesteps,
            verbose=verbose
        )
        
        alphas_cumprod = self.alphas_cumprod.cpu().numpy()
        
        # 计算DDIM采样参数 sigmas (这部分不变)
        ddim_sigmas, _ = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod,
            ddim_timesteps=ddim_timesteps,
            eta=self.ddim_eta,
            verbose=verbose
        )
        
        # ==========================================================
        # ✨ 核心修复：预计算 alpha_cumprod 和 alpha_cumprod_prev ✨
        # ==========================================================
        # ddim_timesteps 是我们采样的具体时间点, 例如 [0, 20, 40, ...]
        self.register_buffer('ddim_timesteps', torch.from_numpy(ddim_timesteps).long())
        
        # 对应 ddim_timesteps 中每个时间点的 alpha_cumprod 值
        ddim_alphas_cumprod = torch.from_numpy(alphas_cumprod[ddim_timesteps]).float()
        
        # 关键：预计算每个 ddim 步骤的 "前一个" alpha_cumprod 值
        # 对于 ddim_timesteps[0]，它的前一个是 1.0
        # 对于 ddim_timesteps[i]，它的前一个是 alphas_cumprod[ddim_timesteps[i-1]]
        ddim_alphas_cumprod_prev = torch.tensor([1.0] + alphas_cumprod[ddim_timesteps[:-1]].tolist(), dtype=torch.float32)

        self.register_buffer('ddim_alphas_cumprod', ddim_alphas_cumprod)
        self.register_buffer('ddim_alphas_cumprod_prev', ddim_alphas_cumprod_prev)
        self.register_buffer('ddim_sigmas', torch.from_numpy(ddim_sigmas).float())


    def q_mean_variance(self, x_start, timesteps):
        """计算前向过程q(x_t|x_0)的均值和方差"""
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, timesteps, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, timesteps, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, timesteps, predicted_noise):
        """
        从噪声预测中恢复原始数据x_0
        基于公式：x_0 = (x_t - sqrt(1-ᾱ_t) * ε) / sqrt(ᾱ_t)
        """
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, timesteps, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, timesteps, x_t.shape) * predicted_noise
        )

    def predict_noise_from_start(self, x_t, timesteps, x0):
        """
        从x_0预测噪声
        基于公式：ε = (x_t - sqrt(ᾱ_t) * x_0) / sqrt(1-ᾱ_t)
        """
        return (
                (extract_into_tensor(self.sqrt_recip_alphas_cumprod, timesteps, x_t.shape) * x_t - x0) /
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, timesteps, x_t.shape)
        )

    def q_posterior(self, x_start, x_t, timesteps):
        """计算后验分布q(x_{t-1}|x_t,x_0)的均值和方差（DDPM兼容）"""
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, timesteps, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, timesteps, x_t.shape) * x_t
        )
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, timesteps, x_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x_noisy, timesteps, condition_features, clip_denoised: bool):
        """计算反向过程p_θ(x_{t-1}|x_t)的均值和方差（DDPM训练兼容）"""
        model_output = self.denoise_model(x_noisy, timesteps, condition_features)
        
        if self.parameterization == "eps":
            x_reconstructed = self.predict_start_from_noise(x_noisy, t=timesteps, predicted_noise=model_output)
        elif self.parameterization == "x0":
            x_reconstructed = model_output
        else:
            raise NotImplementedError()
            
        if clip_denoised:
            x_reconstructed.clamp_(-1., 1.)
            
        model_mean, posterior_log_variance = self.q_posterior(x_start=x_reconstructed, x_t=x_noisy, timesteps=timesteps)
        return model_mean, posterior_log_variance, x_reconstructed

    def p_sample(self, x_current, timesteps, condition_features, clip_denoised=True, repeat_noise=False):
        """
        DDPM风格的一步采样（训练时使用）
        """
        batch_size, *_, device = *x_current.shape, x_current.device
        model_mean, model_log_variance, _ = self.p_mean_variance(x_noisy=x_current, timesteps=timesteps, 
                                                               condition_features=condition_features, 
                                                               clip_denoised=clip_denoised)
        
        nonzero_mask = (timesteps != 0).float().view(batch_size, *((1,) * (len(x_current.shape) - 1)))
        noise = noise_like(x_current.shape, device, repeat_noise)
        
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def q_sample(self, x_start, timesteps, noise=None):
        """前向扩散过程：从x_0采样x_t"""
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape) * noise)

    def p_losses(self, x_start, timesteps, condition_features, noise=None):
        """计算模型预测结果（不计算损失）"""
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, timesteps=timesteps, noise=noise)# 1. 前向加噪
        model_output = self.denoise_model(x_noisy, timesteps, condition_features) # 2. 模型预测
        if self.parameterization == "eps":  # 3. 确定目标和预测结果
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise ValueError(f"Unknown parametrization: {self.parameterization}")        
        return model_output, target # 4. 返回预测结果和目标（不计算损失）

    def sample(self, shape, condition_input=None):
        """从噪声生成样本（推理过程）"""
        device = self.betas.device
        batch_size = shape[0]
        
        generated_sample = torch.randn(shape, device=device)
        condition_features = self.condition_model(condition_input) if exists(self.condition_model) else condition_input

        # DDIM逆向去噪循环
        for step_idx in reversed(range(self.ddim_num_steps)):
            # 计算当前步骤的timesteps
            current_timestep = self.ddim_timesteps[step_idx]
            timesteps = current_timestep.expand(batch_size) # 将0维张量扩展为1维，高效且无数据依赖
            
            generated_sample, _ = self.ddim_sample(generated_sample,
                                                timesteps,
                                                step_idx,
                                                condition_features=condition_features,
                                                clip_denoised=self.ddim_clip_denoised)
        return generated_sample
    
    def ddim_sample(self, x_t, timesteps, t_idx, condition_features, clip_denoised=True, temperature=1.0, model_output=None):
        """
        DDIM采样步骤 (最终无分支版)
        将 if sigma_t > 0 的控制流转换为无条件的数学运算。
        """
        # ... (函数前半部分，直到 x_prev 的计算，都保持不变) ...
        if model_output is None:
            model_output = self.denoise_model(x_t, timesteps, condition_features)

        if self.parameterization == "eps":
            pred_eps = model_output
            pred_x0 = self.predict_start_from_noise(x_t, timesteps, pred_eps)
        else: # "x0"
            pred_x0 = model_output
            pred_eps = self.predict_noise_from_start(x_t, timesteps, pred_x0)

        if clip_denoised:
            pred_x0 = pred_x0.clamp(-1., 1.)
            if self.parameterization == "x0":
                pred_eps = self.predict_noise_from_start(x_t, timesteps, pred_x0)

        alpha_cumprod_t_prev = self.ddim_alphas_cumprod_prev[t_idx]
        sigma_t = self.ddim_sigmas[t_idx]

        pred_x0_coeff = torch.sqrt(alpha_cumprod_t_prev)
        pred_eps_coeff = torch.sqrt(1 - alpha_cumprod_t_prev - sigma_t ** 2)
        x_prev = pred_x0_coeff * pred_x0 + pred_eps_coeff * pred_eps
        
        # ==========================================================
        # ✨ 最终核心修复：用无分支数学运算替换if语句 ✨
        # ==========================================================
        # 原始代码:
        # if sigma_t > 0:
        #     noise = torch.randn_like(x_t) * temperature
        #     x_prev = x_prev + sigma_t * noise
        
        # 无分支版本:
        # 无条件地生成噪声
        noise = torch.randn_like(x_t) * temperature 
        # 无条件地添加噪声项。如果sigma_t为0，则相当于加0。
        x_prev = x_prev + sigma_t * noise
        
        return x_prev, pred_x0
    
    def ddim_sample_o(self, x_t, timesteps, t_idx, condition_features, clip_denoised=True, temperature=1.0, model_output=None):
        """
        DDIM采样步骤
        
        DDIM核心公式：
        x_{t-1} = √(ᾱ_{t-1}) * pred_x0 + √(1 - ᾱ_{t-1} - σ_t²) * pred_eps + σ_t * ε
        
        其中：
        - ᾱ_t = ∏_{s=1}^t α_s (累积alpha值)
        - σ_t = η * √((1 - ᾱ_{t-1}) / (1 - ᾱ_t)) * √(1 - ᾱ_t / ᾱ_{t-1}) (噪声系数)
        - pred_x0 是从噪声预测的清洁图像
        - pred_eps 是预测的噪声
        """
        batch_size, *_, device = *x_t.shape, x_t.device
        
        if model_output is None:
            model_output = self.denoise_model(x_t, timesteps, condition_features)

        # 根据参数化方式计算pred_x0和pred_eps
        if self.parameterization == "eps":
            pred_eps = model_output # 模型预测噪声 ε_θ(x_t, t)
            pred_x0 = self.predict_start_from_noise(x_t, timesteps, pred_eps) # 从噪声预测清洁图像：pred_x0 = (x_t - √(1-ᾱ_t) * pred_eps) / √(ᾱ_t)
        elif self.parameterization == "x0":
            pred_x0 = model_output # 模型直接预测清洁图像 x_θ(x_t, t)
            pred_eps = self.predict_noise_from_start(x_t, timesteps, pred_x0) # 从清洁图像反推噪声：pred_eps = (x_t - √(ᾱ_t) * pred_x0) / √(1-ᾱ_t)
        else:
            raise NotImplementedError()
    
        # 可选的裁剪操作，将pred_x0限制在[-1, 1]范围内
        if clip_denoised:
            pred_x0 = pred_x0.clamp(-1., 1.)
            if self.parameterization == "x0": # 如果裁剪了pred_x0，需要重新计算pred_eps以保持一致性
                pred_eps = self.predict_noise_from_start(x_t, timesteps, pred_x0)

        alpha_cumprod_t_prev = self.alphas_cumprod[self.ddim_timesteps[t_idx - 1]] if t_idx > 0 else torch.tensor(1.0, device=device)
        sigma_t = self.ddim_sigmas[t_idx]
        pred_x0_coeff = torch.sqrt(alpha_cumprod_t_prev) # 系数1：√(ᾱ_{t-1}) - 清洁图像的权重
        pred_eps_coeff = torch.sqrt(1 - alpha_cumprod_t_prev - sigma_t ** 2) # 系数2：√(1 - ᾱ_{t-1} - σ_t²) - 确定性噪声的权重
        x_prev = pred_x0_coeff * pred_x0 + pred_eps_coeff * pred_eps # x_{t-1} = √(ᾱ_{t-1}) * pred_x0 + √(1 - ᾱ_{t-1} - σ_t²) * pred_eps
        
        
        if sigma_t > 0: # 添加随机噪声项：σ_t * ε （如果σ_t > 0）
            noise = self._get_noise_tensor(x_t.shape, device, temperature) # 生成标准高斯噪声
            x_prev = x_prev + sigma_t * noise  # 添加噪声：x_{t-1} += σ_t * ε
        
        return x_prev, pred_x0

    def full_denoise_sample(self, condition_features, shape, use_grad=False):
        """优化的完整去噪"""
        device = self.betas.device
        batch_size = shape[0]
        
        # 预分配x_t
        x_t = torch.randn(shape, device=device)
        
        if not use_grad:
            with torch.no_grad():
                # 批量处理多个步骤以减少Python循环开销
                for step_idx in reversed(range(self.ddim_num_steps)):
                    # 计算当前步骤的timesteps
                    current_timestep = self.ddim_timesteps[step_idx]
                    timesteps = torch.full((batch_size,), current_timestep, device=device, dtype=torch.long)
                    
                    x_t, _ = self.ddim_sample(x_t, timesteps, step_idx, condition_features, 
                                            clip_denoised=self.ddim_clip_denoised)
        else:
            for step_idx in reversed(range(self.ddim_num_steps)):
                # 计算当前步骤的timesteps
                current_timestep = self.ddim_timesteps[step_idx]
                timesteps = torch.full((batch_size,), current_timestep, device=device, dtype=torch.long)
                
                x_t, _ = self.ddim_sample(x_t, timesteps, step_idx, condition_features, 
                                        clip_denoised=self.ddim_clip_denoised)
        
        return x_t
    
    
    def proxy_gradient_sample_compiled(self, 
                                    condition_features, 
                                    shape, 
                                    proxy_indices_set,     # 用于 'in' 判断
                                    proxy_indices_map      # 新增参数：用于索引查找
                                    ):
        """
        代理梯度采样 - 最终架构版 v2
        接收预计算的索引查找表(map)，消除所有动态形状操作。
        """
        device = self.betas.device
        batch_size = shape[0]
        
        x_t = torch.randn(shape, device=device, requires_grad=False)
        total_steps = self.ddim_num_steps
        num_proxy_steps = len(proxy_indices_set)
        
        ddim_timesteps_tensor = self.ddim_timesteps.to(device)
        
        proxy_samples_tensor = torch.zeros(num_proxy_steps, *shape, device=device, dtype=x_t.dtype)

        for step_idx in reversed(range(total_steps)):
            current_timestep = ddim_timesteps_tensor[step_idx]
            timesteps = current_timestep.expand(batch_size)

            is_proxy_step = step_idx in proxy_indices_set

            model_output = self.denoise_model(x_t, timesteps, condition_features)
            
            if self.parameterization == "eps":
                pred_x0 = self.predict_start_from_noise(x_t, timesteps, model_output)
            else:
                pred_x0 = model_output
            
            if is_proxy_step:
                # ==========================================================
                # ✨ 核心修复：使用字典进行O(1)查找，替代.nonzero() ✨
                # ==========================================================
                storage_idx = proxy_indices_map[step_idx]
                proxy_samples_tensor[storage_idx] = pred_x0
            
            x_t, _ = self.ddim_sample(
                x_t, timesteps, step_idx, condition_features, 
                clip_denoised=self.ddim_clip_denoised, 
                model_output=model_output.detach()
            )

        # 格式化输出部分也需要使用原始的 proxy_indices_tensor 来排序
        # 我们让调用者来完成这个格式化
        return x_t, proxy_samples_tensor

    def forward(self, x_original=None, condition_input=None, num_pred=512, dim=12, training_mode='standard', num_proxy_steps=None):
        """
        主入口函数
        
        Args:
            x_original: 原始数据（训练时必需）
            condition_input: 条件输入
            num_pred: 预测点数量
            dim: 每个点的维度
            training_mode: 训练模式 ('standard', 'full_reverse', 'proxy_reverse')
            num_proxy_steps: 代理梯度采样的步数（仅在proxy_reverse模式下使用）
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
            
            if training_mode == 'standard': # 标准训练：随机采样时间步，使用与DDPM相同的训练目标
                timesteps = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
                condition_features = self.condition_model(condition_input) if exists(self.condition_model) else condition_input
                
                return self.p_losses(x_start=x_original, timesteps=timesteps, condition_features=condition_features)
                
            elif training_mode == 'full_denoise': # 新增：完整去噪模式，先训练diffusion再生成高质量x_0
                timesteps = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
                condition_features = self.condition_model(condition_input) if exists(self.condition_model) else condition_input
                model_output, target = self.p_losses(x_start=x_original, timesteps=timesteps, condition_features=condition_features)
                
                shape = (batch_size, num_pred, dim) # 完整去噪生成高质量x_0（无梯度）
                denoised_x0 = self.full_denoise_sample(condition_features, shape, use_grad=False)

                return model_output, target, denoised_x0
            
            elif training_mode == 'proxy_generation':
                # --- 原子任务2: 生成代理样本 ---
                if num_proxy_steps is None:
                    raise ValueError("num_proxy_steps must be provided for 'proxy_generation' mode.")

                # ✨ 核心: 将数据准备 (创建set和map) 与计算执行 (调用compiled函数) 分离
                # 1. 数据准备
                total_steps = self.ddim_num_steps
                proxy_indices_tensor = torch.linspace(0, total_steps - 1, num_proxy_steps, device=device, dtype=torch.long)
                proxy_indices_list = proxy_indices_tensor.tolist()
                proxy_indices_set = set(proxy_indices_list)
                proxy_indices_map = {step: i for i, step in enumerate(proxy_indices_list)}
                
                shape = (batch_size, num_pred, dim)
                condition_features = self.condition_model(condition_input) if exists(self.condition_model) else condition_input

                # 2. 调用纯计算函数
                final_x_t, proxy_samples_tensor = self.proxy_gradient_sample_compiled(
                    condition_features=condition_features, 
                    shape=shape, 
                    proxy_indices_set=proxy_indices_set,
                    proxy_indices_map=proxy_indices_map
                )
                
                # 3. 在外部进行最终的格式化
                proxy_samples_with_indices = []
                sorted_proxy_indices, sort_indices = torch.sort(proxy_indices_tensor, descending=True)
                sorted_proxy_samples = proxy_samples_tensor[sort_indices]

                for i in range(num_proxy_steps):
                    proxy_samples_with_indices.append((sorted_proxy_indices[i].item(), sorted_proxy_samples[i]))

                return final_x_t, proxy_samples_with_indices
            else:
                raise ValueError(f"Unknown training_mode: {training_mode}")

        else:  # 推理模式
            shape = (batch_size, num_pred, dim)
            return self.sample(shape=shape, condition_input=condition_input)
