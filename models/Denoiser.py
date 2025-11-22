import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


def positional_encoding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    生成正弦位置编码
    
    Args:
        timesteps: [B] 时间步
        dim: 编码维度
        max_period: 最大周期
    
    Returns:
        [B, dim] 位置编码
    """
    half = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(max_period, dtype=torch.float32)) *
        torch.arange(half, dtype=torch.float32) / half
    ).to(timesteps.device)
    
    args = timesteps[:, None].float() * freqs[None, :]
    encoding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    
    # 处理奇数维度
    if dim % 2:
        encoding = torch.cat([encoding, torch.zeros_like(encoding[:, :1])], dim=-1)
    
    return encoding


class TimeEmbedding(nn.Module):
    """时间编码MLP"""
    
    def __init__(self, time_dim: int, embed_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, embed_dim),
            nn.SiLU(),  # SiLU在扩散模型中表现更好
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(t)


class ConvBlock(nn.Module):
    """
    1D卷积块，支持时间编码和条件注入
    使用GroupNorm以提供更好的训练稳定性
    """
    
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 time_emb_dim: Optional[int] = None, 
                 cond_channels: int = 0,
                 groups: int = 8):
        super().__init__()
        
        # 时间编码投影
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        ) if time_emb_dim is not None else None

        # 计算总输入通道数（包含条件信息）
        total_in_channels = in_channels + cond_channels
        
        # 第一个卷积层
        self.conv1 = nn.Conv1d(total_in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(min(groups, out_channels), out_channels)
        
        # 第二个卷积层
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(groups, out_channels), out_channels)
        
        # 残差连接
        self.residual_conv = (
            nn.Conv1d(total_in_channels, out_channels, 1) 
            if total_in_channels != out_channels 
            else nn.Identity()
        )
        
        # 激活函数
        self.act = nn.SiLU()

    def forward(self, 
                x: torch.Tensor, 
                t_emb: Optional[torch.Tensor] = None, 
                cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [B, C_in, S] 输入特征
            t_emb: [B, time_emb_dim] 时间编码
            cond: [B, C_cond, S] 条件特征
        
        Returns:
            [B, C_out, S] 输出特征
        """
        # 保存输入用于残差连接
        residual_input = x
        # 拼接条件信息
        if cond is not None:
            x = torch.cat([x, cond], dim=1)
            residual_input = x
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        
        # 注入时间编码
        if t_emb is not None and self.time_mlp is not None:
            t_add = self.time_mlp(t_emb)  
            h = h + t_add.unsqueeze(-1)   # [B, C_out]  广播到 [B, C_out, S]
        # 第二个卷积块
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        #print(f"ConvBlock: x shape {x.shape}, h shape {h.shape}, residual_input shape {residual_input.shape}")
        return h + self.residual_conv(residual_input) # 残差连接

class DownBlock(nn.Module):
    """下采样块"""
    
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 time_emb_dim: int, 
                 cond_channels: int):
        super().__init__()
        
        self.conv_block = ConvBlock(in_channels, out_channels, time_emb_dim, cond_channels)
        self.downsample = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        
        # 条件信息也需要下采样
        self.cond_downsample = (
            nn.Conv1d(cond_channels, cond_channels, kernel_size=3, stride=2, padding=1) 
            if cond_channels > 0 else None
        )

    def forward(self, 
                x: torch.Tensor, 
                t_emb: torch.Tensor, 
                cond: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: [B, C_in, S] 输入特征
            t_emb: [B, time_emb_dim] 时间编码
            cond: [B, C_cond, S] 条件特征
        
        Returns:
            downsampled_x: [B, C_out, S//2] 下采样后的特征
            skip_x: [B, C_out, S] 跳跃连接特征
            downsampled_cond: [B, C_cond, S//2] 下采样后的条件
        """
        # 卷积处理
        x_processed = self.conv_block(x, t_emb, cond)
        skip_x = x_processed
        #print(f"skip_x shape: {skip_x.shape}")
        
        # 下采样主特征
        downsampled_x = self.downsample(x_processed)
        
        # 下采样条件特征
        downsampled_cond = None
        if cond is not None and self.cond_downsample is not None:
            downsampled_cond = self.cond_downsample(cond)
        
        return downsampled_x, skip_x, downsampled_cond


class UpBlock(nn.Module):
    """上采样块"""
    
    def __init__(self, 
                 in_channels: int, 
                 skip_channels: int, 
                 out_channels: int, 
                 time_emb_dim: int, 
                 cond_channels: int):
        super().__init__()
        
        # 上采样层
        self.upsample = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=2, stride=2)
        
        # 拼接后的总通道数
        total_channels = in_channels + skip_channels
        self.conv_block = ConvBlock(total_channels, out_channels, time_emb_dim, cond_channels)
        
        # 条件信息上采样
        self.cond_upsample = (
            nn.ConvTranspose1d(cond_channels, cond_channels, kernel_size=2, stride=2) 
            if cond_channels > 0 else None
        )

    def forward(self, 
                x: torch.Tensor, 
                skip_x: torch.Tensor, 
                t_emb: torch.Tensor, 
                cond: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: [B, C_in, S] 来自上层的特征
            skip_x: [B, C_skip, S*2] 跳跃连接特征
            t_emb: [B, time_emb_dim] 时间编码
            cond: [B, C_cond, S] 当前条件特征
        
        Returns:
            x_out: [B, C_out, S*2] 输出特征
            upsampled_cond: [B, C_cond, S*2] 上采样后的条件
        """
        # 上采样主特征
        x_upsampled = self.upsample(x)
        
        # 上采样条件特征
        upsampled_cond = None
        if cond is not None and self.cond_upsample is not None:
            upsampled_cond = self.cond_upsample(cond)
    
        # 确保尺寸匹配（处理潜在的尺寸不匹配）
        if skip_x.shape[-1] != x_upsampled.shape[-1]:
            x_upsampled = F.interpolate(x_upsampled, size=skip_x.shape[-1], mode='linear', align_corners=False)
        
        # 拼接跳跃连接
        x_concat = torch.cat([skip_x, x_upsampled], dim=1)
        # 卷积处理
        x_out = self.conv_block(x_concat, t_emb, upsampled_cond)
        
        return x_out, upsampled_cond


class UNet_Conv1D_Denoiser(nn.Module):
    """
    1D U-Net去噪器，用于扩散模型
    
    特点：
    - 支持条件生成
    - 时间编码注入
    - 跳跃连接
    - 分组归一化
    """
    
    def __init__(self,
                 latent_dim: int = 12,
                 time_dim: int = 128,
                 cond_dim: int = 128,
                 base_dim: int = 64,
                 dim_mults: Tuple[int, ...] = (1, 2, 4),
                 groups: int = 8):
        super().__init__()

        self.latent_dim = latent_dim
        self.time_dim = time_dim
        self.cond_dim = cond_dim
        
        # 时间编码维度
        time_emb_dim = base_dim * 4
        self.time_embedding = TimeEmbedding(time_dim, time_emb_dim)

        self.input_conv = nn.Conv1d(latent_dim, base_dim, kernel_size=1)
        
        cond_channels = base_dim // 2 # 这个控制了条件特征的通道数
        self.cond_conv = nn.Conv1d(cond_dim, cond_channels, kernel_size=1) if cond_dim > 0 else None
        
        # 计算各层通道数
        dims = [base_dim] + [base_dim * mult for mult in dim_mults]  # U-Net通道数: [64, 64, 128, 256] 

        # 编码器（下采样路径）
        self.downs = nn.ModuleList()
        for i in range(len(dim_mults)):
            self.downs.append(DownBlock(
                in_channels=dims[i],
                out_channels=dims[i + 1],
                time_emb_dim=time_emb_dim,
                cond_channels=cond_channels if cond_dim > 0 else 0
            ))

        # 瓶颈层
        mid_dim = dims[-1]
        self.mid_block1 = ConvBlock(mid_dim, mid_dim, time_emb_dim, cond_channels if cond_dim > 0 else 0)
        self.mid_block2 = ConvBlock(mid_dim, mid_dim, time_emb_dim, cond_channels if cond_dim > 0 else 0)

        # 解码器（上采样路径）
        self.ups = nn.ModuleList()
        for i in reversed(range(len(dim_mults))):
            self.ups.append(UpBlock(
                in_channels=dims[i + 1],
                skip_channels=dims[i + 1],  # 跳跃连接来自对应下采样层的输出
                out_channels=dims[i],
                time_emb_dim=time_emb_dim,
                cond_channels=cond_channels if cond_dim > 0 else 0
            ))

        # 输出投影
        self.output_conv = nn.Sequential(
            nn.GroupNorm(min(groups, base_dim), base_dim),
            nn.SiLU(),
            nn.Conv1d(base_dim, latent_dim, kernel_size=1)
        )

    def forward(self, 
                x: torch.Tensor, 
                time: torch.Tensor, 
                cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [B, S, latent_dim] 输入噪声特征
            time: [B] 时间步
            cond: [B, S, cond_dim] 条件特征（可选）
        
        Returns:
            [B, S, latent_dim] 预测的噪声
        """
        B, S, _ = x.shape

        # 转置为Conv1d格式: [B, S, C] -> [B, C, S]
        x = x.transpose(1, 2)
        
        # 处理条件信息
        current_cond = None
        if cond is not None and self.cond_conv is not None:
            cond = cond.transpose(1, 2)
            current_cond = self.cond_conv(cond) # [B, 32, 512]

        # 时间编码
        t_emb = positional_encoding(time, self.time_dim)
        t_emb = self.time_embedding(t_emb) # [B, time_emb_dim] [B, 256] 

        # 输入投影
        x = self.input_conv(x) # [B, 12, 512] -> [B, 64, 512]

        skips = [] # 编码器路径 - 收集跳跃连接
        for down_block in self.downs:
            x, skip, current_cond = down_block(x, t_emb, current_cond)
            skips.append(skip)

        # 瓶颈层
        x = self.mid_block1(x, t_emb, current_cond)
        x = self.mid_block2(x, t_emb, current_cond)

        # 解码器路径 - 使用跳跃连接
        for up_block in self.ups:
            skip_x = skips.pop()
            x, current_cond = up_block(x, skip_x, t_emb, current_cond)

        # 输出投影
        x = self.output_conv(x)
        
        # 转置回原始格式: [B, C, S] -> [B, S, C]
        return x.transpose(1, 2)
        
class LightweightPointDenoiser(nn.Module):
    """
    一个为点云特征去噪而优化的、轻量化的1D U-Net。

    核心优化点:
    - 通过 `base_dim` 和 `dim_mults` 控制模型大小。
    - 结构与原版一致，但参数更少，计算更快。
    - 完全兼容混合精度训练。
    """
    def __init__(self,
                 latent_dim: int = 12,
                 time_dim: int = 128,
                 cond_dim: int = 128,
                 # --- 轻量化关键参数 ---
                 base_dim: int = 32,  # 基础通道数，从64降低到32
                 dim_mults: Tuple[int, ...] = (1, 2, 3), # 通道增长乘数，将最深的x4改为x3
                 groups: int = 4      # GroupNorm的组数，可以适当减小
                ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.time_dim = time_dim
        self.cond_dim = cond_dim
        
        # 1. 时间编码
        time_emb_dim = base_dim * 4
        self.time_embedding = TimeEmbedding(time_dim, time_emb_dim)

        # 2. 输入层
        self.input_conv = nn.Conv1d(latent_dim, base_dim, kernel_size=1)
        
        # 3. 条件射影
        # 条件特征的通道数，设为基础维度的一半，保持轻量
        cond_channels = base_dim // 2
        self.cond_conv = nn.Conv1d(cond_dim, cond_channels, kernel_size=1) if cond_dim > 0 else None
        
        # 4. U-Net 路径
        # 计算各层通道数，例如 base_dim=32, dim_mults=(1,2,3) -> [32, 32, 64, 96]
        dims = [base_dim] + [base_dim * mult for mult in dim_mults]

        # --- 编码器 (Down Blocks) ---
        self.downs = nn.ModuleList()
        for i in range(len(dim_mults)):
            self.downs.append(DownBlock(
                in_channels=dims[i],
                out_channels=dims[i+1],
                time_emb_dim=time_emb_dim,
                cond_channels=cond_channels if cond_dim > 0 else 0
            ))

        # --- 瓶颈层 (Bottleneck) ---
        mid_dim = dims[-1]
        self.mid_block = ConvBlock(mid_dim, mid_dim, time_emb_dim, cond_channels if cond_dim > 0 else 0)

        # --- 解码器 (Up Blocks) ---
        self.ups = nn.ModuleList()
        for i in reversed(range(len(dim_mults))):
            self.ups.append(UpBlock(
                in_channels=dims[i+1],
                skip_channels=dims[i+1],
                out_channels=dims[i],
                time_emb_dim=time_emb_dim,
                cond_channels=cond_channels if cond_dim > 0 else 0
            ))

        # 5. 输出层
        self.output_conv = nn.Sequential(
            nn.GroupNorm(min(groups, base_dim), base_dim),
            nn.SiLU(),
            nn.Conv1d(base_dim, latent_dim, kernel_size=1)
        )
        
        # 打印参数量，方便调试
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[LightweightPointDenoiser] Model initialized. Total parameters: {total_params:,}")

    def forward(self, 
                x: torch.Tensor, 
                time: torch.Tensor, 
                cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # 输入格式转换: [B, S, C] -> [B, C, S]
        x = x.transpose(1, 2)
        
        # 条件处理
        current_cond = None
        if cond is not None and self.cond_conv is not None:
            cond = cond.transpose(1, 2)
            current_cond = self.cond_conv(cond)

        # 时间编码
        t_emb = positional_encoding(time, self.time_dim)
        t_emb = self.time_embedding(t_emb)
        
        # U-Net 主体
        x = self.input_conv(x)
        
        skips = []
        for down_block in self.downs:
            x, skip, current_cond = down_block(x, t_emb, current_cond)
            skips.append(skip)

        x = self.mid_block(x, t_emb, current_cond)

        for up_block in self.ups:
            skip_x = skips.pop()
            x, current_cond = up_block(x, skip_x, t_emb, current_cond)

        # 输出
        x = self.output_conv(x)
        
        # 输出格式转换回: [B, C, S] -> [B, S, C]
        return x.transpose(1, 2)

# 测试代码
if __name__ == "__main__":
    def test_model():
        """测试模型的基本功能"""
        print("开始测试U-Net模型...")
        
        # 模型参数
        batch_size = 4
        seq_len = 512  # 使用2的幂次方以便下采样
        latent_dim = 12
        cond_dim = 128
        
        # 创建模型
        model = LightweightPointDenoiser(
            latent_dim=latent_dim,
            time_dim=128,
            cond_dim=cond_dim,
            base_dim=32,
            dim_mults=(1, 2, 3)
        )

        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型参数数量: {total_params:,}")
        
        # 创建测试数据
        x = torch.randn(batch_size, seq_len, latent_dim)
        time = torch.randint(0, 1000, (batch_size,))
        cond = torch.randn(batch_size, seq_len, cond_dim)
        print(f"输入形状: {x.shape}, 时间步形状: {time.shape}, 条件形状: {cond.shape}")
        # 前向传播测试
        model.eval()
        with torch.no_grad():
            output = model(x, time, cond)
        
        print(f"有条件测试的输出形状: {output.shape}")
        
        # 验证形状
        assert output.shape == x.shape, f"输出形状 {output.shape} 与输入形状 {x.shape} 不匹配!"
        
        # 测试无条件生成
        print("\n测试无条件生成...")
        model_no_cond = UNet_Conv1D_Denoiser(
            latent_dim=latent_dim,
            cond_dim=0,  # 无条件
            base_dim=32,
            dim_mults=(1, 2)
        )
        
        with torch.no_grad():
            output_no_cond = model_no_cond(x, time, None)
        
        print(f"无条件的测试输出形状: {output_no_cond.shape}")
        assert output_no_cond.shape == x.shape, "无条件生成形状不匹配!"
        
        print("\n✅ 所有测试通过!")
        return model
    
    # 运行测试
    model = test_model()