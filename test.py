# 文件名: run_test.py
# 位置: /root/shared-nvme/PoinTr-baseline-c/run_test.py
import torch
import torch.nn as nn
import sys
import types
import time
from collections import OrderedDict

# --- 核心逻辑：从模型文件中导入我们想要测试的类 ---
# 这样导入可以确保模型代码只被执行一次，避免了重复注册的问题。
from models.DiffSymm_refine import DiffSymm_refine
# --- 模拟和准备工作 (与之前相同) ---
from extensions.chamfer_dist import ChamferDistanceL1

print("--- 准备测试环境: 正在模拟依赖项... ---")

# 1. 模拟 (Mock) MODELS 注册器和构建器
#    虽然我们不再有重复注册的问题，但在独立测试时仍需模拟这个框架依赖。
class MockModelsRegistry:
    def build(self, config):
        print(f"  - (Mock) 正在构建 base_model，使用 nn.Identity() 作为占位符。")
        return nn.Identity()
    def register_module(self):
        # 在这个测试文件中，这个装饰器什么都不用做
        return lambda x: x

# 由于 DiffSymm_refine 已经导入并使用了原始的 MODELS 对象,
# 我们需要用 monkey-patch 的方式替换掉它，以便模型能成功初始化。
import models.DiffSymm_refine as model_file
model_file.MODELS = MockModelsRegistry()

# 2. 定义一个能同时支持 .属性 和 .get() 方法的 MockConfig 类
class MockConfig(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

print("--- 依赖项模拟完成。开始测试模型... ---")

# 3. 创建配置对象
config = MockConfig({
    'up_factors': '2,2,4',
    'num_proxy_steps': 5,
    'use_proxy_refiner': False,
    'pretrain': None,
    'base_model': {},
    'diffusion_cfg': {
        'training_mode': 'standard',
        'beta_schedule': 'linear',
        'ddim_num_steps': 50,
        'ddim_discretize': 'uniform',
        'ddim_eta': 0.0,
    }
})

# --- 辅助函数：计算模型参数量 ---
def count_parameters(model):
    """计算模型的参数量"""
    total_params = 0
    trainable_params = 0
    
    param_details = {}
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        if param.requires_grad:
            trainable_params += param_count
            
        # 记录每个模块的参数量
        module_name = name.split('.')[0] if '.' in name else name
        if module_name not in param_details:
            param_details[module_name] = {'total': 0, 'trainable': 0}
        
        param_details[module_name]['total'] += param_count
        if param.requires_grad:
            param_details[module_name]['trainable'] += param_count
    
    return total_params, trainable_params, param_details

def format_number(num):
    """格式化数字显示"""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)

# --- 辅助函数：计算FLOPs ---
def calculate_flops(model, input_tensor):
    """简单的FLOPs计算"""
    flops = 0
    
    def flop_count_hook(module, input, output):
        nonlocal flops
        
        if isinstance(module, nn.Conv1d):
            # Conv1d: FLOPs = batch_size * output_length * kernel_size * in_channels * out_channels
            if hasattr(output, 'shape'):
                batch_size, out_channels, out_length = output.shape
                kernel_size = module.kernel_size[0]
                in_channels = module.in_channels
                flops += batch_size * out_length * kernel_size * in_channels * out_channels
                
        elif isinstance(module, nn.Linear):
            # Linear: FLOPs = batch_size * input_features * output_features
            if hasattr(output, 'shape') and len(output.shape) >= 2:
                batch_size = output.shape[0]
                output_features = output.shape[-1]
                input_features = module.in_features
                flops += batch_size * input_features * output_features
                
        elif isinstance(module, nn.BatchNorm1d):
            # BatchNorm: FLOPs = batch_size * num_features * length
            if hasattr(output, 'shape'):
                flops += output.numel()
    
    # 注册hook
    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear, nn.BatchNorm1d)):
            hooks.append(module.register_forward_hook(flop_count_hook))
    
    # 运行一次前向传播
    with torch.no_grad():
        model(input_tensor)
    
    # 移除hook
    for hook in hooks:
        hook.remove()
    
    return flops

# --- 辅助函数：测量推理时间 ---
def measure_inference_time(model, input_tensor, num_runs=10):
    """测量推理时间"""
    model.eval()
    
    # 预热
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_tensor)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # 测量时间
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model(input_tensor)
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    return avg_time, times

# --- 正式开始测试 ---
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n--- 正在设备: {device} 上运行测试 ---")
    
    model = DiffSymm_refine(config=config).to(device)
    model.eval()
    print("✅ 模型 DiffSymm_refine 实例化成功。")
    
    batch_size = 1
    num_points = 2048
    
    # 创建 (B, N, 3) 形状的张量
    dummy_point_cloud = torch.randn(batch_size, num_points, 3).to(device)
    print(f"✅ 创建虚拟输入张量，形状为: {dummy_point_cloud.shape}")
    
    # === 计算模型参数量 ===
    print("\n" + "="*60)
    print("📊 模型参数统计")
    print("="*60)
    
    total_params, trainable_params, param_details = count_parameters(model)
    
    print(f"总参数量: {format_number(total_params)} ({total_params:,})")
    print(f"可训练参数量: {format_number(trainable_params)} ({trainable_params:,})")
    print(f"不可训练参数量: {format_number(total_params - trainable_params)} ({total_params - trainable_params:,})")
    
    print(f"\n各模块参数分布:")
    for module_name, counts in param_details.items():
        print(f"  - {module_name}: {format_number(counts['total'])} (可训练: {format_number(counts['trainable'])})")
    
    # === 计算FLOPs ===
    print("\n" + "="*60)
    print("⚡ 计算复杂度统计")
    print("="*60)
    
    print("正在计算FLOPs...")
    flops = calculate_flops(model, dummy_point_cloud)
    print(f"FLOPs: {format_number(flops)} ({flops:,})")
    
    # === 测量推理时间 ===
    print("\n正在测量推理时间...")
    avg_time, times = measure_inference_time(model, dummy_point_cloud)
    print(f"平均推理时间: {avg_time*1000:.2f} ms")
    print(f"推理时间范围: {min(times)*1000:.2f} - {max(times)*1000:.2f} ms")
    
    # === 内存使用情况 ===
    if torch.cuda.is_available():
        print("\n" + "="*60)
        print("💾 GPU内存使用情况")
        print("="*60)
        
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**2   # MB
        
        print(f"GPU内存分配: {memory_allocated:.2f} MB")
        print(f"GPU内存保留: {memory_reserved:.2f} MB")
    
    # === 前向传播测试 ===
    print("\n" + "="*60)
    print("🚀 前向传播测试")
    print("="*60)
    
    with torch.no_grad():
        outputs = model(dummy_point_cloud)
    
    print("✅ 前向传播成功")
    print("模型返回一个包含多个点云的列表，其形状如下:")
    
    output_names = ['coarse', 'fine1', 'fine2', 'fine3']
    for i, out_tensor in enumerate(outputs):
        print(f"  - 输出 {i} ({output_names[i]}): {out_tensor.shape}")
    
    # === 总结 ===
    print("\n" + "="*60)
    print("📋 测试总结")
    print("="*60)
    print(f"✅ 模型名称: DiffSymm_refine")
    print(f"✅ 输入形状: {dummy_point_cloud.shape}")
    print(f"✅ 输出数量: {len(outputs)}")
    print(f"✅ 总参数量: {format_number(total_params)}")
    print(f"✅ 可训练参数: {format_number(trainable_params)}")
    print(f"✅ 计算复杂度: {format_number(flops)} FLOPs")
    print(f"✅ 平均推理时间: {avg_time*1000:.2f} ms")
    print(f"✅ 设备: {device}")
    
    print("\n🎉 测试完成，所有问题已解决！")
    
except Exception as e:
    print("\n--- ❌ 测试过程中发生错误 ---")
    print(f"错误类型: {type(e).__name__}")
    print(f"错误信息: {e}")
    import traceback
    traceback.print_exc()