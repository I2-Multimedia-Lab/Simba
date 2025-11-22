import torch
import torch.nn as nn
from .build import MODELS
from extensions.chamfer_dist import ChamferDistanceL2

@MODELS.register_module()
class PCN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.number_fine = config.num_pred
        self.encoder_channel = config.encoder_channel
        grid_size = 4 # set default
        self.grid_size = grid_size
        assert self.number_fine % grid_size**2 == 0
        self.number_coarse = self.number_fine // (grid_size ** 2 )
        self.first_conv = nn.Sequential(
            nn.Conv1d(3,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128,256,1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512,512,1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512,self.encoder_channel,1)
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.encoder_channel,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,3*self.number_coarse)
        )
        self.final_conv = nn.Sequential(
            nn.Conv1d(1024+3+2,512,1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512,512,1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512,3,1)
        )
        a = torch.linspace(-0.05, 0.05, steps=grid_size, dtype=torch.float).view(1, grid_size).expand(grid_size, grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=grid_size, dtype=torch.float).view(grid_size, 1).expand(grid_size, grid_size).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, grid_size ** 2).cuda() # 1 2 S
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_func = ChamferDistanceL2()

    def get_loss(self, ret, gt, epoch=0):
        loss_coarse = self.loss_func(ret[0], gt)
        loss_fine = self.loss_func(ret[1], gt)
        return loss_coarse, loss_fine

    def forward(self, xyz):
        bs , n , _ = xyz.shape
        # encoder
        feature = self.first_conv(xyz.transpose(2,1))  # B 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # B 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# B 512 n
        feature = self.second_conv(feature) # B 1024 n
        feature_global = torch.max(feature,dim=2,keepdim=False)[0] # B 1024
        # decoder
        coarse = self.mlp(feature_global).reshape(-1,self.number_coarse,3) # B M 3
        point_feat = coarse.unsqueeze(2).expand(-1,-1,self.grid_size**2,-1) # B M S 3
        point_feat = point_feat.reshape(-1,self.number_fine,3).transpose(2,1) # B 3 N

        seed = self.folding_seed.unsqueeze(2).expand(bs,-1,self.number_coarse, -1) # B 2 M S
        seed = seed.reshape(bs,-1,self.number_fine)  # B 2 N

        feature_global = feature_global.unsqueeze(2).expand(-1,-1,self.number_fine) # B 1024 N
        feat = torch.cat([feature_global, seed, point_feat], dim=1) # B C N
    
        fine = self.final_conv(feat) + point_feat   # B 3 N

        return (coarse.contiguous(), fine.transpose(1,2).contiguous())

if __name__ == "__main__":
    import time
    try:
        from thop import profile, clever_format
        THOP_AVAILABLE = True
    except ImportError:
        print("Warning: 'thop' library not found. FLOP calculation will be skipped.")
        print("Install it with: pip install thop")
        THOP_AVAILABLE = False
    
    # Config class for testing
    class Config:
        def __init__(self):
            self.num_pred = 8192  # Number of fine points
            self.encoder_channel = 1024  # Encoder channel dimension
    
    def count_parameters(model):
        """Count the number of trainable parameters in the model"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    def measure_inference_time(model, input_tensor, num_runs=100):
        """Measure average inference time"""
        model.eval()
        
        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
        
        # Time measurement
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input_tensor)
                
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        return avg_time
    
    def test_pcn_model():
        """Test PCN model parameters and computational complexity"""
        print("=" * 60)
        print("PCN Model Testing")
        print("=" * 60)
        
        # Create config
        config = Config()
        
        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        model = PCN(config)
        
        # Handle folding_seed device placement
        if device.type == 'cpu':
            # Move folding_seed to CPU for testing
            model.folding_seed = model.folding_seed.cpu()
        
        model = model.to(device)
        model.eval()
        
        # Test with different input sizes
        input_sizes = [
            (1, 2048, 3),   # Single sample with 2048 points
            (4, 2048, 3),   # Batch of 4 samples
            (8, 1024, 3),   # Batch of 8 samples with 1024 points
        ]
        
        print("\nModel Configuration:")
        print(f"  Number of fine points: {config.num_pred}")
        print(f"  Number of coarse points: {config.num_pred // 16}")
        print(f"  Encoder channels: {config.encoder_channel}")
        print(f"  Grid size: 4x4")
        
        # Count parameters
        total_params, trainable_params = count_parameters(model)
        print(f"\nModel Parameters:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: {total_params * 4 / (1024**2):.2f} MB (float32)")
        
        print("\n" + "=" * 60)
        print("Performance Analysis for Different Input Sizes")
        print("=" * 60)
        
        for batch_size, num_points, channels in input_sizes:
            print(f"\nInput shape: [{batch_size}, {num_points}, {channels}]")
            
            # Create random input
            input_tensor = torch.randn(batch_size, num_points, channels).to(device)
            
            try:
                # Calculate FLOPs using thop if available
                if THOP_AVAILABLE:
                    model_copy = PCN(config)
                    if device.type == 'cpu':
                        model_copy.folding_seed = model_copy.folding_seed.cpu()
                    model_copy = model_copy.to(device)
                    
                    flops, params = profile(model_copy, inputs=(input_tensor,), verbose=False)
                    flops_formatted, params_formatted = clever_format([flops, params], "%.3f")
                    
                    print(f"  FLOPs: {flops_formatted}")
                    print(f"  Parameters: {params_formatted}")
                    print(f"  TFLOPs: {flops / 1e12:.6f}")
                else:
                    print(f"  FLOPs: Not available (install thop)")
                    print(f"  TFLOPs: Not available (install thop)")
                
                # Test forward pass
                with torch.no_grad():
                    output = model(input_tensor)
                    coarse_output, fine_output = output
                    
                print(f"  Output shapes:")
                print(f"    Coarse: {list(coarse_output.shape)}")
                print(f"    Fine: {list(fine_output.shape)}")
                
                # Measure inference time
                if torch.cuda.is_available():
                    avg_time = measure_inference_time(model, input_tensor, num_runs=50)
                    print(f"  Average inference time: {avg_time*1000:.2f} ms")
                    print(f"  Throughput: {batch_size/avg_time:.2f} samples/sec")
                
            except Exception as e:
                print(f"  Error during testing: {str(e)}")
                continue
        
        print("\n" + "=" * 60)
        print("Model Architecture Summary")
        print("=" * 60)
        
        # Print model architecture
        print("\nEncoder:")
        print("  First Conv Block: 3 -> 128 -> 256")
        print("  Second Conv Block: 512 -> 512 -> 1024")
        print("  Max pooling for global features")
        
        print("\nDecoder:")
        print("  MLP: 1024 -> 1024 -> 1024 -> (3 * coarse_points)")
        print("  Folding operation with 4x4 grid")
        print("  Final Conv Block: 1029 -> 512 -> 512 -> 3")
        print("  Residual connection for fine points")
        
        print(f"\nPoint Cloud Completion:")
        print(f"  Input points: Variable (tested with {num_points})")
        print(f"  Coarse output: {config.num_pred // 16} points")
        print(f"  Fine output: {config.num_pred} points")
        
        # Memory estimation
        print(f"\nMemory Estimation (batch_size=1, input_points=2048):")
        input_mem = 1 * 2048 * 3 * 4 / (1024**2)  # MB
        model_mem = total_params * 4 / (1024**2)  # MB
        print(f"  Input tensor: {input_mem:.2f} MB")
        print(f"  Model parameters: {model_mem:.2f} MB")
        print(f"  Estimated total GPU memory: ~{(input_mem + model_mem) * 3:.2f} MB")
        
        print("\n" + "=" * 60)
        print("Testing completed successfully!")
        print("=" * 60)
    
    # Run the test
    try:
        test_pcn_model()
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        print("Make sure you have the required dependencies:")
        print("  pip install torch torchvision")
        print("  pip install thop  # for FLOP calculation")