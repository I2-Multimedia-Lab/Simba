import torch
import torch.nn as nn
from timm.models.layers import DropPath

# --- Helper functions (from your original code) ---
def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def knn_point(nsample, xyz, new_xyz):
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx

def query_knn_point(nsample, xyz, new_xyz):
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx

def grouping_operation(features, idx):
    B, C, N = features.shape
    _, N_query, K = idx.shape
    idx_base = torch.arange(0, B, device=features.device).view(-1, 1, 1) * N
    features_flat = features.transpose(1, 2).reshape(B * N, C)
    idx_flat = (idx + idx_base).view(-1)
    grouped_features = features_flat[idx_flat, :].view(B, N_query, K, C)
    return grouped_features.permute(0, 3, 1, 2).contiguous()

def einsum(equation, *operands):
    return torch.einsum(equation, *operands)

# --- LCM Style Modules (act_layer fixed internally) ---
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.LeakyReLU(negative_slope=0.2) # Fixed activation
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LAL(nn.Module):
    def __init__(self, dim=512, downrate=8, gcn_k=20):
        super(LAL, self).__init__()
        self.k = gcn_k
        self.dim = dim
        self.bn1 = nn.BatchNorm2d(int(self.dim // downrate))
        self.bn2 = nn.BatchNorm1d(self.dim)

        self.conv1 = nn.Sequential(nn.Conv2d(self.dim * 2, int(self.dim // downrate), kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2), # Fixed activation
                                   )
        self.conv2 = nn.Sequential(nn.Conv1d(int(self.dim // downrate), self.dim, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2)) # Fixed activation

    def forward(self, x_features, xyz_coords):
        bs, num_points, num_dims = x_features.shape

        idx = knn_point(self.k, xyz_coords, xyz_coords)
        
        device = x_features.device
        idx_base = torch.arange(0, bs, device=device).view(-1, 1, 1) * num_points
        idx_flat = (idx + idx_base).view(-1)

        x_features_flat = x_features.view(bs * num_points, num_dims)
        neighbor_features = x_features_flat[idx_flat, :].view(bs, num_points, self.k, num_dims)

        query_features_expanded = x_features.unsqueeze(2).repeat(1, 1, self.k, 1)
        fused_features = torch.cat((neighbor_features - query_features_expanded, query_features_expanded), dim=3)
        fused_features = fused_features.permute(0, 3, 1, 2).contiguous() 

        x_processed = self.conv1(fused_features)
        x_processed = x_processed.max(dim=-1, keepdim=False)[0]
        output_features = self.conv2(x_processed)
        
        return output_features.permute(0, 2, 1).contiguous()


class CrossLAL(nn.Module):
    def __init__(self, dim, k_neighbors, downrate=8): # Removed act_layer parameter
        super(CrossLAL, self).__init__()
        self.k = k_neighbors
        self.dim = dim
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.dim * 2, int(self.dim // downrate), kernel_size=1, bias=False),
            nn.BatchNorm2d(int(self.dim // downrate)),
            nn.LeakyReLU(negative_slope=0.2), # Fixed activation
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(int(self.dim // downrate), self.dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.dim),
            nn.LeakyReLU(negative_slope=0.2) # Fixed activation
        )

    def forward(self, query_features, query_xyz, key_value_features, key_value_xyz):
        bs, n_q, c = query_features.shape
        _, n_kv, _ = key_value_features.shape

        idx = knn_point(self.k, key_value_xyz, query_xyz)

        key_value_features_flat = key_value_features.view(bs * n_kv, c)
        idx_base = torch.arange(0, bs, device=query_xyz.device).view(-1, 1, 1) * n_kv
        idx_flat = (idx + idx_base).view(-1)
        gathered_kv_features = key_value_features_flat[idx_flat, :].view(bs, n_q, self.k, c)

        query_features_expanded = query_features.unsqueeze(2).repeat(1, 1, self.k, 1)
        fused_features = torch.cat((gathered_kv_features - query_features_expanded, query_features_expanded), dim=3)
        fused_features = fused_features.permute(0, 3, 1, 2).contiguous() 

        x = self.conv1(fused_features)
        x = x.max(dim=-1, keepdim=False)[0]
        output = self.conv2(x)
        
        return output.permute(0, 2, 1).contiguous()


class BlockLCM_SA(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0., norm_layer=nn.LayerNorm, downrate=8, gcn_k=20): # Removed act_layer parameter
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim / downrate), drop=drop) # Removed act_layer parameter
        self.attn = LAL(dim, downrate=downrate, gcn_k=gcn_k) # Removed act_layer parameter

    def forward(self, x_features, xyz_coords):
        identity = x_features
        x_norm1 = self.norm1(x_features)
        attn_output = self.attn(x_norm1, xyz_coords)
        x_features = identity + self.drop_path(attn_output)
        
        x_features = x_features + self.drop_path(self.mlp(self.norm2(x_features)))
        return x_features

class BlockLCM_CA(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0., norm_layer=nn.LayerNorm, downrate=8, cross_k=20): # Removed act_layer parameter
        super().__init__()
        self.norm_query = norm_layer(dim)
        self.norm_kv = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.cross_attn = CrossLAL(dim=dim, k_neighbors=cross_k, downrate=downrate) # Removed act_layer parameter
        self.norm_ffn = norm_layer(dim)
        self.ffn = Mlp(in_features=dim, hidden_features=int(dim / downrate), drop=drop) # Removed act_layer parameter

    def forward(self, query_features, query_xyz, key_value_features, key_value_xyz):
        identity = query_features
        normalized_query_features = self.norm_query(query_features)
        normalized_key_value_features = self.norm_kv(key_value_features)
        cross_output = self.cross_attn(
            normalized_query_features, query_xyz,
            normalized_key_value_features, key_value_xyz
        )
        x = identity + self.drop_path(cross_output)

        x = x + self.drop_path(self.ffn(self.norm_ffn(x)))
        return x
    
class CrossFormer(nn.Module):  #cross attention/self attention

    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, drop_path=0.1):
        super().__init__()
        self.bn1 = nn.LayerNorm(dim)
        self.bn2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop)#, batch_first=True)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = nn.Identity()
        self.bn3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(dim, out_dim),
        )

    def forward(self, x, y):
        short_cut = x
        x = self.bn1(x)
        y = self.bn2(y)
        x_t = x.transpose(0, 1)
        y_t = y.transpose(0, 1)
        x_attn = self.attn(query=x_t, key=y_t, value=y_t)[0]
        x_attn = x_attn.transpose(0, 1)
        x = short_cut + self.drop_path(x_attn)
        x = x + self.drop_path(self.ffn(self.bn3(x)))
        return x
    
# Wrapper to make BlockLCM_SA match original Transformer's (B, C, N) interface
class TransformerWrapperLCM_SA(nn.Module):
    def __init__(self, in_channel, dim, n_knn, drop_path_rate, downrate): # Removed act_layer parameter
        super().__init__()
        self.linear_start = nn.Conv1d(in_channel, dim, 1)
        self.block_lcm_sa = BlockLCM_SA(
            dim=dim,
            drop_path=drop_path_rate, 
            downrate=downrate,
            gcn_k=n_knn
        )
        self.linear_end = nn.Conv1d(dim, in_channel, 1)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x, pos):
        identity = x
        x = self.linear_start(x)
        
        x_features_transposed = x.transpose(2,1).contiguous()
        pos_transposed = pos.transpose(2,1).contiguous()

        x_out_features = self.block_lcm_sa(x_features_transposed, pos_transposed)
        
        x_out = x_out_features.transpose(2,1).contiguous()
        
        y = self.linear_end(x_out)
        return y + self.drop_path(identity)