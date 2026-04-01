import torch
from torch import nn
import torch.nn.functional as F
import math


class CrossAttentionLayer(nn.Module):
    def __init__(self, dim_query, dim_key_value, num_heads=1, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim_query = dim_query
        self.dim_key_value = dim_key_value
        
        self.W_q = nn.Linear(dim_query, dim_query)
        self.W_k = nn.Linear(dim_key_value, dim_query)
        self.W_v = nn.Linear(dim_key_value, dim_query)
        
        self.out_proj = nn.Linear(dim_query, dim_query)
        self.dropout = nn.Dropout(dropout)
        
        self.norm1 = nn.LayerNorm(dim_query)
        self.norm2 = nn.LayerNorm(dim_query)
    
    def forward(self, query, key_value):
        if len(query.shape) == 2:
            query = query.unsqueeze(1)
        if len(key_value.shape) == 2:
            key_value = key_value.unsqueeze(1)
        
        Q = self.W_q(query)
        K = self.W_k(key_value)
        V = self.W_v(key_value)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dim_query)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attended = torch.matmul(attention_weights, V)
        attended = self.norm1(query + attended)
        
        output = self.out_proj(attended)
        output = self.norm2(attended + output)
        
        if output.shape[1] == 1:
            output = output.squeeze(1)
        
        return output


class BidirectionalCrossAttention(nn.Module):
    def __init__(self, cnn_dim, vae_dim, num_heads=1, dropout=0.1):
        super().__init__()
        
        self.cnn_to_vae = CrossAttentionLayer(
            dim_query=cnn_dim,
            dim_key_value=vae_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.vae_to_cnn = CrossAttentionLayer(
            dim_query=vae_dim,
            dim_key_value=cnn_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.combine = nn.Linear(cnn_dim + vae_dim, cnn_dim + vae_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.norm_cnn = nn.LayerNorm(cnn_dim)
        self.norm_vae = nn.LayerNorm(vae_dim)
    
    def forward(self, cnn_features, vae_features):
        cnn_attended = self.cnn_to_vae(cnn_features, vae_features)
        vae_attended = self.vae_to_cnn(vae_features, cnn_features)
        
        cnn_attended = self.norm_cnn(cnn_features + self.dropout(cnn_attended))
        vae_attended = self.norm_vae(vae_features + self.dropout(vae_attended))
        
        combined = torch.cat([cnn_attended, vae_attended], dim=1)
        combined = self.combine(combined)
        
        return combined


class FeatureReducer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.net(x)


class CombinedMLP(nn.Module):
    def __init__(self, cnn_dim, vae_dim, num_classes, combination_factor=0.8, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        self.cnn_dim_original = cnn_dim
        self.vae_dim = vae_dim
        self.combination_factor = combination_factor
        
        total_features = (vae_dim / (1 - combination_factor))
        self.cnn_reduced_dim = int(total_features - vae_dim)
        self.cnn_reduced_dim = min(self.cnn_reduced_dim, cnn_dim)
        self.combined_dim = self.cnn_reduced_dim + vae_dim
        
        if use_attention:
            self.attention = BidirectionalCrossAttention(
                cnn_dim=self.cnn_reduced_dim,
                vae_dim=vae_dim,
                num_heads=4,
                dropout=0.1
            )
            post_attention_dim = self.cnn_reduced_dim + vae_dim
        else:
            post_attention_dim = self.combined_dim
        
        self.cnn_reducer = nn.Sequential(
            nn.Linear(cnn_dim, self.cnn_reduced_dim),
            nn.BatchNorm1d(self.cnn_reduced_dim),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(post_attention_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, cnn_feats, vae_feats):
        cnn_reduced = self.cnn_reducer(cnn_feats)
        
        if self.use_attention:
            combined = self.attention(cnn_reduced, vae_feats)
        else:
            combined = torch.cat([cnn_reduced, vae_feats], dim=1)
        
        logits = self.classifier(combined)
        return logits, None
    
    def get_dims_info(self):
        return {
            'cnn_original': self.cnn_dim_original,
            'cnn_reduced': self.cnn_reduced_dim,
            'vae_dim': self.vae_dim,
            'combined_dim': self.combined_dim,
            'combination_factor': self.combination_factor,
            'cnn_reduction_ratio': self.cnn_reduced_dim / self.cnn_dim_original,
            'use_attention': self.use_attention
        }