import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class MultiModalDataset(torch.utils.data.Dataset):
    def __init__(self, cnn_feats, vae_feats, labels):
        self.cnn_feats = torch.FloatTensor(cnn_feats)
        self.vae_feats = torch.FloatTensor(vae_feats)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.cnn_feats[idx], self.vae_feats[idx], self.labels[idx]


def load_and_prepare_data(model_config, vae_features_path, test_size=0.2, random_state=42):
    print(f"Loading data for {model_config['display_name']}")
    
    cnn_df = pd.read_csv(model_config['data'])
    vae_df_all = pd.read_csv(vae_features_path)
    
    print(f"- CNN data shape: {cnn_df.shape}")
    print(f"- VAE data shape: {vae_df_all.shape}")
    
    vae_df_all['img_id'] = vae_df_all['img_id'].str.split('.').str[0]
    cnn_df['img_id'] = cnn_df['img_id'].str.split('.').str[0]
    
    cnn_cols = [c for c in cnn_df.columns if c.startswith('f_')]
    vae_cols = [c for c in vae_df_all.columns if c.startswith('f_')]
    
    unique_ids = cnn_df['img_id'].unique()
    unique_labels = cnn_df.groupby('img_id')['label'].first().values
    
    train_ids, test_ids = train_test_split(
        unique_ids,
        test_size=test_size,
        stratify=unique_labels,
        random_state=random_state
    )
    
    print(f"- Train samples: {len(train_ids)}")
    print(f"- Test samples: {len(test_ids)}")
    
    cnn_train_df = cnn_df[cnn_df['img_id'].isin(train_ids)].copy()
    cnn_test_df = cnn_df[cnn_df['img_id'].isin(test_ids)].copy()
    
    vae_rename = {c: f"{c}_vae" for c in vae_cols}
    vae_df_all.rename(columns=vae_rename, inplace=True)
    vae_cols_renamed = [f"{c}_vae" for c in vae_cols]
    
    vae_train_df = vae_df_all[vae_df_all['img_id'].isin(train_ids)].copy()
    vae_test_df = vae_df_all[vae_df_all['img_id'].isin(test_ids)].copy()
    
    train_merged = pd.merge(
        cnn_train_df[['img_id'] + cnn_cols + ['label']],
        vae_train_df[['img_id'] + vae_cols_renamed],
        on='img_id',
        how='inner'
    )
    
    test_merged = pd.merge(
        cnn_test_df[['img_id'] + cnn_cols + ['label']],
        vae_test_df[['img_id'] + vae_cols_renamed],
        on='img_id',
        how='inner'
    )
    
    print(f"- Train merged shape: {train_merged.shape}")
    print(f"- Test merged shape: {test_merged.shape}")
    
    cnn_train_features = train_merged[cnn_cols].values
    cnn_test_features = test_merged[cnn_cols].values
    vae_train_features = train_merged[vae_cols_renamed].values
    vae_test_features = test_merged[vae_cols_renamed].values
    train_labels = train_merged['label'].values
    test_labels = test_merged['label'].values
    
    cnn_scaler = StandardScaler()
    cnn_train_features = cnn_scaler.fit_transform(cnn_train_features)
    cnn_test_features = cnn_scaler.transform(cnn_test_features)
    
    vae_scaler = StandardScaler()
    vae_train_features = vae_scaler.fit_transform(vae_train_features)
    vae_test_features = vae_scaler.transform(vae_test_features)
    
    print(f"- Training samples: {cnn_train_features.shape[0]}")
    print(f"- Test samples: {cnn_test_features.shape[0]}")
    
    return {
        'cnn_train': cnn_train_features,
        'vae_train': vae_train_features,
        'labels_train': train_labels,
        'cnn_test': cnn_test_features,
        'vae_test': vae_test_features,
        'labels_test': test_labels,
        'cnn_dim': len(cnn_cols),
        'vae_dim': len(vae_cols),
        'train_ids': train_merged['img_id'].values,
        'test_ids': test_merged['img_id'].values,
        'cnn_scaler': cnn_scaler,
        'vae_scaler': vae_scaler
    }