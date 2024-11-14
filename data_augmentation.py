import torch
import numpy as np
import torchvision

def shannon_augment(input_data, act_dim, n=16):
    data_without_action = input_data[:, :-act_dim]
    action = input_data[:, -act_dim:]
    
    # 将data_without_action纵向分为n个块
    chunk_size = data_without_action.shape[1] // n
    chunks = [data_without_action[:, i*chunk_size:(i+1)*chunk_size] for i in range(n)]
    
    # 计算每个块的香农熵并乘以相应的块
    augmented_chunks = []
    for chunk in chunks:
        binned_data = torch.histc(chunk, bins=256, min=chunk.min(), max=chunk.max())
        probs = binned_data / binned_data.sum()
        entropy = -torch.sum(probs * torch.log2(probs + 1e-10))
        augmented_chunks.append(chunk * entropy.item())
    
    augmented_data = torch.cat(augmented_chunks, dim=1)
    return torch.cat([augmented_data, action], dim=-1)

def cutout_augment(input_data, act_dim, obs_dim):
    n = int(np.log2(obs_dim) )
    data_without_action = input_data[:, :-act_dim]
    action = input_data[:, -act_dim:]
    
    num_cols = data_without_action.shape[1]
    cols_to_zero = torch.randperm(num_cols)[:n]
    data_without_action[:, cols_to_zero] = 0
    
    return torch.cat([data_without_action, action], dim=-1)

def gaussian_noise_augment(input_data, act_dim, sigma=0.1):
    data_without_action = input_data[:, :-act_dim]
    action = input_data[:, -act_dim:]
    
    noise = torch.randn_like(data_without_action) * sigma
    augmented_data = data_without_action + noise
    
    return torch.cat([augmented_data, action], dim=-1)

def flip_augment(input_data, act_dim):
    data_without_action = input_data[:, :-act_dim]
    action = input_data[:, -act_dim:]
    
    obs_dim = data_without_action.shape[1] // 2
    obs = data_without_action[:, :obs_dim]
    obs_next = data_without_action[:, obs_dim:]
    
    flipped_obs = torch.flip(obs, dims=[1])
    flipped_obs_next = torch.flip(obs_next, dims=[1])
    
    flipped_data = torch.cat([flipped_obs, flipped_obs_next], dim=1)
    return torch.cat([flipped_data, action], dim=-1)

def scale_augment(input_data, act_dim, scale_range=(0.8, 1.2)):
    data_without_action = input_data[:, :-act_dim]
    action = input_data[:, -act_dim:]
    
    obs_dim = data_without_action.shape[1] // 2
    obs = data_without_action[:, :obs_dim]
    obs_next = data_without_action[:, obs_dim:]
    
    batch_size = obs.shape[0]
    scale_factors = torch.empty(batch_size, 1).uniform_(scale_range[0], scale_range[1]).to(obs.device)
    
    scaled_obs = obs * scale_factors
    scaled_obs_next = obs_next * scale_factors
    
    scaled_data = torch.cat([scaled_obs, scaled_obs_next], dim=1)
    return torch.cat([scaled_data, action], dim=-1)

def translate_augment(input_data, act_dim, translate_range=(0.1, 0.1)):
    data_without_action = input_data[:, :-act_dim]
    action = input_data[:, -act_dim:]
    
    obs_dim = data_without_action.shape[1] // 2
    obs = data_without_action[:, :obs_dim]
    obs_next = data_without_action[:, obs_dim:]
    
    transform = torchvision.transforms.RandomAffine(
        degrees=0,
        translate=translate_range
    )
    
    translated_obs = transform(obs.unsqueeze(0)).squeeze(0)
    translated_obs_next = transform(obs_next.unsqueeze(0)).squeeze(0)
    
    translated_data = torch.cat([translated_obs, translated_obs_next], dim=1)
    return torch.cat([translated_data, action], dim=-1)

def smooth_augment(input_data, act_dim, n=3):
    data_without_action = input_data[:, :-act_dim]
    action = input_data[:, -act_dim:]
    
    smoothed_data = torch.zeros_like(data_without_action)
    for i in range(data_without_action.shape[1]):
        start = max(0, i - n // 2)
        end = min(data_without_action.shape[1], i + n // 2 + 1)
        smoothed_data[:, i] = torch.mean(data_without_action[:, start:end], dim=1)
    
    return torch.cat([smoothed_data, action], dim=-1)
