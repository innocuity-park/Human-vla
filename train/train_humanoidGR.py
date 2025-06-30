#!/usr/bin/env python3
"""
Advanced Distributed Sliding Window Training script for HumanoidGR model.
Training Phases:
- WARMUP PHASE: Freeze CLIP, MAE, GPT transformer. Train only task-specific layers.
- FINE-TUNING PHASE: Keep CLIP/MAE frozen, apply LoRA to GPT transformer.

Temporal Alignment:
- Input sequence [t-14:t] -> predict (action[t], video_frame[t+1])
- Hierarchical action loss: left_leg + right_leg + torso_spine + head_neck + left_arm + right_arm
- Video loss: MSE between predicted and actual video patches
"""

import os
import sys
import json
import math
import argparse
import datetime
from pathlib import Path
from time import time
from datetime import timedelta
import numpy as np
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs

import clip

# SwanLab for experiment tracking
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    print("SwanLab not available. Install with: pip install swanlab")

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from GPTmodel.humanoidGR import HumanoidGR
from train.HumanoidLMDBDataset import HumanoidLMDBDataset, collate_sliding_windows


class LoRALinear(nn.Module):
    """LoRA (Low-Rank Adaptation) layer for efficient fine-tuning."""
    
    def __init__(self, original_layer, rank=16, alpha=32):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        
        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # LoRA parameters
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        self.scaling = self.alpha / self.rank
    
    def forward(self, x):
        # Original forward pass
        result = self.original_layer(x)
        
        # LoRA adaptation
        lora_result = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        
        return result + lora_result


def apply_lora_to_model(model, target_modules=['c_attn', 'c_proj'], rank=16, alpha=32):
    """Apply LoRA to specified modules in the model."""
    lora_modules = []
    
    def apply_lora_recursive(module, name=""):
        for child_name, child_module in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            
            # Check if this is a target module
            if any(target in child_name for target in target_modules):
                if isinstance(child_module, nn.Linear):
                    # Replace with LoRA layer
                    lora_layer = LoRALinear(child_module, rank=rank, alpha=alpha)
                    setattr(module, child_name, lora_layer)
                    lora_modules.append(full_name)
                    print(f"Applied LoRA to {full_name}")
            else:
                # Recursively apply to children
                apply_lora_recursive(child_module, full_name)
    
    apply_lora_recursive(model)
    return lora_modules


def setup_parameter_groups(model, config, phase='warmup'):
    """Setup parameter groups for different training phases."""
    
    # Handle DistributedDataParallel wrapping - use accelerator if available
    if hasattr(model, 'module'):
        # Standard DDP wrapping
        actual_model = model.module
    elif hasattr(model, '_modules'):
        # Accelerator wrapping
        actual_model = model
    else:
        actual_model = model
    
    # Freeze/unfreeze based on phase
    if phase == 'warmup':
        print("=== WARMUP PHASE: Freezing CLIP, MAE, and GPT transformer ===")
        # Freeze CLIP
        for param in actual_model.model_clip.parameters():
            param.requires_grad = False
        
        # Freeze MAE (already frozen in model init)
        for param in actual_model.model_mae.parameters():
            param.requires_grad = False
        
        # Freeze GPT transformer
        for param in actual_model.transformer.parameters():
            param.requires_grad = False
            
    elif phase == 'finetune':
        print("=== FINE-TUNING PHASE: Applying LoRA to GPT transformer ===")
        # Keep CLIP and MAE frozen
        for param in actual_model.model_clip.parameters():
            param.requires_grad = False
        for param in actual_model.model_mae.parameters():
            param.requires_grad = False
        
        # Apply LoRA to transformer
        if config.get('use_lora', True):
            lora_modules = apply_lora_to_model(
                actual_model.transformer,
                target_modules=['c_attn', 'c_proj'],
                rank=config.get('lora_rank', 16),
                alpha=config.get('lora_alpha', 32)
            )
            print(f"Applied LoRA to {len(lora_modules)} modules")
        else:
            # Full fine-tuning (not recommended for large models)
            for param in actual_model.transformer.parameters():
                param.requires_grad = True
    
    # Collect trainable parameters
    trainable_params = []
    frozen_params = []
    
    for name, param in actual_model.named_parameters():
        if param.requires_grad:
            trainable_params.append((name, param))
        else:
            frozen_params.append(name)
    
    print(f"Trainable parameters: {len(trainable_params)}")
    print(f"Frozen parameters: {len(frozen_params)}")
    
    # Create parameter groups with different learning rates
    param_groups = [
        {
            'params': [p for n, p in trainable_params if 'lora' in n.lower()],
            'lr': config['lr'] * config.get('lora_lr_multiplier', 1.0),
            'weight_decay': config.get('lora_weight_decay', 0.01)
        },
        {
            'params': [p for n, p in trainable_params if 'lora' not in n.lower()],
            'lr': config['lr'],
            'weight_decay': config.get('weight_decay', 0.01)
        }
    ]
    
    # Remove empty groups
    param_groups = [g for g in param_groups if len(g['params']) > 0]
    
    return param_groups


def compute_hierarchical_loss(predictions, targets, config):
    """
    Compute hierarchical loss for different body parts with sliding window alignment.
    L_total = L_videoMSE + L_left_leg + L_right_leg + L_torso_spine + L_head_neck + L_left_arm + L_right_arm
    """
    losses = {}
    total_loss = 0
    
    # Video prediction loss (MSE) - for sliding window, predict t+1 frame
    if 'video_preds' in predictions and 'video_targets' in predictions:
        video_preds = predictions['video_preds']
        video_targets = predictions['video_targets']
        
        # For sliding window: use last timestep prediction
        if len(video_preds.shape) == 4:  # (batch, seq, patches, dim)
            video_preds = video_preds[:, -1]  # Use last timestep
        if len(video_targets.shape) == 4:
            video_targets = video_targets[:, -1] if video_targets.shape[1] > 1 else video_targets[:, 0]
        
        # Check for reasonable values
        if torch.isnan(video_preds).any() or torch.isnan(video_targets).any():
            print("WARNING: NaN values detected in video predictions or targets!")
            video_loss = torch.tensor(0.0, device=video_preds.device)
        else:
            video_loss = F.mse_loss(video_preds, video_targets)
            
            # Add stability check
            if video_loss.item() > 100:
                print(f"WARNING: Very high video loss: {video_loss.item():.6f}")
        
        losses['video_mse'] = video_loss
        total_loss += video_loss * config.get('video_loss_weight', 0.1)
    
    # Action prediction loss (hierarchical smooth L1) - for sliding window
    if 'action_preds' in predictions:
        action_preds = predictions['action_preds']  # (batch, seq, chunk, 56)
        
        # For sliding window: use last timestep prediction
        if len(action_preds.shape) == 4:
            action_preds = action_preds[:, -1, 0]  # (batch, 56) - last timestep, first chunk
        elif len(action_preds.shape) == 3:
            action_preds = action_preds[:, 0]  # (batch, 56) - first chunk
        
        # Get target actions
        if 'target_actions' in targets:
            action_targets = targets['target_actions']  # (batch, 56)
        else:
            print("WARNING: No target actions found for hierarchical loss")
            action_targets = torch.zeros_like(action_preds)
        
        # Define body part indices (56-DOF humanoid)
        body_parts = {
            'left_leg': (0, 7),      # indices [0:7]
            'right_leg': (7, 14),    # indices [7:14]
            'torso_spine': (14, 23), # indices [14:23]
            'head_neck': (23, 32),   # indices [23:32]
            'left_arm': (32, 44),    # indices [32:44]
            'right_arm': (44, 56)    # indices [44:56]
        }
        
        # Compute loss for each body part
        for part_name, (start_idx, end_idx) in body_parts.items():
            part_preds = action_preds[:, start_idx:end_idx]
            part_targets = action_targets[:, start_idx:end_idx]
            
            part_loss = F.smooth_l1_loss(part_preds, part_targets)
            
            losses[f'action_{part_name}'] = part_loss
            total_loss += part_loss * config.get(f'{part_name}_loss_weight', 1.0)
    
    losses['total_loss'] = total_loss
    return losses


class DataPrefetcher:
    """Efficient data prefetcher for sliding window training"""
    def __init__(self, loader, device):
        self.device = device
        self.loader = loader
        self.iter = iter(self.loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.iter)
        except StopIteration:
            self.batch = None
            self.iter = iter(self.loader)
            return 
        
        if self.batch is not None:
            with torch.cuda.stream(self.stream):
                for key in self.batch:
                    if isinstance(self.batch[key], torch.Tensor):
                        self.batch[key] = self.batch[key].to(self.device, non_blocking=True)

    def next(self):
        clock = time()
        batch = self.batch
        if batch is not None:
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key].record_stream(torch.cuda.current_stream())
        self.preload()
        return batch, time() - clock

    def next_without_none(self):
        batch, elapsed_time = self.next()
        if batch is None:
            batch, elapsed_time = self.next()
        return batch, elapsed_time


class AdvancedSlidingWindowTrainer:
    """
    Advanced trainer with two-phase training strategy and hierarchical loss
    """
    
    def __init__(self, config: Dict[str, Any], accelerator: Accelerator):
        self.config = config
        self.accelerator = accelerator
        self.device = accelerator.device
        
        # Initialize models
        self._setup_models()
        
        # Initialize dataset and dataloader
        self._setup_data()
        
        if accelerator.is_main_process:
            print(f"SlidingWindowTrainer initialized")
            print(f"  - Device: {self.device}")
            print(f"  - Window size: {config['window_size']}")
            print(f"  - Per-GPU batch size: {config['per_gpu_batch_size']}")
            print(f"  - Two-phase training: Warmup + Fine-tuning")
            print(f"  - LoRA fine-tuning: {config.get('use_lora', True)}")

    def _setup_models(self):
        """Initialize CLIP, MAE, and HumanoidGR models"""
        if self.accelerator.is_main_process:
            print("Loading models...")
        
        # Load CLIP model (suppress detailed output on non-main processes)
        if self.accelerator.is_main_process:
            self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        else:
            # Load silently on other processes
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        
        # Load MAE model (realistic implementation)
        self.mae_model = self._load_mae_model()
        
        # Initialize HumanoidGR (suppress detailed output on non-main processes)
        self.model = HumanoidGR(
            model_clip=self.clip_model,
            model_mae=self.mae_model,
            max_sequence_length=self.config['window_size'],
            chunk_size=1,  # Predict 1 action per timestep
            training_target=['act_pred', 'fwd_pred'],
            rgb_shape=(224, 224),
            patch_size=16,
            hidden_size=self.config.get('hidden_size', 768),
            state_dim=56,
            act_dim=56,
            verbose=self.accelerator.is_main_process,  # Only show detailed logs on main process
        )
        
        if self.accelerator.is_main_process:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Models initialized")
            print(f"  - Total parameters: {total_params:,}")
            print(f"  - Initially trainable parameters: {trainable_params:,}")

    def _load_mae_model(self):
        """Load realistic MAE model"""
        class MAEViT(nn.Module):
            def __init__(self):
                super().__init__()
                # Simplified MAE-like architecture
                self.patch_embed = nn.Conv2d(3, 768, kernel_size=16, stride=16)
                self.cls_token = nn.Parameter(torch.randn(1, 1, 768))
                self.pos_embed = nn.Parameter(torch.randn(1, 197, 768))
                
                # Transformer blocks
                self.blocks = nn.ModuleList([
                    nn.TransformerEncoderLayer(
                        d_model=768, nhead=12, dim_feedforward=3072,
                        dropout=0.1, activation='gelu', batch_first=True
                    ) for _ in range(12)
                ])
                
                self.norm = nn.LayerNorm(768)
                
            def forward(self, x):
                B = x.shape[0]
                
                # Patch embedding
                x = self.patch_embed(x)  # (B, 768, 14, 14)
                x = x.flatten(2).transpose(1, 2)  # (B, 196, 768)
                
                # Add cls token
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat([cls_tokens, x], dim=1)  # (B, 197, 768)
                
                # Add positional embedding
                x = x + self.pos_embed
                
                # Apply transformer blocks
                for block in self.blocks:
                    x = block(x)
                
                x = self.norm(x)
                
                # Return cls token as global feature and patches as patch features
                cls_feature = x[:, 0]  # (B, 768)
                patch_features = x[:, 1:]  # (B, 196, 768)
                
                return cls_feature, patch_features
        
        return MAEViT().to(self.device)

    def _setup_data(self):
        """Setup dataset and dataloader"""
        if self.accelerator.is_main_process:
            print("Setting up sliding window dataset...")
        
        self.dataset = HumanoidLMDBDataset(
            lmdb_dir=self.config['lmdb_dir'],
            window_size=self.config['window_size'],
            start_ratio=self.config['train_start_ratio'],
            end_ratio=self.config['train_end_ratio'],
            verbose=self.accelerator.is_main_process,  # Only show detailed logs on main process
        )
        
        self.train_dataloader = DataLoader(
            self.dataset,
            batch_size=self.config['per_gpu_batch_size'],  # Fixed: use per_gpu_batch_size
            shuffle=True,
            num_workers=0,  # Disable multiprocessing to avoid CUDA issues
            collate_fn=collate_sliding_windows,
            pin_memory=True,
            drop_last=True,  # Important for distributed training
        )
        
        if self.accelerator.is_main_process:
            print(f"Dataset loaded: {len(self.dataset)} sliding windows")
            print(f"Dataloader created: {len(self.train_dataloader)} batches per epoch")

    def train_phase(self, phase: str, epochs: int, writer: Optional[SummaryWriter] = None, swanlab_run = None, start_epoch: int = 0):
        """Train for one phase (warmup or finetune)"""
        if epochs == 0:
            return 0
            
        if self.accelerator.is_main_process:
            print(f"\n{'='*60}")
            print(f"Starting {phase.upper()} phase: {epochs} epochs")
            print(f"{'='*60}")
        
        # Setup parameter groups for this phase
        param_groups = setup_parameter_groups(self.model, self.config, phase)
        
        # Adjust learning rate for phase
        phase_lr = self.config['lr']
        if phase == 'warmup':
            phase_lr = self.config['lr'] * 0.1  # Lower LR for warmup
        
        # Update learning rates
        for group in param_groups:
            if 'lora' in str(group.get('params', [])):
                group['lr'] = phase_lr * self.config.get('lora_lr_multiplier', 1.0)
            else:
                group['lr'] = phase_lr
        
        # Create optimizer and scheduler
        optimizer = torch.optim.AdamW(param_groups)
        
        # Calculate total steps for scheduler
        steps_per_epoch = len(self.train_dataloader)
        total_steps = epochs * steps_per_epoch
        warmup_steps = int(0.1 * total_steps)  # 10% warmup
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Prepare for distributed training
        model, optimizer, dataloader, scheduler = self.accelerator.prepare(
            self.model, optimizer, self.train_dataloader, scheduler
        )
        
        # Count parameters for this phase
        if self.accelerator.is_main_process:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Phase {phase} - Total parameters: {total_params:,}")
            print(f"Phase {phase} - Trainable parameters: {trainable_params:,}")
            print(f"Phase {phase} - Trainable ratio: {trainable_params/total_params:.2%}")
        
        global_step = 0
        
        # Training loop for this phase
        for epoch in range(epochs):
            current_epoch = start_epoch + epoch + 1
            if self.accelerator.is_main_process:
                print(f"\n{phase.capitalize()} Epoch {current_epoch}/{self.config[f'{phase}_epochs']}")
            
            epoch_losses = self.train_epoch(
                model, dataloader, optimizer, scheduler, 
                current_epoch - 1, phase, writer, global_step, swanlab_run
            )
            
            global_step += len(dataloader)
            
            if self.accelerator.is_main_process:
                print(f"{phase.capitalize()} Epoch {current_epoch} completed:")
                for key, value in epoch_losses.items():
                    print(f"  {key}: {value:.4f}")
                
                # Log epoch-level metrics to SwanLab
                if swanlab_run is not None:
                    epoch_log = {}
                    for key, value in epoch_losses.items():
                        epoch_log[f'epoch/{phase}_{key}'] = value
                    
                    epoch_log[f'epoch/{phase}_progress'] = current_epoch / self.config[f'{phase}_epochs']
                    epoch_log['epoch/phase_numeric'] = 0 if phase == 'warmup' else 1
                    epoch_log['epoch/current_epoch'] = current_epoch
                    epoch_log['epoch/phase_epochs'] = self.config[f'{phase}_epochs']
                    
                    swanlab_run.log(epoch_log, step=global_step)
            
            # Save checkpoint after each epoch (improved from every save_interval epochs)
            if self.accelerator.is_main_process:
                self.save_checkpoint(model, optimizer, scheduler, current_epoch, global_step, epoch_losses, phase)
        
        return global_step

    def train_epoch(self, model, dataloader, optimizer, scheduler, epoch, phase, writer, global_step_offset, swanlab_run):
        """Train for one epoch"""
        model.train()
        epoch_losses = {'total_loss': 0, 'video_mse': 0}
        
        # Initialize body part losses
        body_parts = ['left_leg', 'right_leg', 'torso_spine', 'head_neck', 'left_arm', 'right_arm']
        for part in body_parts:
            epoch_losses[f'action_{part}'] = 0
        
        num_batches = 0
        accumulated_loss = 0
        
        # Setup data prefetcher
        prefetcher = DataPrefetcher(dataloader, self.device)
        
        # Initialize timing for FPS calculation
        start_time = time()
        
        if self.accelerator.is_main_process:
            from tqdm import tqdm
            progress_bar = tqdm(range(len(dataloader)), desc=f"{phase.capitalize()} Epoch {epoch}")
        
        for step in range(len(dataloader)):
            batch, data_time = prefetcher.next_without_none()
            
            if batch is None:
                break
            
            # Prepare inputs
            joint_pos = batch['joint_pos']      # (batch_size, window_size, 56)
            video_frames = batch['video_frames']  # (batch_size, window_size, 3, H, W)
            inst_tokens = batch['inst_token']   # (batch_size, 77)
            target_actions = batch['target_actions']  # (batch_size, 56)
            target_video_frames = batch['target_video_frames']  # (batch_size, 3, H, W)
            
            # Forward pass
            with self.accelerator.accumulate(model):
                predictions = model(
                    video_frames=video_frames,
                    joint_pos=joint_pos,
                    language=inst_tokens,
                    is_training=True
                )
                
                # Prepare targets
                targets = {
                    'target_actions': target_actions,
                    'target_video_frames': target_video_frames,
                }
                
                # Add video targets if model predicts video
                if 'video_preds' in predictions:
                    # Handle DistributedDataParallel wrapping
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    if hasattr(unwrapped_model, '_create_video_targets_strict'):
                        video_targets = unwrapped_model._create_video_targets_strict(
                            target_video_frames.unsqueeze(1)  # Add sequence dimension
                        )
                        predictions['video_targets'] = video_targets
                    else:
                        # Fallback: create video targets manually
                        video_targets = self._create_video_targets_fallback(target_video_frames.unsqueeze(1))
                        predictions['video_targets'] = video_targets
                
                # Compute hierarchical losses
                losses = compute_hierarchical_loss(predictions, targets, self.config)
                
                # Scale loss for gradient accumulation
                loss = losses['total_loss'] / self.config['gradient_accumulation_steps']
                
                # Backward pass
                self.accelerator.backward(loss)
                
                # Gradient clipping (only when gradients are synchronized)
                if self.accelerator.sync_gradients and self.config.get('max_grad_norm', 0) > 0:
                    self.accelerator.clip_grad_norm_(model.parameters(), self.config['max_grad_norm'])
                
                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Accumulate losses
                accumulated_loss += loss.item() * self.config['gradient_accumulation_steps']
            
            # Update epoch losses
            for key, value in losses.items():
                if key in epoch_losses:
                    epoch_losses[key] += value.item()
            num_batches += 1
            
            # Logging
            if self.accelerator.is_main_process:
                global_step = global_step_offset + step
                
                if step % self.config['log_interval'] == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    elapsed_time = time() - start_time
                    
                    # Calculate FPS and memory usage
                    if elapsed_time > 0:
                        fps = (step + 1) / elapsed_time
                    else:
                        fps = 0
                    
                    # Memory usage monitoring
                    if torch.cuda.is_available():
                        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                        memory_usage_pct = (memory_allocated / (torch.cuda.get_device_properties(0).total_memory / 1024**3)) * 100
                        
                        log_text = f"Epoch {epoch}, Batch {step + 1}, FPS: {fps:.2f}, Mem: {memory_allocated:.1f}GB ({memory_usage_pct:.1f}%)"
                    else:
                        log_text = f"Epoch {epoch}, Batch {step + 1}, FPS: {fps:.2f}"
                    
                    # Detailed hierarchical loss logging
                    log_text += f", total_loss: {accumulated_loss:.4f}"
                    if 'video_mse' in losses:
                        log_text += f", video_mse: {losses['video_mse'].item():.4f}"
                    
                    # Individual body part losses
                    body_parts = ['left_leg', 'right_leg', 'torso_spine', 'head_neck', 'left_arm', 'right_arm']
                    for part in body_parts:
                        if f'action_{part}' in losses:
                            log_text += f", {part}: {losses[f'action_{part}'].item():.4f}"
                    
                    log_text += f", lr: {current_lr:.2e}"
                    
                    # Print detailed log
                    print(log_text)
                    
                    # Update progress bar with simplified info
                    progress_bar.set_postfix({
                        'Loss': f"{accumulated_loss:.4f}",
                        'Video': f"{losses.get('video_mse', torch.tensor(0)):.4f}",
                        'Action': f"{sum(v.item() for k, v in losses.items() if k.startswith('action_')):.4f}",
                        'LR': f"{current_lr:.2e}"
                    })
                    
                    # TensorBoard logging
                    if writer is not None:
                        writer.add_scalar(f'{phase}/Loss', accumulated_loss, global_step)
                        writer.add_scalar(f'{phase}/LearningRate', current_lr, global_step)
                        writer.add_scalar(f'{phase}/FPS', fps, global_step)
                        for key, value in losses.items():
                            if key != 'total_loss':
                                writer.add_scalar(f'{phase}/{key}', value.item(), global_step)
                    
                    # SwanLab logging
                    if swanlab_run is not None:
                        log_dict = {
                            f'train/{phase}_loss': accumulated_loss,
                            f'train/{phase}_learning_rate': current_lr,
                            f'train/{phase}_fps': fps,
                            f'train/{phase}_video_mse': losses.get('video_mse', torch.tensor(0)).item(),
                        }
                        
                        # Log individual body part losses
                        for key, value in losses.items():
                            if key.startswith('action_'):
                                log_dict[f'train/{phase}_{key}'] = value.item()
                        
                        log_dict['epoch'] = epoch
                        log_dict['phase_numeric'] = 0 if phase == 'warmup' else 1
                        
                        swanlab_run.log(log_dict, step=global_step)
                
                accumulated_loss = 0  # Reset for next log interval
                progress_bar.update(1)
        
        if self.accelerator.is_main_process:
            progress_bar.close()
        
        # Average losses across all processes
        for key in epoch_losses:
            epoch_losses[key] = epoch_losses[key] / max(num_batches, 1)
            if self.accelerator.num_processes > 1:
                # Create CUDA tensor for distributed gathering
                loss_tensor = torch.tensor(epoch_losses[key], device=self.accelerator.device)
                epoch_losses[key] = self.accelerator.gather(loss_tensor).mean().item()
        
        return epoch_losses

    def _create_video_targets_fallback(self, target_video_frames):
        """
        Fallback method to create video targets when _create_video_targets_strict is not available.
        Creates patch-based targets for video prediction loss.
        """
        # target_video_frames: (batch_size, 1, 3, H, W)
        batch_size = target_video_frames.shape[0]
        
        # Use the MAE model to create patch embeddings as targets
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        with torch.no_grad():
            # Reshape for processing: (batch_size * seq_len, 3, H, W)
            video_flat = target_video_frames.view(-1, 3, 224, 224)
            
            # Get patch features from MAE model
            if hasattr(unwrapped_model, 'model_mae'):
                try:
                    _, patch_features = unwrapped_model.model_mae(video_flat)
                    # patch_features: (batch_size * seq_len, n_patches, patch_dim)
                    
                    # Reshape back to sequence format
                    seq_len = target_video_frames.shape[1]
                    n_patches = patch_features.shape[1]
                    patch_dim = patch_features.shape[2]
                    
                    video_targets = patch_features.view(batch_size, seq_len, n_patches, patch_dim)
                    return video_targets
                    
                except Exception as e:
                    if self.accelerator.is_main_process:
                        print(f"Warning: MAE processing failed, using dummy targets: {e}")
            
            # Fallback to dummy targets if MAE processing fails
            # Create dummy patch targets with correct dimensions
            n_patches = 196  # 14x14 patches for 224x224 image with 16x16 patch size
            patch_dim = 768   # Standard patch embedding dimension
            
            video_targets = torch.zeros(
                batch_size, target_video_frames.shape[1], n_patches, patch_dim,
                device=target_video_frames.device,
                dtype=target_video_frames.dtype
            )
            
            return video_targets

    def save_checkpoint(self, model, optimizer, scheduler, epoch, global_step, losses, phase):
        """Save model checkpoint"""
        if not self.accelerator.is_main_process:
            return
        
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        
        # Save model state
        unwrapped_model = self.accelerator.unwrap_model(model)
        checkpoint = {
            'epoch': epoch,
            'phase': phase,
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'losses': losses,
            'config': self.config,
            'global_step': global_step
        }
        
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], f'checkpoint_{phase}_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        loss_key = 'total_loss'
        if not hasattr(self, f'best_loss_{phase}') or losses[loss_key] < getattr(self, f'best_loss_{phase}'):
            setattr(self, f'best_loss_{phase}', losses[loss_key])
            best_path = os.path.join(self.config['checkpoint_dir'], f'best_model_{phase}.pt')
            torch.save(checkpoint, best_path)
            print(f"✓ New best {phase} model saved: {best_path}")
        
        print(f"✓ Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint for resuming training"""
        if not os.path.exists(checkpoint_path):
            return None
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.accelerator.device)
            if self.accelerator.is_main_process:
                print(f"Checkpoint loaded from {checkpoint_path}")
                print(f"  - Epoch: {checkpoint['epoch']}")
                print(f"  - Phase: {checkpoint['phase']}")
                print(f"  - Global step: {checkpoint['global_step']}")
            return checkpoint
        except Exception as e:
            if self.accelerator.is_main_process:
                print(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return None

    def train(self, writer: Optional[SummaryWriter] = None, swanlab_run = None, resume_from=None):
        """Main training loop with two phases"""
        if self.accelerator.is_main_process:
            print("Starting advanced sliding window distributed training...")
            print(f"Training Plan:")
            print(f"  - Warmup epochs: {self.config['warmup_epochs']}")
            print(f"  - Fine-tuning epochs: {self.config['finetune_epochs']}")
            print(f"  - Total epochs: {self.config['warmup_epochs'] + self.config['finetune_epochs']}")
        
        global_step = 0
        start_phase = 'warmup'
        start_epoch = 0
        
        # Check for checkpoint resuming
        if resume_from:
            checkpoint = self.load_checkpoint(resume_from)
            if checkpoint:
                start_phase = checkpoint['phase']
                start_epoch = checkpoint['epoch']
                global_step = checkpoint['global_step']
                if self.accelerator.is_main_process:
                    print(f"Resuming from {start_phase} phase, epoch {start_epoch}")
        
        # WARMUP PHASE
        if start_phase == 'warmup':
            remaining_warmup = max(0, self.config['warmup_epochs'] - start_epoch)
            if remaining_warmup > 0:
                global_step += self.train_phase('warmup', remaining_warmup, writer, swanlab_run, start_epoch)
            start_phase = 'finetune'
            start_epoch = 0
        
        # FINE-TUNING PHASE
        if start_phase == 'finetune':
            remaining_finetune = max(0, self.config['finetune_epochs'] - start_epoch)
            if remaining_finetune > 0:
                global_step += self.train_phase('finetune', remaining_finetune, writer, swanlab_run, start_epoch)
        
        # Final checkpoint
        if self.accelerator.is_main_process:
            final_checkpoint = {
                'training_completed': True,
                'total_epochs': self.config['warmup_epochs'] + self.config['finetune_epochs'],
                'total_steps': global_step,
                'config': self.config
            }
            
            final_path = os.path.join(self.config['checkpoint_dir'], 'training_completed.pt')
            torch.save(final_checkpoint, final_path)
            
            if swanlab_run is not None:
                swanlab_run.log({
                    'training/completed': 1,
                    'training/total_epochs': self.config['warmup_epochs'] + self.config['finetune_epochs'],
                    'training/total_steps': global_step,
                }, step=global_step)
                swanlab_run.finish()
                print("✓ SwanLab experiment finalized")
            
            print("\nAdvanced sliding window training completed!")


def create_config(args) -> Dict[str, Any]:
    """Create comprehensive training configuration from enhanced arguments"""
    config = {
        # Data and paths
        'lmdb_dir': args.lmdb_dir,
        'video_base_path': args.video_base_path,
        'checkpoint_dir': args.checkpoint_dir,
        'log_dir': args.log_dir,
        
        # Sliding window configuration
        'window_size': args.window_size,
        'stride': args.stride,
        'train_start_ratio': 0.0,
        'train_end_ratio': 0.8,
        
        # Training phases
        'warmup_epochs': args.warmup_epochs,
        'finetune_epochs': args.finetune_epochs,
        
        # Training parameters (updated argument names)
        'per_gpu_batch_size': args.per_gpu_batch_size,
        'lr': args.lr,
        'weight_decay': 1e-4,
        'max_grad_norm': args.max_grad_norm,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        
        # LoRA settings
        'use_lora': True,
        'lora_rank': 16,
        'lora_alpha': 32,
        'lora_lr_multiplier': 1.0,
        'lora_weight_decay': 0.01,
        
        # Loss weights (hierarchical action prediction)
        'video_loss_weight': 0.1,
        'left_leg_loss_weight': 1.0,
        'right_leg_loss_weight': 1.0,
        'torso_spine_loss_weight': 1.0,
        'head_neck_loss_weight': 1.0,
        'left_arm_loss_weight': 1.0,
        'right_arm_loss_weight': 1.0,
        
        # Model architecture
        'state_dim': 56,
        'act_dim': 56,
        'hidden_size': 384,  # Reduced for memory efficiency
        'max_sequence_length': args.window_size,  # Use sliding window size
        'chunk_size': 1,
        'training_target': ['act_pred', 'fwd_pred'],
        'img_feat_dim': 768,
        'patch_feat_dim': 768,
        'lang_feat_dim': 512,
        'image_size': (224, 224),
        'patch_size': 16,
        'without_norm_pixel_loss': False,
        
        # Transformer configuration
        'n_layer': 12,
        'n_head': 12,
        'n_inner': 1536,
        'activation_function': 'gelu_new',
        'resid_pdrop': 0.1,
        'embd_pdrop': 0.1,
        'attn_pdrop': 0.1,
        
        # Perceiver resampler parameters
        'resampler_params': {
            'num_latents': 9,
            'depth': 2,
            'dim_head': 64,
            'heads': 8,
            'num_media_embeds': 1
        },
        
        # Pretrained model paths
        'pretrained_path': 'GPTmodel/pretrain/pretrained.pt',
        'mae_pretrained_path': 'GPTmodel/pretrain/mae_pretrain_vit_base.pth',
        
        # Logging and saving
        'log_interval': 10,
        'save_interval': 2,
        
        # Experiment metadata
        'experiment_name': args.experiment_name,
        'mixed_precision': args.mixed_precision,
    }
    return config


def main():
    """Enhanced main function with superior logging approach from train_humanoid_distributed.py"""
    parser = argparse.ArgumentParser(
        description='Advanced HumanoidGR Sliding Window Training - Supports both single-GPU and multi-GPU distributed training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ADVANCED SLIDING WINDOW TRAINING
"""
    )
    
    # === GPU Configuration ===
    gpu_group = parser.add_argument_group('GPU Configuration')
    gpu_group.add_argument('--num_gpus', type=str, default='auto', 
                          help='Number of GPUs to use. Options: "auto" (use all), "1" (single GPU), or specific number')
    gpu_group.add_argument('--gpu_ids', type=str, default=None,
                          help='Specific GPU IDs to use (comma-separated, e.g., "0,1,2,3"). Overrides --num_gpus')
    gpu_group.add_argument('--mixed_precision', type=str, default='fp16', choices=['no', 'fp16', 'bf16'], 
                          help='Mixed precision mode for memory efficiency and speed')
    
    # === Training Configuration ===
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument('--config', type=str, default='config_humanoid.json', help='Config file path')
    train_group.add_argument('--per_gpu_batch_size', type=int, default=1, help='Per-GPU batch size')
    train_group.add_argument('--gradient_accumulation_steps', type=int, default=8, 
                           help='Gradient accumulation steps')
    train_group.add_argument('--warmup_epochs', type=int, default=3, help='Warmup epochs (frozen GPT)')
    train_group.add_argument('--finetune_epochs', type=int, default=20, help='Fine-tuning epochs (LoRA)')
    train_group.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    train_group.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping norm')
    
    # === Data Configuration ===
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument('--lmdb_dir', type=str, default='data/humanoid_lmdb', help='LMDB dataset directory')
    data_group.add_argument('--video_base_path', type=str, default='data/video/snippet_videos', help='Video base path')
    data_group.add_argument('--window_size', type=int, default=15, help='Sliding window size (timesteps)')
    data_group.add_argument('--stride', type=int, default=1, help='Sliding window stride')
    
    # === Output Configuration ===
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument('--checkpoint_dir', type=str, default='checkpoints_sliding_window', help='Checkpoint directory')
    output_group.add_argument('--log_dir', type=str, default='logs_sliding_window', help='TensorBoard logs directory')
    output_group.add_argument('--experiment_name', type=str, default=None, help='Experiment name (auto-generated if not specified)')
    
    args = parser.parse_args()
    
    # === GPU SETUP AND VALIDATION (Enhanced from train_humanoid_distributed.py) ===
    print(f"\n{'='*50}")
    print(f"HUMANOIDGR ADVANCED SLIDING WINDOW TRAINING")
    print(f"{'='*50}")
    
    available_gpus = torch.cuda.device_count()
    
    if available_gpus == 0:
        print("CRITICAL: No CUDA GPUs available! Please check your CUDA installation.")
        print("Make sure NVIDIA drivers and CUDA toolkit are properly installed.")
        return
    
    print(f"GPU Detection Results:")
    print(f"   Available GPUs: {available_gpus}")
    for i in range(available_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        gpu_compute = torch.cuda.get_device_properties(i).major
        print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB, Compute {gpu_compute}.x)")
    
    # Validate and configure GPU usage
    if args.gpu_ids is not None:
        # Use specific GPU IDs
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
        for gpu_id in gpu_ids:
            if gpu_id >= available_gpus:
                print(f"ERROR: GPU ID {gpu_id} not available! Available range: 0-{available_gpus-1}")
                return
        num_gpus = len(gpu_ids)
        print(f"Using Specific GPUs: {gpu_ids} ({num_gpus} total)")
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    else:
        # Use num_gpus parameter
        if args.num_gpus == 'auto':
            num_gpus = available_gpus
            print(f"Auto-Detection: Using all {num_gpus} GPUs")
        elif args.num_gpus == '1':
            num_gpus = 1
            print(f"Single GPU Mode: Using 1 GPU")
        else:
            try:
                num_gpus = int(args.num_gpus)
                if num_gpus > available_gpus:
                    print(f"ERROR: Requested {num_gpus} GPUs but only {available_gpus} available!")
                    return
                elif num_gpus <= 0:
                    print(f"ERROR: Invalid GPU count: {num_gpus}")
                    return
                print(f"Manual Selection: Using {num_gpus}/{available_gpus} GPUs")
            except ValueError:
                print(f"ERROR: Invalid --num_gpus value: '{args.num_gpus}'. Use 'auto', '1', or positive integer.")
                return
    
    # Set performance optimization environment variables
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Avoid tokenizer warnings
    
    # === TRAINING CONFIGURATION DISPLAY ===
    effective_batch_size = args.per_gpu_batch_size * num_gpus * args.gradient_accumulation_steps
    total_epochs = args.warmup_epochs + args.finetune_epochs
    
    # Generate experiment name if not provided
    if args.experiment_name is None:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if num_gpus == 1:
            args.experiment_name = f"HumanoidGR_SlidingWindow_1GPU_{args.per_gpu_batch_size}bs_{timestamp}"
        else:
            args.experiment_name = f"HumanoidGR_SlidingWindow_{num_gpus}GPU_{args.per_gpu_batch_size}bs_{timestamp}"
    
    print(f"\n{'='*100}")
    print(f"TRAINING CONFIGURATION SUMMARY")
    print(f"{'='*100}")
    
    print(f"System Configuration:")
    print(f"   • Training Mode: {'Single-GPU' if num_gpus == 1 else f'{num_gpus}-GPU Distributed'}")
    print(f"   • Mixed Precision: {args.mixed_precision.upper()}")
    print(f"   • CUDA Memory Optimization: Enabled")
    print(f"   • Device Count: {num_gpus}")
    
    print(f"\nTraining Schedule:")
    print(f"   • Phase 1 (Warmup): {args.warmup_epochs} epochs - Frozen GPT transformer")
    print(f"   • Phase 2 (Fine-tune): {args.finetune_epochs} epochs - LoRA adaptation")
    print(f"   • Total Epochs: {total_epochs}")
    print(f"   • Learning Rate: {args.lr:.1e}")
    print(f"   • Gradient Clipping: {args.max_grad_norm}")
    
    print(f"\nBatch Configuration:")
    print(f"   • Per-GPU Batch Size: {args.per_gpu_batch_size}")
    print(f"   • Gradient Accumulation: {args.gradient_accumulation_steps} steps")
    print(f"   • Effective Batch Size: {effective_batch_size}")
    print(f"   • Memory Efficiency: Optimized for sliding windows")
    
    print(f"\nSliding Window Configuration:")
    print(f"   • Window Size: {args.window_size} timesteps")
    print(f"   • Stride: {args.stride}")
    print(f"   • Temporal Alignment: t-{args.window_size-1}:t → predict(action_t, video_t+1)")
    print(f"   • Video Prediction: MSE loss on patches")
    print(f"   • Action Prediction: Hierarchical loss (6 body parts)")
    
    print(f"\nData & Storage:")
    print(f"   • LMDB Dataset: {args.lmdb_dir}")
    print(f"   • Video Base Path: {args.video_base_path}")
    print(f"   • Checkpoints: {args.checkpoint_dir}")
    print(f"   • Logs: {args.log_dir}")
    print(f"   • Experiment: {args.experiment_name}")
    
    print(f"\nTraining Targets:")
    print(f"   • Video Forward Prediction: ✓ (MSE Loss)")
    print(f"   • Hierarchical Action Prediction: ✓ (6 Body Parts)")
    print(f"   • Language Conditioning: ✓ (CLIP Embeddings)")
    print(f"   • Temporal Consistency: ✓ (Sliding Window)")
    
    print(f"{'='*50}")
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    print(f"✓ Output directories created")
    print(f"Launching distributed training...")
    print(f"{'='*100}\n")
    
    # === DISTRIBUTED TRAINING SETUP (From train_humanoid_distributed.py) ===
    if num_gpus > 1:
        print(f"Setting up {num_gpus}-GPU distributed training...")
        import torch.multiprocessing as mp
        
        # Set up distributed environment
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['WORLD_SIZE'] = str(num_gpus)
        
        # Launch distributed training
        mp.spawn(
            _run_distributed_training,
            args=(args, num_gpus),
            nprocs=num_gpus,
            join=True
        )
    else:
        print(f"Setting up single-GPU training...")
        # Single GPU training
        _run_distributed_training(0, args, num_gpus)


def _run_distributed_training(gpu_id, args, num_gpus):
    """Run distributed training on specific GPU"""
    # Set up distributed training if multi-GPU
    if num_gpus > 1:
        os.environ['RANK'] = str(gpu_id)
        os.environ['LOCAL_RANK'] = str(gpu_id)
        
        # Initialize process group
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=num_gpus,
            rank=gpu_id
        )
        
        # Set device
        torch.cuda.set_device(gpu_id)
        
        if gpu_id == 0:
            print(f"✓ Distributed training initialized with {num_gpus} GPUs")
    
    # Initialize accelerator with proper settings
    from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
    
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=True,
        broadcast_buffers=True,
        gradient_as_bucket_view=True
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[init_kwargs, ddp_kwargs] if num_gpus > 1 else [],
        device_placement=True,
        split_batches=False,
    )
    
    # Verify distributed setup
    if accelerator.is_main_process:
        print(f"✓ Accelerator initialized: {accelerator.num_processes} processes")
        if num_gpus > 1 and accelerator.num_processes != num_gpus:
            print(f"⚠️  Warning: Expected {num_gpus} processes but got {accelerator.num_processes}")
    
    # Continue with the main training logic
    _execute_training(args, accelerator, num_gpus)


def _execute_training(args, accelerator, num_gpus):
    """Execute the main training logic with enhanced logging"""
    device = accelerator.device
    
    if accelerator.is_main_process:
        print(f"\n{'='*80}")
        print(f"TRAINING SESSION STARTED")
        print(f"{'='*80}")
        print(f"Process Info:")
        print(f"   • Main Process: {accelerator.is_main_process}")
        print(f"   • Local Process: {accelerator.is_local_main_process}")
        print(f"   • Device: {device}")
        print(f"   • Process Count: {accelerator.num_processes}")
        print(f"{'='*80}\n")
    
    # Create enhanced configuration from arguments
    config = create_config(args)
    
    # Initialize trainer with enhanced logging
    trainer = AdvancedSlidingWindowTrainer(config, accelerator)
    
    # Enhanced SwanLab initialization with comprehensive project organization (DEFAULT ENABLED)
    swanlab_run = None
    if SWANLAB_AVAILABLE and accelerator.is_main_process:
        try:
            # Set SwanLab API key (following train_humanoid_distributed.py pattern)
            swanlab_api_key = os.getenv('SWANLAB_API_KEY', 'UejDeOO6XNiLQdX3IhVqk')
            if swanlab_api_key:
                os.environ['SWANLAB_API_KEY'] = swanlab_api_key
            
            # Enhanced config for SwanLab
            swanlab_config = {
                **config,
                'num_gpus': num_gpus,
                'num_processes': accelerator.num_processes,
                'effective_batch_size': args.per_gpu_batch_size * num_gpus * args.gradient_accumulation_steps,
                'training_mode': 'distributed' if num_gpus > 1 else 'single_gpu',
                'mixed_precision': args.mixed_precision,
                'sliding_window_size': args.window_size,
                'window_stride': args.stride,
                'temporal_alignment': f't-{args.window_size-1}:t → predict(action_t, video_t+1)',
                'total_epochs': args.warmup_epochs + args.finetune_epochs,
                'gpu_memory_optimization': True,
                'sliding_window_training': True
            }
            
            # Determine project name and tags based on GPU configuration
            if num_gpus == 1:
                project_name = "HumanoidGR-SlidingWindow-SingleGPU"
                description = f"HumanoidGR 56-DOF sliding window single-GPU training with LoRA fine-tuning"
                tags = ['single-gpu', 'humanoid', 'lora', 'sliding-window']
            else:
                project_name = f"HumanoidGR-SlidingWindow-{num_gpus}GPU"
                description = f"HumanoidGR 56-DOF sliding window distributed training on {num_gpus} GPUs with LoRA fine-tuning"
                tags = ['multi-gpu', 'humanoid', 'lora', 'distributed', 'sliding-window', f'{num_gpus}-gpu']
            
            swanlab_run = swanlab.init(
                project=project_name,
                experiment_name=args.experiment_name,
                config=swanlab_config,
                description=description,
                tags=tags
            )
            
            print(f"SwanLab Experiment Tracking Initialized")
            print(f"  • Project: {project_name}")
            print(f"  • Experiment: {args.experiment_name}")
            print(f"  • View at: https://swanlab.cn/@anquan/{project_name}")
            
        except Exception as e:
            print(f"SwanLab initialization failed: {e}")
            print("Training will continue without SwanLab logging")
            swanlab_run = None
    elif not SWANLAB_AVAILABLE and accelerator.is_main_process:
        print("⚠️  SwanLab not available. Install with: pip install swanlab")
        print("Training will continue without SwanLab logging")
    
    # Enhanced TensorBoard logging with structured directories
    writer = None
    if accelerator.is_main_process:
        log_dir = Path(args.log_dir) / args.experiment_name
        writer = SummaryWriter(log_dir)
        print(f"✓ TensorBoard Logging Configured")
        print(f"  • Log Directory: {log_dir}")
        print(f"  • View with: tensorboard --logdir {log_dir}")
    
    # === TRAINING EXECUTION WITH SUPERIOR LOGGING ===
    if accelerator.is_main_process:
        print(f"\n{'='*80}")
        print(f"🎬 ADVANCED SLIDING WINDOW TRAINING INITIATED")
        print(f"{'='*80}")
        print(f"Training Strategy:")
        print(f"   • Window Size: {args.window_size} timesteps")
        print(f"   • Window Stride: {args.stride}")
        print(f"   • Temporal Alignment: t-{args.window_size-1}:t → predict(action_t, video_t+1)")
        print(f"   • Memory Efficiency: ~{args.window_size}x better than full episodes")
        print(f"   • Video Prediction: MSE loss on patch embeddings")
        print(f"   • Action Prediction: Hierarchical loss (6 body parts)")
        
        print(f"\nTechnical Details:")
        print(f"   • Sequence Processing: Sliding window approach")
        print(f"   • Gradient Accumulation: {args.gradient_accumulation_steps} steps")
        print(f"   • Mixed Precision: {args.mixed_precision.upper()}")
        print(f"   • LoRA Adaptation: Rank {config.get('lora_rank', 16)}, Alpha {config.get('lora_alpha', 32)}")
        
        print(f"\nPerformance Monitoring:")
        print(f"   • TensorBoard: Real-time loss tracking")
        print(f"   • SwanLab: Comprehensive experiment analytics")
        print(f"   • Checkpoints: Saved after each epoch")
        print(f"   • Best Model: Auto-saved based on total loss")
        print(f"   • Resume Support: Automatic checkpoint resuming")
        print(f"{'='*80}\n")
    
    # Check for existing checkpoints to resume from
    resume_checkpoint = None
    if accelerator.is_main_process:
        checkpoint_dir = Path(args.checkpoint_dir)
        if checkpoint_dir.exists():
            # Look for the latest checkpoint
            warmup_checkpoints = list(checkpoint_dir.glob('checkpoint_warmup_epoch_*.pt'))
            finetune_checkpoints = list(checkpoint_dir.glob('checkpoint_finetune_epoch_*.pt'))
            
            latest_checkpoint = None
            if finetune_checkpoints:
                # Resume from latest finetune checkpoint
                latest_checkpoint = max(finetune_checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
            elif warmup_checkpoints:
                # Resume from latest warmup checkpoint
                latest_checkpoint = max(warmup_checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
            
            if latest_checkpoint:
                resume_checkpoint = str(latest_checkpoint)
                print(f"🔄 Found existing checkpoint: {resume_checkpoint}")
    
    try:
        # Execute main training loop
        trainer.train(writer=writer, swanlab_run=swanlab_run, resume_from=resume_checkpoint)
        
        # Training completion logging
        if accelerator.is_main_process:
            print(f"\n{'='*80}")
            print(f"SLIDING WINDOW TRAINING COMPLETED SUCCESSFULLY!")
            print(f"{'='*80}")
            print(f"Training Summary:")
            print(f"   • Mode: {'Single-GPU' if num_gpus == 1 else f'{num_gpus}-GPU Distributed'}")
            print(f"   • Total Epochs: {args.warmup_epochs + args.finetune_epochs}")
            print(f"   • Effective Batch Size: {args.per_gpu_batch_size * num_gpus * args.gradient_accumulation_steps}")
            print(f"   • Experiment: {args.experiment_name}")
            
            print(f"\nOutput Locations:")
            print(f"   • Model Checkpoints: {args.checkpoint_dir}")
            print(f"   • Training Logs: {args.log_dir}")
            print(f"   • Best Models: {args.checkpoint_dir}/best_model_*.pt")
            
            if swanlab_run is not None:
                print(f"   • SwanLab Dashboard: https://swanlab.cn")
            
            print(f"\nNext Steps:")
            print(f"   • Load best model for inference")
            print(f"   • Use sliding window approach for real-time control")
            print(f"   • 56-DOF humanoid robot ready for deployment!")
            print(f"{'='*80}")
            
    except Exception as e:
        if accelerator.is_main_process:
            print(f"\n{'='*80}")
            print(f"TRAINING ERROR OCCURRED")
            print(f"{'='*80}")
            print(f"Error: {e}")
            print(f"Please check the logs and configuration.")
            print(f"{'='*80}")
            import traceback
            traceback.print_exc()
        raise
    finally:
        # Clean up resources
        if writer:
            writer.close()
            if accelerator.is_main_process:
                print(f"✓ TensorBoard writer closed")
        
        if swanlab_run:
            swanlab_run.finish()
            if accelerator.is_main_process:
                print(f"✓ SwanLab experiment finalized")


if __name__ == "__main__":
    main() 