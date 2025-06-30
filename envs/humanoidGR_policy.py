"""
HumanoidGR Policy Wrapper for SB3-style Evaluation

This module provides a policy wrapper that interfaces the trained HumanoidGR model
with stable baselines 3 evaluation system for online evaluation in DMControl.

FIXED: Added LoRA compatibility for loading fine-tuned checkpoints.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import sys
import os
import clip

# Add project paths
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "GPTmodel"))

from GPTmodel.humanoidGR import HumanoidGR


class LoRALinear(nn.Module):
    """LoRA (Low-Rank Adaptation) layer for loading fine-tuned models."""
    
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
            else:
                # Recursively apply to children
                apply_lora_recursive(child_module, full_name)
    
    apply_lora_recursive(model)
    return lora_modules


def merge_lora_weights(state_dict, verbose=True):
    """
    Merge LoRA weights back into original linear layer weights.
    
    This converts a LoRA checkpoint back to a standard checkpoint that can be loaded
    without LoRA layers.
    """
    merged_state_dict = {}
    lora_keys_to_remove = []
    
    # Find all LoRA parameter keys
    lora_pairs = {}
    
    for key in state_dict.keys():
        if '.lora_A' in key:
            base_key = key.replace('.lora_A', '')
            if base_key not in lora_pairs:
                lora_pairs[base_key] = {}
            lora_pairs[base_key]['lora_A'] = key
            lora_keys_to_remove.append(key)
        elif '.lora_B' in key:
            base_key = key.replace('.lora_B', '')
            if base_key not in lora_pairs:
                lora_pairs[base_key] = {}
            lora_pairs[base_key]['lora_B'] = key
            lora_keys_to_remove.append(key)
    
    # Copy all non-LoRA parameters
    for key, value in state_dict.items():
        if key not in lora_keys_to_remove:
            merged_state_dict[key] = value
    
    # Merge LoRA weights into original weights
    for base_key, lora_keys in lora_pairs.items():
        if 'lora_A' in lora_keys and 'lora_B' in lora_keys:
            # Get original weight key
            original_weight_key = f"{base_key}.original_layer.weight"
            
            if original_weight_key in state_dict:
                # Get LoRA parameters
                lora_A = state_dict[lora_keys['lora_A']]  # (rank, in_features)
                lora_B = state_dict[lora_keys['lora_B']]  # (out_features, rank)
                original_weight = state_dict[original_weight_key]  # (out_features, in_features)
                
                # Default LoRA scaling (alpha=32, rank=16 -> scaling=2.0)
                alpha = 32  # Default from training
                rank = lora_A.shape[0]
                scaling = alpha / rank
                
                # Compute LoRA delta: B @ A * scaling
                lora_delta = (lora_B @ lora_A) * scaling  # (out_features, in_features)
                
                # Merge into original weight
                merged_weight = original_weight + lora_delta
                
                # Store with standard weight key (remove .original_layer)
                standard_weight_key = f"{base_key}.weight"
                merged_state_dict[standard_weight_key] = merged_weight
                
                if verbose:
                    print(f"  ✓ Merged LoRA weights for {base_key}")
                
                # Remove the original_layer weight since we've merged it
                if original_weight_key in merged_state_dict:
                    del merged_state_dict[original_weight_key]
    
    if verbose and lora_pairs:
        print(f"Merged {len(lora_pairs)} LoRA layer pairs into standard weights")
    
    return merged_state_dict


class HumanoidGRPolicy:
    """
    HumanoidGR Policy Wrapper for Online Evaluation
    
    FIXED: Now supports loading LoRA fine-tuned checkpoints with automatic weight merging.
    
    Interfaces the trained HumanoidGR model with DMControl environment for:
    - Sliding window sequence management
    - Vision-language-action prediction
    - Real-time humanoid robot control
    - SB3-compatible predict() interface
    - LoRA checkpoint compatibility
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        window_size: int = 15,
        action_dim: int = 56,
        state_dim: int = 56,
        deterministic: bool = True,
        action_scale: float = 1.0,
        load_method: str = 'auto',  # 'auto', 'merge_lora', 'apply_lora'
        verbose: bool = True
    ):
        """
        Initialize HumanoidGR policy
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device for inference ('cuda' or 'cpu')
            window_size: Sliding window size for sequences
            action_dim: Action space dimension (56 DOF)
            state_dim: State space dimension (56 DOF)
            deterministic: Whether to use deterministic predictions
            action_scale: Scaling factor for actions
            load_method: How to handle LoRA checkpoints ('auto', 'merge_lora', 'apply_lora')
            verbose: Whether to print detailed logs
        """
        
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.window_size = window_size
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.deterministic = deterministic
        self.action_scale = action_scale
        self.load_method = load_method
        self.verbose = verbose
        
        # Initialize models
        self._setup_models()
        
        # Load trained checkpoint (with LoRA support)
        self._load_checkpoint()
        
        # Initialize sequence buffers (sliding windows)
        self.reset_sequences()
        
        if verbose:
            print(f"✓ HumanoidGR Policy initialized")
            print(f"  - Checkpoint: {checkpoint_path}")
            print(f"  - Device: {self.device}")
            print(f"  - Window size: {window_size}")
            print(f"  - Action/State dim: {action_dim}/{state_dim}")
            print(f"  - Deterministic: {deterministic}")
            print(f"  - Load method: {load_method}")

    def _setup_models(self):
        """Setup CLIP, MAE, and HumanoidGR models"""
        if self.verbose:
            print("Loading models for policy...")
        
        # Load CLIP model
        try:
            self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
            if self.verbose:
                print("✓ CLIP model loaded")
        except Exception as e:
            print(f"Warning: Failed to load CLIP model: {e}")
            self.clip_model = None
        
        # Detect model configuration from checkpoint
        transformer_hidden_size, mae_hidden_size, lora_config = self._detect_model_config_from_checkpoint()
        
        # Load MAE model (adjusted for detected MAE hidden size)
        self.mae_model = self._create_mae_model(mae_hidden_size)
        if self.verbose:
            print(f"✓ MAE model created (hidden_size={mae_hidden_size})")
        
        # Initialize HumanoidGR model with detected configuration
        self.model = HumanoidGR(
            model_clip=self.clip_model,
            model_mae=self.mae_model,
            max_sequence_length=self.window_size,
            chunk_size=1,
            training_target=['act_pred', 'fwd_pred'],
            rgb_shape=(224, 224),
            patch_size=16,
            hidden_size=transformer_hidden_size,  # Use transformer hidden size for the main model
            state_dim=self.state_dim,
            act_dim=self.action_dim,
            pretrained_path=None,  # Skip GR1 pretrained loading for policy
            mae_pretrained_path=None,  # Skip MAE pretrained loading for policy
        ).to(self.device)
        
        # Apply LoRA if needed
        self.lora_applied = False
        if lora_config['has_lora'] and self.load_method in ['auto', 'apply_lora']:
            if self.verbose:
                print(f"Applying LoRA structure for checkpoint compatibility...")
            
            lora_modules = apply_lora_to_model(
                self.model.transformer,
                target_modules=['c_attn', 'c_proj'],
                rank=lora_config.get('rank', 16),
                alpha=lora_config.get('alpha', 32)
            )
            self.lora_applied = True
            
            if self.verbose:
                print(f"✓ Applied LoRA to {len(lora_modules)} modules")
        
        # Set to evaluation mode
        self.model.eval()
        
        if self.verbose:
            print(f"✓ HumanoidGR model initialized (transformer_hidden_size={transformer_hidden_size}, mae_hidden_size={mae_hidden_size})")

    def _detect_model_config_from_checkpoint(self):
        """Detect model configuration and LoRA status from checkpoint"""
        lora_config = {'has_lora': False, 'rank': 16, 'alpha': 32}
        
        try:
            if not os.path.exists(self.checkpoint_path):
                if self.verbose:
                    print(f"  Checkpoint not found, using default hidden_size=768")
                return 768, 768, lora_config
            
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Check for LoRA parameters
            lora_keys = [k for k in state_dict.keys() if '.lora_A' in k or '.lora_B' in k]
            if lora_keys:
                lora_config['has_lora'] = True
                if self.verbose:
                    print(f" Detected LoRA checkpoint with {len(lora_keys)} LoRA parameters")
                
                # Try to detect LoRA rank from first lora_A parameter
                for key in state_dict.keys():
                    if '.lora_A' in key:
                        lora_config['rank'] = state_dict[key].shape[0]
                        break
            
            # Detect transformer and MAE hidden sizes separately
            transformer_hidden_size = None
            mae_hidden_size = None
            
            # Check transformer dimensions
            for key in state_dict.keys():
                if 'transformer.h.0.ln_1.weight' in key:
                    transformer_hidden_size = state_dict[key].shape[0]
                    if self.verbose:
                        print(f"✓ Detected transformer hidden_size={transformer_hidden_size}")
                    break
                elif 'transformer.h.0.attn.c_proj.weight' in key:
                    transformer_hidden_size = state_dict[key].shape[0]
                    if self.verbose:
                        print(f"✓ Detected transformer hidden_size={transformer_hidden_size}")
                    break
                elif 'transformer.h.0.attn.c_proj.original_layer.weight' in key:
                    transformer_hidden_size = state_dict[key].shape[0]
                    if self.verbose:
                        print(f"✓ Detected transformer hidden_size={transformer_hidden_size} (LoRA)")
                    break
            
            # Check MAE dimensions
            for key in state_dict.keys():
                if 'model_mae.cls_token' in key:
                    mae_hidden_size = state_dict[key].shape[2]
                    if self.verbose:
                        print(f"✓ Detected MAE hidden_size={mae_hidden_size}")
                    break
                elif 'model_mae.patch_embed.weight' in key:
                    mae_hidden_size = state_dict[key].shape[0]
                    if self.verbose:
                        print(f"✓ Detected MAE hidden_size={mae_hidden_size}")
                    break
                elif 'model_mae.pos_embed' in key:
                    mae_hidden_size = state_dict[key].shape[2]
                    if self.verbose:
                        print(f"✓ Detected MAE hidden_size={mae_hidden_size}")
                    break
            
            # Use detected sizes or fallback to defaults
            if transformer_hidden_size is None:
                transformer_hidden_size = 768
                if self.verbose:
                    print(f"  Could not detect transformer hidden_size, using default=768")
            
            if mae_hidden_size is None:
                mae_hidden_size = 768
                if self.verbose:
                    print(f"  Could not detect MAE hidden_size, using default=768")
            
            # Check for mismatched dimensions
            if transformer_hidden_size != mae_hidden_size:
                if self.verbose:
                    print(f"  Mismatched hidden sizes: transformer={transformer_hidden_size}, MAE={mae_hidden_size}")
                    print(f"   Using transformer={transformer_hidden_size}, MAE={mae_hidden_size}")
            
            return transformer_hidden_size, mae_hidden_size, lora_config
            
        except Exception as e:
            if self.verbose:
                print(f"  Error detecting model config: {e}, using default hidden_size=768")
            return 768, 768, lora_config

    def _create_mae_model(self, hidden_size=768):
        """Create simplified MAE model for inference"""
        import torch.nn as nn
        
        class InferenceMAE(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.hidden_size = hidden_size
                
                # Calculate number of heads based on hidden_size
                if hidden_size % 64 == 0:
                    nhead = hidden_size // 64  # 64 dim per head
                elif hidden_size % 32 == 0:
                    nhead = hidden_size // 32  # 32 dim per head
                else:
                    nhead = max(1, hidden_size // 64)  # Fallback
                
                self.patch_embed = nn.Conv2d(3, hidden_size, kernel_size=16, stride=16)
                self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
                self.pos_embed = nn.Parameter(torch.randn(1, 197, hidden_size))
                
                # Transformer blocks (simplified)
                self.blocks = nn.ModuleList([
                    nn.TransformerEncoderLayer(
                        d_model=hidden_size, 
                        nhead=nhead, 
                        dim_feedforward=hidden_size * 4,
                        dropout=0.0, activation='gelu', batch_first=True
                    ) for _ in range(6)  # Reduced layers for inference
                ])
                
                self.norm = nn.LayerNorm(hidden_size)
                
            def forward(self, x):
                B = x.shape[0]
                
                # Patch embedding
                x = self.patch_embed(x)  # (B, hidden_size, 14, 14)
                x = x.flatten(2).transpose(1, 2)  # (B, 196, hidden_size)
                
                # Add cls token
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat([cls_tokens, x], dim=1)  # (B, 197, hidden_size)
                
                # Add positional embedding
                x = x + self.pos_embed
                
                # Apply transformer blocks
                for block in self.blocks:
                    x = block(x)
                
                x = self.norm(x)
                
                # Return cls token as global feature and patches as patch features
                cls_feature = x[:, 0]  # (B, hidden_size)
                patch_features = x[:, 1:]  # (B, 196, hidden_size)
                
                return cls_feature, patch_features
        
        return InferenceMAE(hidden_size).to(self.device)

    def _load_checkpoint(self):
        """Load trained model checkpoint with LoRA support"""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        try:
            if self.verbose:
                print(f"Loading checkpoint: {self.checkpoint_path}")
            
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Check if this is a LoRA checkpoint
            lora_keys = [k for k in state_dict.keys() if '.lora_A' in k or '.lora_B' in k]
            has_lora = len(lora_keys) > 0
            
            if self.verbose and has_lora:
                print(f" Detected LoRA checkpoint with {len(lora_keys)} LoRA parameters")
            
            # Handle LoRA loading based on method
            if has_lora and self.load_method == 'merge_lora':
                if self.verbose:
                    print(f"Merging LoRA weights into standard weights...")
                state_dict = merge_lora_weights(state_dict, verbose=self.verbose)
            elif has_lora and not self.lora_applied and self.load_method == 'auto':
                if self.verbose:
                    print(f"Auto-merging LoRA weights (no LoRA structure applied)...")
                state_dict = merge_lora_weights(state_dict, verbose=self.verbose)
            
            # Load model weights
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            
            # Filter out expected missing/unexpected keys for better reporting
            filtered_missing = []
            filtered_unexpected = []
            
            for key in missing_keys:
                # Skip expected missing keys (from CLIP/MAE frozen models)
                if not any(skip in key for skip in ['model_clip.', 'model_mae.', 'text_encoder.']):
                    filtered_missing.append(key)
            
            for key in unexpected_keys:
                # Skip LoRA keys if we merged them
                if not (has_lora and ('.lora_A' in key or '.lora_B' in key or '.original_layer.' in key)):
                    filtered_unexpected.append(key)
            
            if self.verbose:
                print(f"✓ Checkpoint loaded successfully")
                if filtered_missing:
                    print(f"    Missing keys: {len(filtered_missing)}")
                    if len(filtered_missing) <= 5:
                        for key in filtered_missing:
                            print(f"    - {key}")
                    else:
                        print(f"    - {filtered_missing[0]} ... and {len(filtered_missing)-1} more")
                
                if filtered_unexpected:
                    print(f"    Unexpected keys: {len(filtered_unexpected)}")
                    if len(filtered_unexpected) <= 5:
                        for key in filtered_unexpected:
                            print(f"    - {key}")
                    else:
                        print(f"    - {filtered_unexpected[0]} ... and {len(filtered_unexpected)-1} more")
                
                if has_lora:
                    if self.lora_applied:
                        print(f"  LoRA checkpoint loaded with LoRA structure")
                    else:
                        print(f"  LoRA weights merged into standard model")
                
                # Print checkpoint info if available
                if 'epoch' in checkpoint:
                    print(f"   Epoch: {checkpoint['epoch']}")
                if 'phase' in checkpoint:
                    print(f"   Phase: {checkpoint['phase']}")
                if 'losses' in checkpoint:
                    losses = checkpoint['losses']
                    print(f"   Total loss: {losses.get('total_loss', 'N/A'):.4f}")
                    if 'video_mse' in losses:
                        print(f"   Video MSE: {losses['video_mse']:.4f}")
                
                # Warn if too many missing/unexpected keys
                if len(filtered_missing) > 10 or len(filtered_unexpected) > 10:
                    print(f"    WARNING: High number of missing/unexpected keys may indicate incompatibility")
        
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")

    def reset_sequences(self):
        """Reset sliding window sequences for new episode"""
        self.video_sequence = []
        self.joint_pos_sequence = []
        self.current_instruction = None
        self.instruction_tokens = None
        
        if self.verbose:
            print(" Sequences reset for new episode")

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Any] = None,
        episode_start: bool = False,
        deterministic: Optional[bool] = None
    ) -> Tuple[np.ndarray, Optional[Any]]:
        """
        Predict action using HumanoidGR model (SB3-compatible interface)
        
        Args:
            observation: Environment observation (dict with video_sequence, joint_pos_sequence, etc.)
            state: RNN state (not used by HumanoidGR)
            episode_start: Whether this is the start of a new episode
            deterministic: Whether to use deterministic prediction
            
        Returns:
            action: Predicted action (56-DOF joint positions)
            state: Updated state (None for HumanoidGR)
        """
        
        if episode_start:
            self.reset_sequences()
        
        # Use deterministic if specified, otherwise use instance setting
        use_deterministic = deterministic if deterministic is not None else self.deterministic
        
        try:
            # Extract observations
            if isinstance(observation, dict):
                # Get sequences from observation
                video_sequence = observation.get('video_sequence')  # (window_size, 3, H, W)
                joint_pos_sequence = observation.get('joint_pos_sequence')  # (window_size, 56)
                language_tokens = observation.get('language_tokens')  # (77,)
                
                # Convert to torch tensors
                if video_sequence is not None:
                    video_frames = torch.from_numpy(video_sequence).float().unsqueeze(0).to(self.device)  # (1, window_size, 3, H, W)
                else:
                    # Create dummy video if not available
                    video_frames = torch.zeros(1, self.window_size, 3, 224, 224, device=self.device)
                
                if joint_pos_sequence is not None:
                    joint_pos = torch.from_numpy(joint_pos_sequence).float().unsqueeze(0).to(self.device)  # (1, window_size, 56)
                else:
                    # Create dummy joint positions if not available
                    joint_pos = torch.zeros(1, self.window_size, self.state_dim, device=self.device)
                
                if language_tokens is not None:
                    language = torch.from_numpy(language_tokens).long().unsqueeze(0).to(self.device)  # (1, 77)
                else:
                    # Create dummy language tokens
                    language = torch.zeros(1, 77, dtype=torch.long, device=self.device)
            
            else:
                # Handle flat observation (fallback)
                print("Warning: Using fallback observation processing")
                video_frames = torch.zeros(1, self.window_size, 3, 224, 224, device=self.device)
                joint_pos = torch.zeros(1, self.window_size, self.state_dim, device=self.device)
                language = torch.zeros(1, 77, dtype=torch.long, device=self.device)
            
            # Forward pass through HumanoidGR
            with torch.no_grad():
                predictions = self.model(
                    video_frames=video_frames,
                    joint_pos=joint_pos,
                    language=language,
                    is_training=False
                )
                
                # Extract action prediction
                if 'action_preds' in predictions:
                    action_preds = predictions['action_preds']  # (1, window_size, chunk_size, action_dim)
                    
                    # Get action for last timestep, first chunk
                    if len(action_preds.shape) == 4:
                        action = action_preds[0, -1, 0].cpu().numpy()  # (action_dim,)
                    elif len(action_preds.shape) == 3:
                        action = action_preds[0, 0].cpu().numpy()  # (action_dim,)
                    else:
                        action = action_preds[0].cpu().numpy()  # (action_dim,)
                
                else:
                    # Fallback: return zero action
                    print("Warning: No action predictions found, using zero action")
                    action = np.zeros(self.action_dim, dtype=np.float32)
            
            # Apply action scaling and clipping
            action = action * self.action_scale
            action = np.clip(action, -1.0, 1.0)  # Clip to valid action range
            
            # Ensure correct shape
            if len(action.shape) == 0:
                action = np.array([action])
            elif len(action) != self.action_dim:
                # Pad or truncate to correct dimension
                if len(action) < self.action_dim:
                    padded_action = np.zeros(self.action_dim, dtype=np.float32)
                    padded_action[:len(action)] = action
                    action = padded_action
                else:
                    action = action[:self.action_dim]
            
            return action.astype(np.float32), None
        
        except Exception as e:
            print(f"Error in HumanoidGR prediction: {e}")
            # Return safe zero action on error
            return np.zeros(self.action_dim, dtype=np.float32), None

    def set_instruction(self, instruction: str):
        """Set natural language instruction for the policy"""
        self.current_instruction = instruction
        
        # Tokenize instruction with CLIP
        if self.clip_model is not None:
            try:
                self.instruction_tokens = clip.tokenize([instruction], truncate=True)[0]
                if self.verbose:
                    print(f" Instruction set: '{instruction}'")
            except Exception as e:
                print(f"Warning: Failed to tokenize instruction: {e}")
                self.instruction_tokens = None
        else:
            self.instruction_tokens = None

    def get_current_instruction(self) -> Optional[str]:
        """Get current instruction"""
        return self.current_instruction

    def save_rollout_video(self, frames: List[np.ndarray], save_path: str):
        """Save rollout video frames"""
        try:
            import imageio
            
            # Convert frames to uint8 if needed
            video_frames = []
            for frame in frames:
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                video_frames.append(frame)
            
            # Save as MP4
            imageio.mimsave(save_path, video_frames, fps=30)
            print(f"✓ Rollout video saved: {save_path}")
            
        except Exception as e:
            print(f"Warning: Failed to save video: {e}")

    def __call__(self, observation, **kwargs):
        """Make policy callable (alternative interface)"""
        action, _ = self.predict(observation, **kwargs)
        return action 