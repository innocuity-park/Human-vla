# Copyright (2024) Bytedance Ltd. and/or its affiliates
# Adapted for Humanoid Robot Control

import torch
import torch.nn as nn
import transformers
from flamingo_pytorch import PerceiverResampler
from transformers import GPT2Model
import cv2
import numpy as np
import os
from pathlib import Path
from .action_encoder import HierarchicalActionEncoder, HierarchicalActionDecoder
from .vision_transformer import Block
from .transformer_utils import get_2d_sincos_pos_embed

# Import vision transformer components
try:
    from .vision_transformer import Block
    from .transformer_utils import get_2d_sincos_pos_embed
except ImportError:
    try:
        from GPTmodel.vision_transformer import Block
        from GPTmodel.transformer_utils import get_2d_sincos_pos_embed
    except ImportError:
        print("Warning: Vision transformer components not found. Please copy from GR1 models/")
        Block = None
        get_2d_sincos_pos_embed = None


class HumanoidGR(nn.Module):
    """
    HumanoidGR: Vision-Language-Action model for 56-DOF humanoid robots.
    
    Based on GR1 architecture with hierarchical action encoding and proper sequence formatting.
    Sequence format: (l, joint_pos_{t-h}, video_frame_{t-h}, [video], [act], ..., l, joint_pos_t, video_frame_t, [video], [act])
    
    TEMPORAL ALIGNMENT: Input at time t -> Predict at time t+1
    - Video frame at t -> Predict video frame at t+1
    - Joint position at t -> Predict action to reach joint position at t+1
    """
    
    def __init__(
        self,
        model_clip,
        model_mae,
        rgb_shape=(224, 224),
        patch_size=16,
        state_dim=56,
        act_dim=56,
        hidden_size=768,
        max_sequence_length=250,  # Maximum sequence length supported
        chunk_size=1,  # Number of action predictions per timestep
        training_target=['act_pred', 'fwd_pred'],
        img_feat_dim=768,
        patch_feat_dim=768,
        lang_feat_dim=512,
        resampler_params=None,
        without_norm_pixel_loss=False,
        video_base_path="data/video/snippet_videos",
        pretrained_path="GPTmodel/pretrain/pretrained.pt",
        mae_pretrained_path="GPTmodel/pretrain/mae_pretrain_vit_base.pth",
        **kwargs
    ):
        super().__init__()
        
        # Model parameters
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.max_sequence_length = max_sequence_length  # Support variable lengths up to this max
        self.chunk_size = chunk_size
        self.patch_size = patch_size
        self.image_size = rgb_shape
        self.img_feat_dim = img_feat_dim
        self.lang_feat_dim = lang_feat_dim
        self.patch_feat_dim = patch_feat_dim
        self.without_norm_pixel_loss = without_norm_pixel_loss
        self.video_base_path = video_base_path
        
        # Training targets
        self.act_pred = 'act_pred' in training_target
        self.fwd_pred = 'fwd_pred' in training_target
        
        # Resampler parameters
        if resampler_params is None:
            resampler_params = {
                'num_latents': 9,
                'depth': 2,
                'dim_head': 64,
                'heads': 8,
                'num_media_embeds': 1
            }
        self.n_patch_latents = resampler_params['num_latents']
        
        # GPT backbone
        config = transformers.GPT2Config(
            vocab_size=1,
            n_embd=hidden_size,
            n_positions=max_sequence_length * 50,  # Much larger to accommodate tokens per timestep
            **kwargs
        )
        self.transformer = GPT2Model(config)
        
        # Perceiver resampler for patch features
        self.perceiver_resampler = PerceiverResampler(
            dim=patch_feat_dim,
            depth=resampler_params['depth'],
            dim_head=resampler_params['dim_head'],
            heads=resampler_params['heads'],
            num_latents=self.n_patch_latents,
            num_media_embeds=resampler_params['num_media_embeds']
        )
        
        # Frozen pretrained models
        self.model_clip = model_clip
        # Don't freeze CLIP parameters for now to ensure it works
        # for _, param in self.model_clip.named_parameters():
        #     param.requires_grad = False
            
        self.model_mae = model_mae
        for _, param in self.model_mae.named_parameters():
            param.requires_grad = False
        
        # Hierarchical encoder for joint positions
        self.joint_pos_encoder = HierarchicalActionEncoder(final_embedding_size=hidden_size)
        
        # Timestep embedding - support variable lengths up to max
        self.embed_timestep = nn.Embedding(max_sequence_length, hidden_size)
        
        # Language embedding
        self.embed_lang = nn.Linear(lang_feat_dim, hidden_size)
        
        # Image and patch embeddings
        self.embed_img = nn.Linear(img_feat_dim, hidden_size)
        self.embed_patch = nn.Linear(patch_feat_dim, hidden_size)
        
        # Layer normalization
        self.embed_ln = nn.LayerNorm(hidden_size)
        
        # Query tokens - Improved action chunking following GR1 approach
        if self.act_pred:
            # Base action queries (like GR1's action_queries)
            self.action_queries = nn.Embedding(1, hidden_size)
            # Chunk-specific queries (like GR1's action_chunk_queries)
            self.action_chunk_queries = nn.Embedding(chunk_size, hidden_size)
            self.action_chunk_queries.weight.data.fill_(0)  # Initialize to zero for fine-tuning
            
        if self.fwd_pred:
            self.video_queries = nn.Embedding(self.n_patch_latents + 1, hidden_size)
            self.video_queries.weight.data.fill_(0)  # Initialize to zero
        
        # Action decoder (hierarchical)
        if self.act_pred:
            self.action_decoder = HierarchicalActionDecoder(input_embedding_size=hidden_size)
        
        # Video decoder (transformer-based like GR1)
        if self.fwd_pred:
            self.video_decoder_embed = nn.Linear(hidden_size, hidden_size, bias=True)
            self.video_mask_token = nn.Parameter(torch.zeros(1, 1, 1, hidden_size))
            
            # Video decoder blocks
            decoder_depth = 2
            self.video_decoder_blocks = nn.ModuleList([
                Block(hidden_size, 16, 4, qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm)
                for _ in range(decoder_depth)
            ])
            self.video_decoder_norm = nn.LayerNorm(hidden_size)
            self.video_decoder_pred = nn.Linear(hidden_size, patch_size**2 * 3, bias=True)
            
            # Positional embeddings for video decoder
            self.video_decoder_pos_embed = nn.Parameter(
                torch.zeros(1, (self.image_size[0]//patch_size)**2, hidden_size), 
                requires_grad=False
            )
            decoder_pos_embed = get_2d_sincos_pos_embed(
                self.video_decoder_pos_embed.shape[-1], 
                (self.image_size[0]//patch_size)
            )
            self.video_decoder_pos_embed.data.copy_(
                torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
            )
        
        # Load pretrained weights
        self._load_pretrained_weights(pretrained_path, mae_pretrained_path)
        
        print(f"HumanoidGR model initialized")
        print(f"  - State dim: {state_dim}, Action dim: {act_dim}")
        print(f"  - Max sequence length: {max_sequence_length} (supports variable lengths)")
        print(f"  - Chunk size: {chunk_size}")
        print(f"  - Training targets: {training_target}")
        print(f"  - Video prediction: {self.fwd_pred}, Action prediction: {self.act_pred}")
        print(f"  - STRICT MODE: No dummy data, episode-based training")
    
    def _load_pretrained_weights(self, pretrained_path, mae_pretrained_path):
        """Load pretrained weights from GR1 and MAE models with simplified logging."""
        try:
            if os.path.exists(pretrained_path):
                print(f"Loading GR1 pretrained weights from {pretrained_path}")
                checkpoint = torch.load(pretrained_path, map_location='cpu')
                
                # Handle checkpoint structure
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # Get current model state dict
                model_dict = self.state_dict()
                
                # Filter and load compatible weights
                pretrained_dict = {}
                incompatible_count = 0
                skipped_count = 0
                
                # Define incompatible key patterns
                incompatible_patterns = [
                    'embed_arm_state', 'embed_gripper_state', 'embed_state',
                    'pred_arm_act', 'pred_gripper_act', 'pred_act_mlps',
                    'model_clip.', 'text_encoder.', 'model_mae.'
                ]
                
                for key, value in state_dict.items():
                    # Skip explicitly incompatible keys
                    if any(pattern in key for pattern in incompatible_patterns):
                        skipped_count += 1
                        continue
                    
                    # Try to load compatible weights
                    if key in model_dict:
                        if model_dict[key].shape == value.shape:
                            pretrained_dict[key] = value
                        else:
                            incompatible_count += 1
                    else:
                        # Check for partial key matches for decoder components
                        for model_key in model_dict.keys():
                            if (key.endswith(model_key.split('.')[-1]) and 
                                'decoder' in key and 'decoder' in model_key and
                                model_dict[model_key].shape == value.shape):
                                pretrained_dict[model_key] = value
                                break
                
                # Enhanced loading for video decoder components
                decoder_loaded = self._load_video_decoder_weights(state_dict, model_dict, pretrained_dict)
                
                # Update model with compatible weights
                if pretrained_dict:
                    missing_keys, unexpected_keys = self.load_state_dict(pretrained_dict, strict=False)
                    print(f"Successfully loaded {len(pretrained_dict)} GR1 parameters")
                    print(f"   Video decoder components: {decoder_loaded}")
                    print(f"   Missing keys: {len(missing_keys)}, Incompatible: {incompatible_count}, Skipped: {skipped_count}")
                else:
                    print("No compatible GR1 weights found!")
                    
            else:
                print(f"⚠️  GR1 weights file not found: {pretrained_path}")
            
            # Load MAE weights (if available)
            if os.path.exists(mae_pretrained_path):
                print(f"Loading MAE pretrained weights...")
                mae_checkpoint = torch.load(mae_pretrained_path, map_location='cpu')
                
                # Extract MAE state dict
                if 'model' in mae_checkpoint:
                    mae_state_dict = mae_checkpoint['model']
                else:
                    mae_state_dict = mae_checkpoint
                
                # Load MAE weights into the MAE model
                if hasattr(self.model_mae, 'load_state_dict'):
                    try:
                        missing_keys, unexpected_keys = self.model_mae.load_state_dict(mae_state_dict, strict=False)
                        print(f"Successfully loaded MAE weights")
                    except Exception as e:
                        print(f"MAE weight loading failed: {e}")
                else:
                    print("⚠️  MAE model does not support load_state_dict")
                    
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print("Continuing with random initialization...")
    
    def _load_video_decoder_weights(self, state_dict, model_dict, pretrained_dict):
        """Enhanced loading for video decoder components with simplified logging."""
        decoder_mappings = {
            'decoder_embed': 'video_decoder_embed',
            'mask_token': 'video_mask_token', 
            'decoder_blocks': 'video_decoder_blocks',
            'decoder_norm': 'video_decoder_norm',
            'decoder_pred': 'video_decoder_pred',
            'decoder_pos_embed': 'video_decoder_pos_embed'
        }
        
        loaded_components = 0
        
        for old_key, new_key in decoder_mappings.items():
            # Find matching keys in state_dict
            matching_keys = [k for k in state_dict.keys() if old_key in k]
            
            for old_full_key in matching_keys:
                # Construct new key
                if old_key == 'decoder_blocks':
                    if 'decoder_blocks' in old_full_key:
                        new_full_key = old_full_key.replace('decoder_blocks', 'video_decoder_blocks')
                        if new_full_key in model_dict and model_dict[new_full_key].shape == state_dict[old_full_key].shape:
                            pretrained_dict[new_full_key] = state_dict[old_full_key]
                            loaded_components += 1
                else:
                    new_full_key = old_full_key.replace(old_key, new_key)
                    if new_full_key in model_dict and model_dict[new_full_key].shape == state_dict[old_full_key].shape:
                        pretrained_dict[new_full_key] = state_dict[old_full_key]
                        loaded_components += 1
        
        return loaded_components
    

    
    def forward(
        self,
        video_frames=None,
        joint_pos=None,
        language=None,
        attention_mask=None,
        actions=None,  # Target actions for training
        is_training=True
    ):
        """
        Forward pass with SLIDING WINDOW support and proper temporal alignment.
        
        SLIDING WINDOW MODE: Input sequence [t-window_size+1:t] -> Predict (action[t], video_frame[t+1])
        
        Args:
            video_frames: (batch_size, window_size, 3, H, W) - Input video sequence  
            joint_pos: (batch_size, window_size, 56) - Input joint positions sequence
            language: (batch_size, 77) - Tokenized instruction
            attention_mask: Optional attention mask
            actions: Target actions for training (not used in forward, for compatibility)
            is_training: Whether in training mode
            
        Returns:
            predictions: Dict with 'action_preds' and 'video_preds'
        
        Sequence format per timestep: (l, joint_pos_t, video_frame_t, [video], [act])
        """
        batch_size, sequence_length = joint_pos.shape[:2]
        device = joint_pos.device
        
        # STRICT: Video frames must be provided for video prediction
        if self.fwd_pred and video_frames is None:
            raise ValueError("Video frames must be provided when forward prediction is enabled")
        
        # Handle video frames
        if video_frames is None:
            # Create dummy video frames only if video prediction is disabled
            if self.fwd_pred:
                raise ValueError("Cannot create dummy video frames when video prediction is enabled")
            video_frames = torch.zeros(batch_size, sequence_length, 3, *self.image_size, device=device)
        
        # Encode language with CLIP
        if language.dtype in [torch.long, torch.int32, torch.int64]:
            # Language is tokenized - encode with CLIP
            try:
                lang_embeddings = self.model_clip.encode_text(language)
                lang_embeddings = lang_embeddings / (lang_embeddings.norm(dim=1, keepdim=True) + 1e-6)
            except Exception as e:
                raise RuntimeError(f"CLIP encoding failed: {e}")
        else:
            # Language is already encoded
            lang_embeddings = language
            if lang_embeddings.dtype != torch.float32:
                lang_embeddings = lang_embeddings.float()
            lang_embeddings = lang_embeddings / (lang_embeddings.norm(dim=1, keepdim=True) + 1e-6)
        
        lang_embeddings = self.embed_lang(lang_embeddings.float())  # (b, h)
        
        # Encode joint positions with hierarchical encoder
        joint_pos_embeddings = self.joint_pos_encoder(joint_pos)  # (b, t, h)
        
        # Encode video frames with MAE
        video_flat = video_frames.view(batch_size * sequence_length, 3, *self.image_size)
        try:
            img_embeddings, patch_embeddings = self.model_mae(video_flat)
        except Exception as e:
            raise RuntimeError(f"MAE encoding failed: {e}")
        
        img_embeddings = img_embeddings.view(batch_size, sequence_length, -1)  # (b, t, img_feat_dim)
        patch_embeddings = patch_embeddings.view(
            batch_size * sequence_length, -1, self.patch_feat_dim
        )  # (b*t, n_patches, patch_feat_dim)
        
        # Process patch embeddings with perceiver resampler
        patch_embeddings = patch_embeddings.unsqueeze(1)  # (b*t, 1, n_patches, patch_feat_dim)
        patch_embeddings = self.perceiver_resampler(patch_embeddings)  # (b*t, 1, n_patch_latents, patch_feat_dim)
        patch_embeddings = patch_embeddings.squeeze(1)  # (b*t, n_patch_latents, patch_feat_dim)
        patch_embeddings = patch_embeddings.view(
            batch_size, sequence_length, self.n_patch_latents, self.patch_feat_dim
        )  # (b, t, n_patch_latents, patch_feat_dim)
        
        # Embed images and patches
        img_embeddings = self.embed_img(img_embeddings.float())  # (b, t, h)
        patch_embeddings = self.embed_patch(patch_embeddings.float())  # (b, t, n_patch_latents, h)
        
        # Add timestep embeddings
        if sequence_length > self.max_sequence_length:
            raise ValueError(f"Sequence length {sequence_length} exceeds max_sequence_length {self.max_sequence_length}")
        
        time_embeddings = self.embed_timestep.weight[:sequence_length]  # (actual_seq_len, h)
        
        # Expand language embeddings for each timestep
        lang_embeddings = lang_embeddings.unsqueeze(1).expand(-1, sequence_length, -1)  # (b, t, h)
        lang_embeddings = lang_embeddings + time_embeddings.unsqueeze(0)  # (b, t, h)
        
        joint_pos_embeddings = joint_pos_embeddings + time_embeddings.unsqueeze(0)  # (b, t, h)
        img_embeddings = img_embeddings + time_embeddings.unsqueeze(0)  # (b, t, h)
        patch_embeddings = patch_embeddings + time_embeddings.unsqueeze(0).unsqueeze(2)  # (b, t, n_patch_latents, h)
        
        # Create query tokens with improved action chunking (following GR1 approach)
        query_tokens = []
        
        if self.fwd_pred:
            video_queries = self.video_queries.weight.unsqueeze(0).unsqueeze(0).expand(
                batch_size, sequence_length, -1, -1
            )  # (b, t, n_patch_latents+1, h)
            query_tokens.append(video_queries)
        
        if self.act_pred:
            # Improved action chunking following GR1 approach
            base_action_queries = self.action_queries.weight  # (1, h)
            chunk_queries = self.action_chunk_queries.weight + base_action_queries  # (chunk_size, h)
            action_queries = chunk_queries.unsqueeze(0).unsqueeze(0).expand(
                batch_size, sequence_length, -1, -1
            )  # (b, t, chunk_size, h)
            query_tokens.append(action_queries)
        
        # Format sequence: (l, joint_pos, img, patch_tokens, [video_queries], [action_queries])
        sequence_tokens = []
        
        # Add basic tokens
        sequence_tokens.append(lang_embeddings.unsqueeze(2))  # (b, t, 1, h)
        sequence_tokens.append(joint_pos_embeddings.unsqueeze(2))  # (b, t, 1, h)
        sequence_tokens.append(img_embeddings.unsqueeze(2))  # (b, t, 1, h)
        sequence_tokens.append(patch_embeddings)  # (b, t, n_patch_latents, h)
        
        # Add query tokens
        sequence_tokens.extend(query_tokens)
        
        # Concatenate all tokens
        stacked_inputs = torch.cat(sequence_tokens, dim=2)  # (b, t, n_tokens, h)
        
        # Calculate number of tokens per timestep
        n_tokens_per_step = stacked_inputs.shape[2]
        
        # Reshape for transformer
        stacked_inputs = stacked_inputs.view(batch_size, sequence_length * n_tokens_per_step, self.hidden_size)
        
        # Apply layer normalization
        stacked_inputs = self.embed_ln(stacked_inputs)
        
        # Create attention mask for GPT2 transformer
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, sequence_length, device=device)
        
        # GPT2 expects a 2D attention mask: (batch_size, sequence_length * n_tokens_per_step)
        # where 1 = attend, 0 = mask (ignore)
        expanded_attention_mask = attention_mask.unsqueeze(2).expand(-1, -1, n_tokens_per_step)
        final_mask = expanded_attention_mask.reshape(batch_size, sequence_length * n_tokens_per_step)
        
        # GPT forward pass
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=final_mask,
        )
        
        x = transformer_outputs['last_hidden_state']
        x = x.view(batch_size, sequence_length, n_tokens_per_step, self.hidden_size)
        
        # Decode predictions
        predictions = {}
        
        # Action prediction
        if self.act_pred:
            # Get action query outputs
            action_start_idx = 3 + self.n_patch_latents  # After lang, joint_pos, img, patches
            if self.fwd_pred:
                action_start_idx += self.n_patch_latents + 1  # After video queries
            
            action_embeddings = x[:, :, action_start_idx:action_start_idx+self.chunk_size]  # (b, t, chunk_size, h)
            action_preds = self.action_decoder(action_embeddings)  # (b, t, chunk_size, act_dim)
            predictions['action_preds'] = action_preds
        
        # Video prediction
        if self.fwd_pred:
            video_start_idx = 3 + self.n_patch_latents  # After lang, joint_pos, img, patches
            video_embeddings = x[:, :, video_start_idx:video_start_idx+self.n_patch_latents+1]  # (b, t, n_patch_latents+1, h)
            
            # Video decoder
            video_pred = self.video_decoder_embed(video_embeddings)  # (b, t, n_patch_latents+1, h)
            
            # Add mask tokens for patches
            n_patches = (self.image_size[0] // self.patch_size) ** 2
            mask_tokens = self.video_mask_token.expand(batch_size, sequence_length, n_patches, -1)
            pos_embed = self.video_decoder_pos_embed.unsqueeze(0).expand(batch_size, sequence_length, -1, -1)
            mask_tokens = mask_tokens + pos_embed
            
            # Concatenate query outputs with mask tokens
            video_decoder_input = torch.cat([video_pred, mask_tokens], dim=2)  # (b, t, n_patch_latents+1+n_patches, h)
            
            # Reshape for decoder blocks
            video_decoder_input = video_decoder_input.view(-1, video_decoder_input.shape[2], self.hidden_size)
            
            # Apply decoder blocks
            for block in self.video_decoder_blocks:
                video_decoder_input = block(video_decoder_input)
            
            video_decoder_input = self.video_decoder_norm(video_decoder_input)
            video_preds = self.video_decoder_pred(video_decoder_input)  # (b*t, n_patch_latents+1+n_patches, patch_size^2*3)
            
            # Reshape and extract patch predictions
            video_preds = video_preds.view(batch_size, sequence_length, -1, video_preds.shape[-1])
            video_preds = video_preds[:, :, self.n_patch_latents+1:]  # Remove query tokens, keep patch predictions
            
            predictions['video_preds'] = video_preds
            
            # Create video targets if training - STRICT TEMPORAL ALIGNMENT
            if is_training and video_frames is not None:
                # CRITICAL FIX: Proper temporal alignment t -> t+1
                if sequence_length > 1:
                    # Use frames [1:] as targets for input frames [:-1] 
                    target_frames = video_frames[:, 1:].clone()  # (b, t-1, c, h, w)
                    video_targets = self._create_video_targets_strict(target_frames)
                    
                    # For the last timestep, predict the same frame (no future available)
                    # This is correct behavior - we can only predict where we have targets
                    last_target = video_targets[:, -1:].clone()  # (b, 1, n_patches, patch_dim)
                    video_targets = torch.cat([video_targets, last_target], dim=1)  # (b, t, n_patches, patch_dim)
                else:
                    # For single frame, we cannot do temporal prediction properly
                    raise ValueError("Cannot perform temporal video prediction with sequence_length=1")
                
                predictions['video_targets'] = video_targets
        
        return predictions
    
    def _create_video_targets_strict(self, video_frames):
        """Create video targets for training - STRICT MODE with proper normalization."""
        batch_size, sequence_length, c, h, w = video_frames.shape
        p = self.patch_size
        h_p, w_p = h // p, w // p
        
        # STRICT: No fallback patterns - work with actual video data
        # Check if input is properly denormalized
        frame_min, frame_max = video_frames.min().item(), video_frames.max().item()
        
        if frame_min < -3.0 or frame_max > 3.0:
            # Input appears to be ImageNet normalized, denormalize properly
            mean = torch.tensor([0.485, 0.456, 0.406], device=video_frames.device).view(1, 1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=video_frames.device).view(1, 1, 3, 1, 1)
            
            # Denormalize from ImageNet stats
            video_frames = video_frames * std + mean
            video_frames = torch.clamp(video_frames, 0, 1)
            
            # Verify denormalization worked
            new_min, new_max = video_frames.min().item(), video_frames.max().item()
            if new_max - new_min < 1e-6:
                raise RuntimeError(f"CRITICAL: Video denormalization failed! "
                                 f"Original range: [{frame_min:.4f}, {frame_max:.4f}], "
                                 f"Denormalized range: [{new_min:.4f}, {new_max:.4f}]")
        
        # Patchify
        video_frames = video_frames.reshape(batch_size, sequence_length, c, h_p, p, w_p, p)
        video_targets = video_frames.permute(0, 1, 3, 5, 4, 6, 2)  # (b, t, h_p, w_p, p, p, c)
        video_targets = video_targets.reshape(batch_size, sequence_length, h_p * w_p, p * p * c)
        
        # Normalization for loss computation
        if not self.without_norm_pixel_loss:
            # Use stable global normalization
            global_mean = video_targets.mean()
            global_std = video_targets.std()
            
            if global_std < 1e-6:
                raise RuntimeError(f"CRITICAL: Video targets have zero variance! "
                                 f"Mean: {global_mean:.6f}, Std: {global_std:.6f}")
            
            video_targets = (video_targets - global_mean) / global_std
        
        return video_targets
    
    def load_video_frames(self, video_paths, sequence_length):
        """Load video frames from paths (kept for compatibility)."""
        # This method is kept for compatibility but video loading should be handled by VideoProcessor
        if isinstance(video_paths, str):
            video_paths = [video_paths]
        
        frames = []
        for video_path in video_paths[:sequence_length]:
            if video_path and os.path.exists(os.path.join(self.video_base_path, video_path)):
                cap = cv2.VideoCapture(os.path.join(self.video_base_path, video_path))
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, self.image_size)
                    frame = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
                    frames.append(frame)
                cap.release()
            else:
                # STRICT MODE: Raise error instead of creating dummy frames
                raise FileNotFoundError(f"Video not found: {os.path.join(self.video_base_path, video_path)}")
        
        # Ensure we have the right number of frames
        if len(frames) != sequence_length:
            raise ValueError(f"Expected {sequence_length} frames, got {len(frames)}")
        
        return torch.stack(frames)  # (T, 3, H, W)
