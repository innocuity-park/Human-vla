"""
Hierarchical Action Encoder for 56-DOF Humanoid Robot
This module implements hierarchical encoding of joint positions and actions
with separate MLPs for different body parts.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class HierarchicalActionEncoder(nn.Module):
    """
    Hierarchical encoder for 56-DOF humanoid joint positions and actions.
    
    Joint arrangement:
    - Left Leg (indices [0:7]): 7 dimensions
    - Right Leg (indices [7:14]): 7 dimensions  
    - Torso/Spine (indices [14:23]): 9 dimensions
    - Head/Neck (indices [23:32]): 9 dimensions
    - Left Arm (indices [32:44]): 12 dimensions
    - Right Arm (indices [44:56]): 12 dimensions
    """
    
    def __init__(self, final_embedding_size: int = 256):
        super().__init__()
        
        self.final_embedding_size = final_embedding_size
        
        # Define joint indices for different body parts
        self.joint_indices = {
            'left_leg': (0, 7),      # 7 joints
            'right_leg': (7, 14),    # 7 joints
            'torso': (14, 23),       # 9 joints
            'head': (23, 32),        # 9 joints
            'left_arm': (32, 44),    # 12 joints
            'right_arm': (44, 56),   # 12 joints
        }
        
        # Individual MLPs for each body part
        self.left_leg_mlp = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        self.right_leg_mlp = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        self.torso_mlp = nn.Sequential(
            nn.Linear(9, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 12)
        )
        
        self.head_mlp = nn.Sequential(
            nn.Linear(9, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 12)
        )
        
        self.left_arm_mlp = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        self.right_arm_mlp = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # Global fusion MLP
        # Total embedding size: 8 + 8 + 12 + 12 + 16 + 16 = 72
        self.global_mlp = nn.Sequential(
            nn.Linear(72, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, final_embedding_size)
        )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(final_embedding_size)
        
    def forward(self, joint_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hierarchical encoder.
        
        Args:
            joint_data: Tensor of shape (..., 56) containing joint positions or actions
            
        Returns:
            Encoded tensor of shape (..., final_embedding_size)
        """
        # Store original shape for reshaping later
        original_shape = joint_data.shape[:-1]
        
        # Flatten all dimensions except the last one (joint dimension)
        joint_data_flat = joint_data.reshape(-1, 56)
        
        # Extract joint data for each body part
        left_leg_data = joint_data_flat[:, self.joint_indices['left_leg'][0]:self.joint_indices['left_leg'][1]]
        right_leg_data = joint_data_flat[:, self.joint_indices['right_leg'][0]:self.joint_indices['right_leg'][1]]
        torso_data = joint_data_flat[:, self.joint_indices['torso'][0]:self.joint_indices['torso'][1]]
        head_data = joint_data_flat[:, self.joint_indices['head'][0]:self.joint_indices['head'][1]]
        left_arm_data = joint_data_flat[:, self.joint_indices['left_arm'][0]:self.joint_indices['left_arm'][1]]
        right_arm_data = joint_data_flat[:, self.joint_indices['right_arm'][0]:self.joint_indices['right_arm'][1]]
        
        # Encode each body part
        left_leg_encoded = self.left_leg_mlp(left_leg_data)      # (..., 8)
        right_leg_encoded = self.right_leg_mlp(right_leg_data)   # (..., 8)
        torso_encoded = self.torso_mlp(torso_data)               # (..., 12)
        head_encoded = self.head_mlp(head_data)                  # (..., 12)
        left_arm_encoded = self.left_arm_mlp(left_arm_data)      # (..., 16)
        right_arm_encoded = self.right_arm_mlp(right_arm_data)   # (..., 16)
        
        # Concatenate all encoded parts
        concatenated = torch.cat([
            left_leg_encoded,    # 8
            right_leg_encoded,   # 8
            torso_encoded,       # 12
            head_encoded,        # 12
            left_arm_encoded,    # 16
            right_arm_encoded    # 16
        ], dim=-1)  # Total: 72
        
        # Global fusion
        global_encoded = self.global_mlp(concatenated)
        
        # Apply layer normalization
        global_encoded = self.layer_norm(global_encoded)
        
        # Reshape back to original shape + embedding dimension
        output_shape = original_shape + (self.final_embedding_size,)
        return global_encoded.view(output_shape)
    
    def get_body_part_embeddings(self, joint_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get individual body part embeddings (useful for analysis/debugging).
        
        Args:
            joint_data: Tensor of shape (..., 56)
            
        Returns:
            Dictionary with body part names as keys and embeddings as values
        """
        original_shape = joint_data.shape[:-1]
        joint_data_flat = joint_data.reshape(-1, 56)
        
        embeddings = {}
        
        # Extract and encode each body part
        left_leg_data = joint_data_flat[:, self.joint_indices['left_leg'][0]:self.joint_indices['left_leg'][1]]
        embeddings['left_leg'] = self.left_leg_mlp(left_leg_data).view(original_shape + (8,))
        
        right_leg_data = joint_data_flat[:, self.joint_indices['right_leg'][0]:self.joint_indices['right_leg'][1]]
        embeddings['right_leg'] = self.right_leg_mlp(right_leg_data).view(original_shape + (8,))
        
        torso_data = joint_data_flat[:, self.joint_indices['torso'][0]:self.joint_indices['torso'][1]]
        embeddings['torso'] = self.torso_mlp(torso_data).view(original_shape + (12,))
        
        head_data = joint_data_flat[:, self.joint_indices['head'][0]:self.joint_indices['head'][1]]
        embeddings['head'] = self.head_mlp(head_data).view(original_shape + (12,))
        
        left_arm_data = joint_data_flat[:, self.joint_indices['left_arm'][0]:self.joint_indices['left_arm'][1]]
        embeddings['left_arm'] = self.left_arm_mlp(left_arm_data).view(original_shape + (16,))
        
        right_arm_data = joint_data_flat[:, self.joint_indices['right_arm'][0]:self.joint_indices['right_arm'][1]]
        embeddings['right_arm'] = self.right_arm_mlp(right_arm_data).view(original_shape + (16,))
        
        return embeddings


class HierarchicalActionDecoder(nn.Module):
    """
    Hierarchical decoder for 56-DOF humanoid actions.
    Takes a global embedding and produces joint-specific actions.
    """
    
    def __init__(self, input_embedding_size: int = 256):
        super().__init__()
        
        self.input_embedding_size = input_embedding_size
        
        # First, decode to intermediate representation
        self.global_decoder = nn.Sequential(
            nn.Linear(input_embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, 72),  # Same size as concatenated body part embeddings
            nn.ReLU()
        )
        
        # Individual decoders for each body part
        self.left_leg_decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 7)  # 7 joint actions
        )
        
        self.right_leg_decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 7)  # 7 joint actions
        )
        
        self.torso_decoder = nn.Sequential(
            nn.Linear(12, 24),
            nn.ReLU(),
            nn.Linear(24, 9)  # 9 joint actions
        )
        
        self.head_decoder = nn.Sequential(
            nn.Linear(12, 24),
            nn.ReLU(),
            nn.Linear(24, 9)  # 9 joint actions
        )
        
        self.left_arm_decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 12)  # 12 joint actions
        )
        
        self.right_arm_decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 12)  # 12 joint actions
        )
        
    def forward(self, global_embedding: torch.Tensor) -> torch.Tensor:
        """
        Decode global embedding to 56-DOF actions.
        
        Args:
            global_embedding: Tensor of shape (..., input_embedding_size)
            
        Returns:
            Actions tensor of shape (..., 56)
        """
        original_shape = global_embedding.shape[:-1]
        
        # Flatten all dimensions except the last one
        global_flat = global_embedding.reshape(-1, self.input_embedding_size)
        
        # Decode to intermediate representation
        intermediate = self.global_decoder(global_flat)  # (..., 72)
        
        # Split into body part embeddings
        left_leg_emb = intermediate[:, 0:8]
        right_leg_emb = intermediate[:, 8:16]
        torso_emb = intermediate[:, 16:28]
        head_emb = intermediate[:, 28:40]
        left_arm_emb = intermediate[:, 40:56]
        right_arm_emb = intermediate[:, 56:72]
        
        # Decode each body part
        left_leg_actions = self.left_leg_decoder(left_leg_emb)      # (..., 7)
        right_leg_actions = self.right_leg_decoder(right_leg_emb)   # (..., 7)
        torso_actions = self.torso_decoder(torso_emb)               # (..., 9)
        head_actions = self.head_decoder(head_emb)                  # (..., 9)
        left_arm_actions = self.left_arm_decoder(left_arm_emb)      # (..., 12)
        right_arm_actions = self.right_arm_decoder(right_arm_emb)   # (..., 12)
        
        # Concatenate all actions
        actions = torch.cat([
            left_leg_actions,    # [0:7]
            right_leg_actions,   # [7:14]
            torso_actions,       # [14:23]
            head_actions,        # [23:32]
            left_arm_actions,    # [32:44]
            right_arm_actions    # [44:56]
        ], dim=-1)  # Total: 56
        
        # Reshape back to original shape
        output_shape = original_shape + (56,)
        return actions.view(output_shape)

