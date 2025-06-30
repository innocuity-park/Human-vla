import lmdb
from pickle import loads
import numpy as np
import torch
from torch.utils.data import Dataset
import clip
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

class HumanoidLMDBDataset(Dataset):
    """
    SLIDING WINDOW Dataset for HumanoidGR Training
    
    Implements sliding window approach with temporal alignment:
    - Input sequence [t-14:t] -> predict (action[t], video_frame[t+1])
    - Window size: 15 timesteps
    - Episode-based processing with sliding window sampling
    """
    
    def __init__(self, 
                 lmdb_dir, 
                 window_size=15,  # Sliding window size
                 action_dim=56, 
                 state_dim=56, 
                 start_ratio=0.0, 
                 end_ratio=1.0, 
                 image_size=(224, 224),
                 verbose=True):
        super(HumanoidLMDBDataset).__init__()
        
        self.window_size = window_size
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.image_size = image_size
        self.lmdb_dir = lmdb_dir
        self.verbose = verbose
        
        # Initialize CLIP model for text tokenization
        self.clip_model = None
        self.clip_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load sliding window samples
        self.sliding_windows = self._create_sliding_windows(start_ratio, end_ratio)
        
        if self.verbose:
            print(f"Sliding Window Dataset initialized")
            print(f"  - Total sliding windows: {len(self.sliding_windows)}")
            print(f"  - Window size: {window_size} timesteps")
            print(f"  - Temporal alignment: input[t-{window_size-1}:t] -> predict(action[t], video[t+1])")
            print(f"  - Episode-based processing with sliding window sampling")

    def _get_clip_model(self):
        """Lazy initialization of CLIP model"""
        if self.clip_model is None:
            try:
                # Load CLIP on GPU for better performance
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.clip_model, _ = clip.load("ViT-B/32", device=device)
                if self.verbose:
                    print(f"✓ CLIP model loaded on {device}")
            except Exception as e:
                print(f"⚠ Failed to load CLIP model: {e}")
                self.clip_model = None
        return self.clip_model

    def _tokenize_text(self, text):
        """Tokenize raw text using CLIP"""
        clip_model = self._get_clip_model()
        if clip_model is None:
            raise RuntimeError("CLIP model failed to load - cannot tokenize text")
        
        try:
            tokens = clip.tokenize([text], truncate=True)  # Tokenize on CPU
            return tokens.squeeze(0)  # Shape: (77,) - return CPU tensor for compatibility
        except Exception as e:
            raise RuntimeError(f"Failed to tokenize text '{text}': {e}")

    def _create_sliding_windows(self, start_ratio, end_ratio):
        """Create sliding window samples from episodes"""
        if self.verbose:
            print("Creating sliding window samples...")
        sliding_windows = []
        
        env = lmdb.open(self.lmdb_dir, readonly=True, create=False, lock=False)
        with env.begin() as txn:
            # Get total episodes
            total_episodes = loads(txn.get('total_episodes'.encode()))
            start_episode = int(total_episodes * start_ratio)
            end_episode = int(total_episodes * end_ratio)
            
            if self.verbose:
                print(f"  Processing episodes {start_episode} to {end_episode-1}")
            
            for episode_id in range(start_episode, end_episode):
                try:
                    # Get episode metadata
                    inst_token_data = txn.get(f'inst_token_{episode_id}'.encode())
                    episode_length_data = txn.get(f'episode_length_{episode_id}'.encode())
                    episode_start_step_data = txn.get(f'episode_start_step_{episode_id}'.encode())
                    
                    if all([inst_token_data, episode_length_data, episode_start_step_data]):
                        episode_length = loads(episode_length_data)
                        episode_start_step = loads(episode_start_step_data)
                        inst_token = loads(inst_token_data)
                        
                        # Create sliding windows for this episode
                        # Need at least window_size + 1 timesteps (window + 1 for prediction target)
                        if episode_length >= self.window_size + 1:
                            # Create sliding windows
                            for window_start in range(episode_length - self.window_size):
                                sliding_windows.append({
                                    'episode_id': episode_id,
                                    'episode_length': episode_length,
                                    'episode_start_step': episode_start_step,
                                    'window_start': window_start,  # Start of window within episode
                                    'window_end': window_start + self.window_size,  # End of window (exclusive)
                                    'prediction_step': window_start + self.window_size,  # Timestep to predict
                                    'inst_token': inst_token
                                })
                        else:
                            if self.verbose and episode_id < start_episode + 5:  # Only warn for first few
                                print(f"⚠ Episode {episode_id}: Too short ({episode_length} < {self.window_size + 1})")
                    
                except Exception as e:
                    if self.verbose and episode_id < start_episode + 5:
                        print(f"⚠ Episode {episode_id}: Error - {e}")
                    continue
        
        env.close()
        
        if len(sliding_windows) == 0:
            raise RuntimeError("No valid sliding windows found!")
        
        if self.verbose:
            print(f"✓ Created {len(sliding_windows)} sliding window samples")
        return sliding_windows

    def open_lmdb(self):
        if not hasattr(self, 'env'):
            self.env = lmdb.open(self.lmdb_dir, readonly=True, create=False, lock=False)
            self.txn = self.env.begin()

    def __getitem__(self, idx):
        if not hasattr(self, 'env'):
            self.open_lmdb()

        # Get sliding window info
        window_info = self.sliding_windows[idx]
        episode_id = window_info['episode_id']
        episode_start_step = window_info['episode_start_step']
        window_start = window_info['window_start']
        window_end = window_info['window_end']
        prediction_step = window_info['prediction_step']
        raw_text = window_info['inst_token']
        
        # Tokenize instruction
        try:
            if isinstance(raw_text, str):
                inst_token = self._tokenize_text(raw_text)
            else:
                inst_token = torch.tensor(raw_text, dtype=torch.long)
        except Exception as e:
            raise RuntimeError(f"Failed to process instruction for episode {episode_id}: {e}")
        
        # Load sliding window sequence data
        window_length = window_end - window_start  # Should be self.window_size
        
        # Input sequence data [window_start:window_end]
        joint_pos_seq = torch.zeros(window_length, self.state_dim, dtype=torch.float32)
        video_frames_seq = torch.zeros(window_length, 3, *self.image_size, dtype=torch.float32)
        
        # Target data for prediction (at prediction_step)
        target_action = torch.zeros(self.action_dim, dtype=torch.float32)
        target_video_frame = torch.zeros(3, *self.image_size, dtype=torch.float32)
        
        # Load input sequence data
        for seq_idx in range(window_length):
            episode_timestep = window_start + seq_idx
            global_step = episode_start_step + episode_timestep
            
            # Load joint positions for input sequence
            try:
                joint_pos_data = loads(self.txn.get(f'joint_pos_{global_step}'.encode()))
                joint_pos_seq[seq_idx] = torch.tensor(joint_pos_data, dtype=torch.float32)
            except Exception as e:
                raise RuntimeError(f"Failed to load joint_pos for window {idx}, seq_idx {seq_idx}: {e}")
            
            # Load video frames for input sequence
            try:
                video_frame_data = loads(self.txn.get(f'video_frame_{global_step}'.encode()))
                if video_frame_data.dtype == np.uint8:
                    frame_tensor = torch.from_numpy(video_frame_data).float() / 255.0
                else:
                    frame_tensor = torch.from_numpy(video_frame_data).float()
                video_frames_seq[seq_idx] = frame_tensor
            except Exception as e:
                raise RuntimeError(f"Failed to load video_frame for window {idx}, seq_idx {seq_idx}: {e}")
        
        # Load target data (at prediction_step)
        prediction_global_step = episode_start_step + prediction_step
        
        # Target action (at prediction_step)
        try:
            action_data = loads(self.txn.get(f'actions_{prediction_global_step}'.encode()))
            target_action = torch.tensor(action_data, dtype=torch.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to load target action for window {idx}: {e}")
        
        # Target video frame (at prediction_step) 
        try:
            video_frame_data = loads(self.txn.get(f'video_frame_{prediction_global_step}'.encode()))
            if video_frame_data.dtype == np.uint8:
                frame_tensor = torch.from_numpy(video_frame_data).float() / 255.0
            else:
                frame_tensor = torch.from_numpy(video_frame_data).float()
            target_video_frame = frame_tensor
        except Exception as e:
            raise RuntimeError(f"Failed to load target video frame for window {idx}: {e}")
        
        return {
            'episode_id': episode_id,
            'window_start': window_start,
            'prediction_step': prediction_step,
            
            # Input sequence data [window_start:window_end]
            'joint_pos': joint_pos_seq,         # (window_size, 56)
            'video_frames': video_frames_seq,   # (window_size, 3, H, W)
            'inst_token': inst_token,           # (77,)
            
            # Target data (at prediction_step)
            'target_action': target_action,           # (56,) - action to take at prediction_step
            'target_video_frame': target_video_frame, # (3, H, W) - video frame at prediction_step
        }

    def __len__(self):
        return len(self.sliding_windows)


def collate_sliding_windows(batch):
    """
    Custom collate function for sliding window data
    
    Returns:
        - Batch of sliding window sequences for model input
        - Batch of targets for loss computation
    """
    batch_size = len(batch)
    
    # Get dimensions from first sample
    window_size = batch[0]['joint_pos'].shape[0]
    state_dim = batch[0]['joint_pos'].shape[1]
    img_channels, img_h, img_w = batch[0]['video_frames'].shape[1:]
    action_dim = batch[0]['target_action'].shape[0]
    token_length = batch[0]['inst_token'].shape[0]
    
    # Create batched tensors
    batched_data = {
        'episode_ids': [],
        'window_starts': [],
        'prediction_steps': [],
        
        # Input sequences
        'joint_pos': torch.zeros(batch_size, window_size, state_dim),
        'video_frames': torch.zeros(batch_size, window_size, img_channels, img_h, img_w),
        'inst_token': torch.zeros(batch_size, token_length, dtype=torch.long),
        
        # Targets
        'target_actions': torch.zeros(batch_size, action_dim),
        'target_video_frames': torch.zeros(batch_size, img_channels, img_h, img_w),
    }
    
    # Fill batched data
    for i, sample in enumerate(batch):
        batched_data['episode_ids'].append(sample['episode_id'])
        batched_data['window_starts'].append(sample['window_start'])
        batched_data['prediction_steps'].append(sample['prediction_step'])
        
        batched_data['joint_pos'][i] = sample['joint_pos']
        batched_data['video_frames'][i] = sample['video_frames']
        batched_data['inst_token'][i] = sample['inst_token']
        
        batched_data['target_actions'][i] = sample['target_action']
        batched_data['target_video_frames'][i] = sample['target_video_frame']
    
    return batched_data 