import cv2
import torch
import numpy as np
import os
from typing import List, Union, Tuple
from torchvision import transforms
from PIL import Image


class VideoProcessor:
    """
    Video processor for extracting and preprocessing video frames for the HumanoidGR model.
    Handles video loading, frame extraction, and preprocessing to match MAE input requirements.
    STRICT MODE: No dummy frames - raises errors if videos cannot be loaded.
    """
    
    def __init__(self, 
                 video_base_path: str = "data/video/snippet_videos",
                 image_size: Tuple[int, int] = (224, 224),
                 normalize: bool = True):
        """
        Initialize video processor.
        
        Args:
            video_base_path: Base path where video files are stored
            image_size: Target image size (H, W) for resizing
            normalize: Whether to normalize frames with ImageNet stats
        """
        self.video_base_path = video_base_path
        self.image_size = image_size
        self.normalize = normalize
        
        # Define preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        
        # ImageNet normalization stats for MAE
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
        
        # Verify video base path exists
        if not os.path.exists(video_base_path):
            raise FileNotFoundError(f"Video base path does not exist: {video_base_path}")
    
    def load_single_video_frames(self, 
                                video_path: str, 
                                num_frames: int,
                                start_frame: int = 0) -> torch.Tensor:
        """
        Load frames from a single video file - STRICT MODE.
        
        Args:
            video_path: Path to video file (relative to video_base_path)
            num_frames: Number of frames to extract
            start_frame: Starting frame index
            
        Returns:
            torch.Tensor: (num_frames, 3, H, W) tensor of video frames
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video cannot be read or has no frames
        """
        full_path = os.path.join(self.video_base_path, video_path)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Video file not found: {full_path}")
        
        cap = cv2.VideoCapture(full_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            raise ValueError(f"No frames found in video: {full_path}")
        
        frames = []
        
        try:
            # Calculate frame indices to extract uniformly
            if num_frames == 1:
                frame_indices = [start_frame] if start_frame < total_frames else [total_frames // 2]
            else:
                end_frame = min(start_frame + num_frames, total_frames)
                if end_frame <= start_frame:
                    raise ValueError(f"Invalid frame range: start={start_frame}, end={end_frame}, total={total_frames}")
                frame_indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    cap.release()
                    raise ValueError(f"Failed to read frame {frame_idx} from {full_path}")
                
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image for transforms
                frame = Image.fromarray(frame)
                # Apply transforms
                frame_tensor = self.transform(frame)
                
                # Normalize with ImageNet stats if required
                if self.normalize:
                    frame_tensor = transforms.Normalize(self.mean, self.std)(frame_tensor)
                
                frames.append(frame_tensor)
            
            cap.release()
            
        except Exception as e:
            cap.release()
            raise RuntimeError(f"Error processing video {full_path}: {e}")
        
        if len(frames) != num_frames:
            raise ValueError(f"Expected {num_frames} frames, got {len(frames)} from {full_path}")
        
        return torch.stack(frames)
    
    def load_video_sequence_for_trajectory(self, 
                                         video_path: str, 
                                         sequence_length: int,
                                         trajectory_length: int = None,
                                         start_time: float = 0.0) -> torch.Tensor:
        """
        Load video frames for a specific trajectory with OPTIMIZED sequential reading.
        
        Args:
            video_path: Path to video file
            sequence_length: Number of frames needed for the sequence
            trajectory_length: Length of the trajectory in the dataset
            start_time: Start time offset in the video (seconds)
            
        Returns:
            torch.Tensor: (sequence_length, 3, H, W) tensor of video frames
        """
        full_path = os.path.join(self.video_base_path, video_path)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Video file not found: {full_path}")
        
        cap = cv2.VideoCapture(full_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames == 0:
            cap.release()
            raise ValueError(f"No frames found in video: {full_path}")
        
        if fps <= 0:
            cap.release()
            raise ValueError(f"Invalid FPS in video: {full_path}")
        
        # Calculate start frame based on start_time
        start_frame = int(start_time * fps)
        
        # If trajectory_length is provided, ensure we have enough frames
        frames_needed = trajectory_length or sequence_length
        if start_frame + frames_needed > total_frames:
            cap.release()
            raise ValueError(f"Not enough frames in video {full_path}: "
                           f"need {frames_needed} frames starting from {start_frame}, "
                           f"but video only has {total_frames} frames")
        
        frames = []
        
        try:
            # OPTIMIZATION: Set start position once and read sequentially
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Pre-allocate tensors for better memory efficiency
            frame_tensors = []
            
            # Read frames sequentially (much faster than seeking each frame)
            for i in range(sequence_length):
                ret, frame = cap.read()
                
                if not ret:
                    cap.release()
                    raise ValueError(f"Failed to read frame {start_frame + i} from {full_path}")
                
                # OPTIMIZATION: Vectorized processing
                # Convert BGR to RGB and resize in one operation
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.image_size)
                
                # Convert to tensor directly (avoiding PIL conversion)
                frame_tensor = torch.from_numpy(frame).float() / 255.0
                frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC -> CHW
                
                # Normalize with ImageNet stats if required
                if self.normalize:
                    frame_tensor = (frame_tensor - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)
                
                frame_tensors.append(frame_tensor)
            
            cap.release()
            
            # Stack all frames at once
            frames = torch.stack(frame_tensors)
            
        except Exception as e:
            cap.release()
            raise RuntimeError(f"Error processing video {full_path}: {e}")
        
        return frames
    
    def load_video_sequence(self, 
                           video_paths: Union[str, List[str]], 
                           sequence_length: int) -> torch.Tensor:
        """
        Load a sequence of video frames - STRICT MODE.
        
        Args:
            video_paths: Single video path or list of video paths
            sequence_length: Number of frames in the sequence
            
        Returns:
            torch.Tensor: (sequence_length, 3, H, W) tensor of video frames
        """
        if isinstance(video_paths, str):
            # Single video - extract frames sequentially
            return self.load_single_video_frames(video_paths, sequence_length)
        
        elif isinstance(video_paths, list):
            # Multiple videos - one frame per video (or handle appropriately)
            frames = []
            for i, video_path in enumerate(video_paths[:sequence_length]):
                if not video_path:
                    raise ValueError(f"Empty video path at index {i}")
                
                frame = self.load_single_video_frames(video_path, 1)  # Extract 1 frame
                frames.append(frame[0])  # Remove the first dimension
            
            # Ensure we have exactly sequence_length frames
            if len(frames) != sequence_length:
                raise ValueError(f"Expected {sequence_length} video paths, got {len(frames)}")
            
            return torch.stack(frames)
        
        else:
            raise ValueError(f"Unsupported video_paths type: {type(video_paths)}")
    
    def batch_load_videos(self, 
                         batch_video_paths: List[Union[str, List[str]]], 
                         sequence_length: int) -> torch.Tensor:
        """
        Load video sequences for a batch - STRICT MODE.
        
        Args:
            batch_video_paths: List of video paths for each item in the batch
            sequence_length: Number of frames per sequence
            
        Returns:
            torch.Tensor: (batch_size, sequence_length, 3, H, W) tensor
        """
        batch_frames = []
        
        for i, video_paths in enumerate(batch_video_paths):
            try:
                frames = self.load_video_sequence(video_paths, sequence_length)
                batch_frames.append(frames)
            except Exception as e:
                raise RuntimeError(f"Failed to load video for batch item {i}: {e}")
        
        return torch.stack(batch_frames)
    
    def extract_video_info(self, video_path: str) -> dict:
        """
        Extract basic information about a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            dict: Video information including fps, frame count, duration
        """
        full_path = os.path.join(self.video_base_path, video_path)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Video file not found: {full_path}")
        
        cap = cv2.VideoCapture(full_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        return {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration": duration
        }
