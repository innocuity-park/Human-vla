"""
HumanoidGR Environment Wrapper for DMControl Mujoco Simulation

This module provides the environment wrapper for evaluating HumanoidGR models
in the DMControl Mujoco simulation with vision-language-action capabilities.
"""

import numpy as np
from pathlib import Path
from gym import spaces
from typing import Any, Dict, Optional, Tuple, Union, List
from dm_control.locomotion.tasks.reference_pose import types
from dm_control.locomotion.mocap import cmu_mocap_data
from dm_control.locomotion.walkers import cmu_humanoid
import sys
import os

# Add paths for imports  
current_dir = Path(__file__).parent
project_root = current_dir.parent

from . import dm_control_wrapper
from .humanoidGR_task import HumanoidGRTask


class HumanoidGRGymEnv(dm_control_wrapper.DmControlWrapper):
    """
    HumanoidGR Environment Wrapper for DMControl Mujoco Simulation
    
    Provides vision-language-action interface with sliding window evaluation:
    - Third-person camera observations (224x224 RGB)
    - Natural language instruction processing
    - 56-DOF humanoid robot control
    - Sliding window sequence management
    """
    
    def __init__(
        self,
        task_instructions: Optional[List[str]] = None,
        window_size: int = 15,
        mocap_path: Optional[Union[str, Path]] = None,
        task_kwargs: Optional[Dict[str, Any]] = None,
        environment_kwargs: Optional[Dict[str, Any]] = None,
        act_noise: float = 0.,
        enable_language_obs: bool = True,
        
        # Camera configuration for video input
        width: int = 224,
        height: int = 224,
        camera_id: int = 3,  # Third-person view
        
        # Arena configuration
        arena_size: Tuple[float, float] = (20., 20.)
    ):
        """
        Initialize HumanoidGR environment
        
        Args:
            task_instructions: List of natural language instructions for tasks
            window_size: Sliding window size for sequence evaluation
            mocap_path: Path to mocap data
            task_kwargs: Task-specific arguments
            environment_kwargs: Environment-specific arguments
            act_noise: Action noise level
            enable_language_obs: Whether to include language observations
            width: Camera image width
            height: Camera image height
            camera_id: Camera ID for rendering (3 = third-person)
            arena_size: Arena size (width, height)
        """
        
        self.window_size = window_size
        self.enable_language_obs = enable_language_obs
        self.height = height
        self.width = width
        self._current_instruction = None
        self._instruction_index = 0
        
        # Default task instructions
        if task_instructions is None:
            task_instructions = [
                "walk forward naturally",
                "walk backward slowly",
                "turn left while walking",
                "turn right while walking", 
                "stand still and balance",
                "walk in a circle",
                "walk with long strides",
                "walk with short steps",
                "walk and stop periodically",
                "walk forward then turn around"
            ]
        
        self.task_instructions = task_instructions
        
        # Initialize sequence buffers for sliding window
        self._video_sequence = []
        self._joint_pos_sequence = []
        self._step_count = 0
        
        # Setup task kwargs
        task_kwargs = task_kwargs or dict()
        task_kwargs['ref_path'] = mocap_path if mocap_path else cmu_mocap_data.get_path_for_cmu(version='2020')
        task_kwargs['task_instructions'] = self.task_instructions
        task_kwargs['window_size'] = self.window_size
        
        # Initialize parent with HumanoidGR task
        super().__init__(
            HumanoidGRTask,
            task_kwargs=task_kwargs,
            environment_kwargs=environment_kwargs,
            act_noise=act_noise,
            arena_size=arena_size,
            width=width,
            height=height,
            camera_id=camera_id
        )
        
        print(f"✓ HumanoidGR Environment initialized")
        print(f"  - Window size: {window_size}")
        print(f"  - Task instructions: {len(task_instructions)}")
        print(f"  - Camera: {width}x{height}, ID {camera_id}")
        print(f"  - Arena: {arena_size[0]}x{arena_size[1]}m")

    def _get_walker(self):
        """Get the CMU humanoid walker"""
        # Use the local StandInitializer
        from .dm_control_wrapper import StandInitializer
        initializer = StandInitializer()
        return cmu_humanoid.CMUHumanoidPositionControlledV2020(initializer=initializer)

    def _get_arena(self, arena_size):
        """Get the floor arena"""
        from dm_control.locomotion.arenas import floors
        return floors.Floor(arena_size)

    def _create_observation_space(self) -> spaces.Dict:
        """Create observation space including video, proprioception, and language"""
        obs_spaces = dict()
        
        # Standard proprioceptive observations
        for k, v in self._env.observation_spec().items():
            if v.dtype == np.float64 and np.prod(v.shape) > 0:
                obs_spaces[k] = spaces.Box(
                    -np.infty,
                    np.infty,
                    shape=(np.prod(v.shape),),
                    dtype=np.float32
                )
            elif v.dtype == np.uint8:
                tmp = v.generate_value()
                obs_spaces[k] = spaces.Box(
                    v.minimum.item(),
                    v.maximum.item(),
                    shape=tmp.shape,
                    dtype=np.uint8
                )
        
        # Add video sequence observation (sliding window)
        obs_spaces['video_sequence'] = spaces.Box(
            low=0,
            high=255,
            shape=(self.window_size, 3, self.height, self.width),
            dtype=np.uint8
        )
        
        # Add joint position sequence observation (sliding window)
        obs_spaces['joint_pos_sequence'] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, 56),  # 56-DOF humanoid
            dtype=np.float32
        )
        
        # Add language instruction observation
        if self.enable_language_obs:
            obs_spaces['language_instruction'] = spaces.Box(
                low=0,
                high=1,
                shape=(512,),  # Language embedding dimension
                dtype=np.float32
            )
            obs_spaces['language_tokens'] = spaces.Box(
                low=0,
                high=49407,  # CLIP vocab size
                shape=(77,),  # CLIP token length
                dtype=np.int64
            )
        
        return spaces.Dict(obs_spaces)

    def get_observation(self, time_step) -> Dict[str, np.ndarray]:
        """Get observation including video sequence, joint positions, and language"""
        # Get standard observations (but exclude sequences that we'll add)
        obs = {}
        
        # Get base observations from dm_control
        base_obs = self._env.step(self._env.action_spec().minimum * 0).observation if hasattr(self._env, 'step') else {}
        time_step_obs = time_step.observation if hasattr(time_step, 'observation') else {}
        
        # Merge observations from time_step
        for k, v in time_step_obs.items():
            if k not in ['video_sequence', 'joint_pos_sequence']:  # Skip our custom sequences
                if isinstance(v, np.ndarray):
                    if v.dtype == np.float64:
                        obs[k] = v.ravel().astype(np.float32)
                    else:
                        obs[k] = v
                else:
                    obs[k] = np.array(v)
        
        # Get current video frame
        current_frame = self.render(mode='rgb_array')  # Shape: (H, W, 3)
        current_frame = current_frame.transpose(2, 0, 1)  # Convert to (3, H, W)
        
        # Get current joint positions (56-DOF)
        if 'walker/joints_pos' in obs:
            current_joint_pos = obs['walker/joints_pos']
        else:
            # Fallback: extract from other observations
            current_joint_pos = np.zeros(56, dtype=np.float32)
            for key in obs:
                if 'joint' in key.lower() and 'pos' in key.lower():
                    joint_data = obs[key]
                    if len(joint_data) <= 56:
                        current_joint_pos[:len(joint_data)] = joint_data[:56]
                    break
        
        # Update sequences (sliding window)
        self._video_sequence.append(current_frame)
        self._joint_pos_sequence.append(current_joint_pos)
        
        # Maintain sliding window
        if len(self._video_sequence) > self.window_size:
            self._video_sequence.pop(0)
            self._joint_pos_sequence.pop(0)
        
        # Pad sequences if not enough frames yet
        while len(self._video_sequence) < self.window_size:
            self._video_sequence.insert(0, current_frame)
            self._joint_pos_sequence.insert(0, current_joint_pos)
        
        # Create sequence observations
        obs['video_sequence'] = np.stack(self._video_sequence, axis=0)  # (window_size, 3, H, W)
        obs['joint_pos_sequence'] = np.stack(self._joint_pos_sequence, axis=0)  # (window_size, 56)
        
        # Add language observations
        if self.enable_language_obs:
            if self._current_instruction is not None:
                # For now, create dummy embeddings (will be processed by HumanoidGR model)
                obs['language_instruction'] = np.zeros(512, dtype=np.float32)
                obs['language_tokens'] = self._encode_instruction_tokens(self._current_instruction)
            else:
                obs['language_instruction'] = np.zeros(512, dtype=np.float32)  
                obs['language_tokens'] = np.zeros(77, dtype=np.int64)
        
        return obs

    def _encode_instruction_tokens(self, instruction: str) -> np.ndarray:
        """Encode instruction to CLIP tokens (simplified version)"""
        # This is a simplified tokenization - in practice, use CLIP tokenizer
        # For now, create a hash-based encoding
        import hashlib
        
        # Create a deterministic encoding based on instruction
        hash_val = int(hashlib.md5(instruction.encode()).hexdigest()[:8], 16)
        tokens = np.zeros(77, dtype=np.int64)
        
        # Fill with deterministic values based on instruction
        for i, char in enumerate(instruction[:77]):
            if i < 77:
                tokens[i] = (ord(char) + hash_val) % 49407
        
        return tokens

    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment and select new instruction"""
        # Select random instruction for this episode
        self._instruction_index = np.random.randint(len(self.task_instructions))
        self._current_instruction = self.task_instructions[self._instruction_index]
        
        # Set instruction in task
        if hasattr(self._env.task, 'set_instruction'):
            self._env.task.set_instruction(self._current_instruction)
        
        # Reset sequences
        self._video_sequence = []
        self._joint_pos_sequence = []
        self._step_count = 0
        
        # Reset environment
        obs = super().reset()
        
        print(f"🎯 Episode instruction: '{self._current_instruction}'")
        return obs

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        """Step environment and update sequences"""
        obs, reward, done, info = super().step(action)
        
        # Add step information
        self._step_count += 1
        info['step_count'] = self._step_count
        info['current_instruction'] = self._current_instruction
        info['instruction_index'] = self._instruction_index
        info['window_size'] = self.window_size
        
        return obs, reward, done, info

    def get_current_instruction(self) -> Optional[str]:
        """Get current instruction"""
        return self._current_instruction
    
    def set_instruction(self, instruction: str):
        """Set specific instruction"""
        self._current_instruction = instruction
        if hasattr(self._env.task, 'set_instruction'):
            self._env.task.set_instruction(instruction)
    
    def get_video_sequence(self) -> np.ndarray:
        """Get current video sequence"""
        if len(self._video_sequence) >= self.window_size:
            return np.stack(self._video_sequence, axis=0)
        else:
            return None
    
    def get_joint_pos_sequence(self) -> np.ndarray:
        """Get current joint position sequence"""
        if len(self._joint_pos_sequence) >= self.window_size:
            return np.stack(self._joint_pos_sequence, axis=0)
        else:
            return None 