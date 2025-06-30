"""
HumanoidGR Task for Natural Language Instruction-based Humanoid Control

This module provides the task implementation for HumanoidGR models in DMControl,
supporting natural language instructions and reward computation.
"""

from typing import Any, Callable, Optional, Sequence, Text, Union, List
import numpy as np
import typing

if typing.TYPE_CHECKING:
    from dm_control.locomotion.walkers import legacy_base
    from dm_control import mjcf

from dm_control import composer
from dm_control.locomotion.tasks.reference_pose import tracking
from dm_control.locomotion.tasks.reference_pose import utils
from dm_control.locomotion.tasks.reference_pose import types
import sys
from pathlib import Path

# Add MoCapAct path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root / "origin-project" / "MoCapAct"))

HUMANOID_COLOR = (0.4, 0.6, 0.8, 1.0)  # Blue color for HumanoidGR


class HumanoidGRTask(composer.Task):
    """
    HumanoidGR Task for Natural Language Instruction-based Control
    
    Supports:
    - Natural language instruction processing
    - Reward computation based on instruction compliance  
    - Humanoid locomotion and manipulation
    - Episode management with instruction variation
    """
    
    def __init__(
        self,
        walker: Callable[..., 'legacy_base.Walker'],
        arena: composer.Arena,
        ref_path: Text,
        task_instructions: List[str],
        window_size: int = 15,
        max_episode_steps: int = 1000,
        physics_timestep: float = 0.005,
        control_timestep: float = 0.03,
        reward_type: str = 'instruction_following',
        termination_error_threshold: float = 0.3,
        min_steps: int = 10,
        **kwargs
    ):
        """
        Initialize HumanoidGR task
        
        Args:
            walker: Walker constructor
            arena: Arena for the task
            ref_path: Reference mocap data path
            task_instructions: List of natural language instructions
            window_size: Sliding window size for observations
            max_episode_steps: Maximum steps per episode
            physics_timestep: Physics simulation timestep
            control_timestep: Control timestep
            reward_type: Type of reward computation
            termination_error_threshold: Threshold for early termination
            min_steps: Minimum steps before termination
        """
        
        self._arena = arena
        self._task_instructions = task_instructions
        self._current_instruction = None
        self._instruction_embedding = None
        self._window_size = window_size
        self._max_episode_steps = max_episode_steps
        self._reward_type = reward_type
        self._termination_error_threshold = termination_error_threshold
        self._min_steps = min_steps
        
        # Episode tracking
        self._episode_step = 0
        self._should_terminate = False
        self._instruction_start_time = 0
        
        # Performance tracking
        self._success_metrics = {
            'forward_progress': 0.0,
            'balance_stability': 0.0,
            'instruction_alignment': 0.0,
            'movement_smoothness': 0.0
        }
        
        # Previous state for computing deltas
        self._prev_position = None
        self._prev_orientation = None
        self._prev_velocity = None
        
        # Initialize walker
        if isinstance(walker, type):
            self._walker = walker()
        else:
            self._walker = walker
        
        # Add walker to arena and create root joints (following MoCapAct pattern)
        self._walker.create_root_joints(self._arena.attach(self._walker))
        
        # Set up observables
        self._setup_observables()
        
        # Set timesteps (instead of super().__init__ for composer.Task)
        self.set_timesteps(physics_timestep=physics_timestep, control_timestep=control_timestep)
        
        print(f"✓ HumanoidGR Task initialized")
        print(f"  - Instructions: {len(task_instructions)}")
        print(f"  - Window size: {window_size}")
        print(f"  - Max episode steps: {max_episode_steps}")
        print(f"  - Reward type: {reward_type}")

    def _setup_observables(self):
        """Setup observables for the walker"""
        # Enable key observables
        if hasattr(self._walker, 'observables'):
            # Core proprioceptive observations (use only available CMU humanoid observables)
            if hasattr(self._walker.observables, 'joints_pos'):
                self._walker.observables.joints_pos.enabled = True
            if hasattr(self._walker.observables, 'joints_vel'):
                self._walker.observables.joints_vel.enabled = True
            if hasattr(self._walker.observables, 'body_height'):
                self._walker.observables.body_height.enabled = True
            if hasattr(self._walker.observables, 'world_zaxis'):
                self._walker.observables.world_zaxis.enabled = True
            if hasattr(self._walker.observables, 'end_effectors_pos'):
                self._walker.observables.end_effectors_pos.enabled = True
            if hasattr(self._walker.observables, 'appendages_pos'):
                self._walker.observables.appendages_pos.enabled = True
            if hasattr(self._walker.observables, 'actuator_activation'):
                self._walker.observables.actuator_activation.enabled = True
            
            # These observables don't exist in CMU humanoid, so we check first
            if hasattr(self._walker.observables, 'upright'):
                self._walker.observables.upright.enabled = True
            if hasattr(self._walker.observables, 'velocity'):
                self._walker.observables.velocity.enabled = True
            
            # Sensor observations (from MoCapAct's available observables)
            if hasattr(self._walker.observables, 'sensors_gyro'):
                self._walker.observables.sensors_gyro.enabled = True
            if hasattr(self._walker.observables, 'sensors_accelerometer'):
                self._walker.observables.sensors_accelerometer.enabled = True
            if hasattr(self._walker.observables, 'sensors_velocimeter'):
                self._walker.observables.sensors_velocimeter.enabled = True
            if hasattr(self._walker.observables, 'sensors_torque'):
                self._walker.observables.sensors_torque.enabled = True
            if hasattr(self._walker.observables, 'sensors_touch'):
                self._walker.observables.sensors_touch.enabled = True
            if hasattr(self._walker.observables, 'sensors_framequat'):
                self._walker.observables.sensors_framequat.enabled = True

    @property
    def root_entity(self):
        return self._arena

    @property
    def task_observables(self):
        return {}

    def initialize_episode_mjcf(self, random_state: np.random.RandomState):
        """Initialize MJCF for the episode"""
        # Reset episode tracking
        self._episode_step = 0
        self._should_terminate = False
        self._instruction_start_time = 0
        
        # Reset performance metrics
        self._success_metrics = {
            'forward_progress': 0.0,
            'balance_stability': 0.0,
            'instruction_alignment': 0.0,
            'movement_smoothness': 0.0
        }
        
        # Reset previous state
        self._prev_position = None
        self._prev_orientation = None
        self._prev_velocity = None

    def initialize_episode(self, physics: 'mjcf.Physics', random_state: np.random.RandomState):
        """Initialize physics for the episode"""
        # Set humanoid color
        try:
            if hasattr(physics.named.model, 'mat_rgba') and 'walker/self' in physics.named.model.mat_rgba._names:
                physics.named.model.mat_rgba['walker/self'][:] = HUMANOID_COLOR
        except (AttributeError, KeyError, TypeError):
            # Material doesn't exist or can't be set, skip silently
            pass
        
        # Initialize walker in standing pose
        self._initialize_walker_pose(physics)
        
        # Select random instruction if not set
        if self._current_instruction is None:
            instruction_idx = random_state.randint(len(self._task_instructions))
            self._current_instruction = self._task_instructions[instruction_idx]
        
        print(f" Episode initialized with instruction: '{self._current_instruction}'")

    def _initialize_walker_pose(self, physics: 'mjcf.Physics'):
        """Initialize walker in a stable standing pose"""
        # Use the stand initializer from MoCapAct
        try:
            from mocapact.envs.dm_control_wrapper import StandInitializer
            initializer = StandInitializer()
            initializer.initialize_pose(physics, self._walker, np.random.RandomState())
        except Exception as e:
            print(f"Warning: Could not use StandInitializer: {e}")
            # Fallback: set to approximate standing pose
            if hasattr(physics.named.data, 'qpos'):
                # Set approximate standing joint positions
                standing_pose = np.array([
                    0, 0, 1.0,  # Root position (x, y, z)
                    1, 0, 0, 0,  # Root orientation (quat)
                    0, 0, 0,     # Free joint velocities
                    0, 0, 0,     # Root angular velocities
                    # Joint positions (approximate standing)
                    0, 0, -0.1, 0.2, -0.1, 0,  # Left leg
                    0, 0, -0.1, 0.2, -0.1, 0,  # Right leg  
                    0, 0, 0, 0, 0, 0, 0, 0, 0,  # Spine and head
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # Left arm
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0   # Right arm
                ])
                
                # Ensure we don't exceed available DOFs
                available_dofs = len(physics.named.data.qpos)
                pose_to_set = standing_pose[:available_dofs]
                physics.named.data.qpos[:len(pose_to_set)] = pose_to_set

    def get_observation(self, physics: 'mjcf.Physics') -> dict:
        """Get task-specific observations"""
        obs = {}
        
        # Add instruction information (placeholder)
        obs['instruction_step'] = np.array([self._episode_step - self._instruction_start_time], dtype=np.float32)
        obs['episode_progress'] = np.array([self._episode_step / self._max_episode_steps], dtype=np.float32)
        
        return obs

    def get_reward(self, physics: 'mjcf.Physics') -> float:
        """Compute reward based on current instruction and performance"""
        
        if self._reward_type == 'instruction_following':
            return self._compute_instruction_following_reward(physics)
        elif self._reward_type == 'locomotion':
            return self._compute_locomotion_reward(physics)
        else:
            return self._compute_default_reward(physics)

    def _compute_instruction_following_reward(self, physics: 'mjcf.Physics') -> float:
        """Compute reward based on instruction following"""
        total_reward = 0.0
        
        # Get current state
        current_position = self._get_position(physics)
        current_orientation = self._get_orientation(physics)
        current_velocity = self._get_velocity(physics)
        
        # Balance and stability reward (always important)
        balance_reward = self._compute_balance_reward(physics)
        total_reward += balance_reward * 0.3
        
        # Instruction-specific rewards
        instruction = self._current_instruction.lower()
        
        if 'forward' in instruction:
            # Reward forward movement
            if self._prev_position is not None:
                forward_progress = current_position[0] - self._prev_position[0]
                forward_reward = np.clip(forward_progress * 10, -1, 1)
                total_reward += forward_reward * 0.4
        
        elif 'backward' in instruction:
            # Reward backward movement
            if self._prev_position is not None:
                backward_progress = self._prev_position[0] - current_position[0]
                backward_reward = np.clip(backward_progress * 10, -1, 1)
                total_reward += backward_reward * 0.4
        
        elif 'left' in instruction:
            # Reward leftward movement or turning
            if self._prev_position is not None:
                left_progress = current_position[1] - self._prev_position[1]
                left_reward = np.clip(left_progress * 10, -1, 1)
                total_reward += left_reward * 0.4
        
        elif 'right' in instruction:
            # Reward rightward movement or turning
            if self._prev_position is not None:
                right_progress = self._prev_position[1] - current_position[1]
                right_reward = np.clip(right_progress * 10, -1, 1)
                total_reward += right_reward * 0.4
        
        elif 'stand' in instruction or 'balance' in instruction:
            # Reward minimal movement and good balance
            if self._prev_position is not None:
                position_change = np.linalg.norm(current_position[:2] - self._prev_position[:2])
                stillness_reward = np.exp(-position_change * 50)  # Exponential decay for movement
                total_reward += stillness_reward * 0.4
        
        elif 'circle' in instruction:
            # Reward circular motion
            if self._prev_position is not None:
                # Check if moving in roughly circular pattern
                distance_from_origin = np.linalg.norm(current_position[:2])
                if distance_from_origin > 0.5:  # Reasonable circle radius
                    # Reward consistent distance from center
                    circle_reward = np.exp(-abs(distance_from_origin - 2.0))
                    total_reward += circle_reward * 0.4
        
        # Movement smoothness reward
        smoothness_reward = self._compute_smoothness_reward(physics)
        total_reward += smoothness_reward * 0.2
        
        # Energy efficiency reward (penalize excessive joint torques)
        efficiency_reward = self._compute_efficiency_reward(physics)
        total_reward += efficiency_reward * 0.1
        
        # Update previous state
        self._prev_position = current_position.copy()
        self._prev_orientation = current_orientation.copy()
        self._prev_velocity = current_velocity.copy()
        
        return float(np.clip(total_reward, -1, 1))

    def _compute_balance_reward(self, physics: 'mjcf.Physics') -> float:
        """Compute reward for maintaining balance"""
        try:
            # Check if walker has upright observable
            if hasattr(self._walker, 'observables') and hasattr(self._walker.observables, 'upright'):
                upright = physics.bind(self._walker.observables.upright).copy()
                if isinstance(upright, np.ndarray) and len(upright) > 0:
                    return float(upright[0])
            
            # Fallback: compute upright from body orientation
            body_orientation = self._get_orientation(physics)
            if len(body_orientation) >= 4:  # Quaternion
                # Extract z-component of up vector from quaternion
                qw, qx, qy, qz = body_orientation[:4]
                up_z = 2 * (qw * qz + qx * qy)  # Z component of rotated up vector
                return float(np.clip(up_z, 0, 1))
            
            return 0.5  # Neutral reward if can't compute
            
        except Exception:
            return 0.5

    def _compute_smoothness_reward(self, physics: 'mjcf.Physics') -> float:
        """Compute reward for smooth movement"""
        if self._prev_velocity is None:
            return 1.0
        
        try:
            current_velocity = self._get_velocity(physics)
            velocity_change = np.linalg.norm(current_velocity - self._prev_velocity)
            # Penalize sudden velocity changes
            smoothness = np.exp(-velocity_change * 2)
            return float(smoothness)
        except Exception:
            return 0.5

    def _compute_efficiency_reward(self, physics: 'mjcf.Physics') -> float:
        """Compute reward for energy efficiency"""
        try:
            # Get actuator forces
            if hasattr(physics.named.data, 'actuator_force'):
                forces = physics.named.data.actuator_force
                if len(forces) > 0:
                    # Penalize high forces
                    force_magnitude = np.mean(np.abs(forces))
                    efficiency = np.exp(-force_magnitude * 0.1)
                    return float(efficiency)
            return 1.0
        except Exception:
            return 0.5

    def _compute_locomotion_reward(self, physics: 'mjcf.Physics') -> float:
        """Simple locomotion reward"""
        balance_reward = self._compute_balance_reward(physics)
        forward_velocity = self._get_velocity(physics)[0] if len(self._get_velocity(physics)) > 0 else 0
        locomotion_reward = np.clip(forward_velocity, 0, 2) / 2.0
        return balance_reward * 0.6 + locomotion_reward * 0.4

    def _compute_default_reward(self, physics: 'mjcf.Physics') -> float:
        """Default reward for staying upright"""
        return self._compute_balance_reward(physics)

    def _get_position(self, physics: 'mjcf.Physics') -> np.ndarray:
        """Get walker position"""
        try:
            if hasattr(physics.named.data, 'body_xpos') and 'walker/root' in physics.named.data.body_xpos:
                return physics.named.data.body_xpos['walker/root'].copy()
            elif hasattr(physics.named.data, 'qpos'):
                return physics.named.data.qpos[:3].copy()  # Assume first 3 are position
            else:
                return np.zeros(3)
        except Exception:
            return np.zeros(3)

    def _get_orientation(self, physics: 'mjcf.Physics') -> np.ndarray:
        """Get walker orientation"""
        try:
            if hasattr(physics.named.data, 'body_xquat') and 'walker/root' in physics.named.data.body_xquat:
                return physics.named.data.body_xquat['walker/root'].copy()
            elif hasattr(physics.named.data, 'qpos') and len(physics.named.data.qpos) >= 7:
                return physics.named.data.qpos[3:7].copy()  # Assume quat at positions 3:7
            else:
                return np.array([1, 0, 0, 0])  # Identity quaternion
        except Exception:
            return np.array([1, 0, 0, 0])

    def _get_velocity(self, physics: 'mjcf.Physics') -> np.ndarray:
        """Get walker velocity"""
        try:
            if hasattr(self._walker, 'observables') and hasattr(self._walker.observables, 'velocity'):
                velocity_obs = physics.bind(self._walker.observables.velocity)
                return velocity_obs.copy()
            elif hasattr(physics.named.data, 'qvel'):
                return physics.named.data.qvel[:3].copy()  # Assume first 3 are linear velocity
            else:
                return np.zeros(3)
        except Exception:
            return np.zeros(3)

    def should_terminate_episode(self, physics: 'mjcf.Physics') -> bool:
        """Check if episode should terminate"""
        # Terminate if maximum steps reached
        if self._episode_step >= self._max_episode_steps:
            return True
        
        # Terminate if walker has fallen (if minimum steps passed)
        if self._episode_step > self._min_steps:
            balance_reward = self._compute_balance_reward(physics)
            if balance_reward < 0.1:  # Very poor balance
                return True
        
        return self._should_terminate

    def after_step(self, physics: 'mjcf.Physics', random_state: np.random.RandomState):
        """Called after each step"""
        self._episode_step += 1
        
        # Update success metrics
        self._update_success_metrics(physics)

    def _update_success_metrics(self, physics: 'mjcf.Physics'):
        """Update success metrics for logging"""
        # Update balance stability
        balance = self._compute_balance_reward(physics)
        self._success_metrics['balance_stability'] = (
            self._success_metrics['balance_stability'] * 0.9 + balance * 0.1
        )
        
        # Update forward progress
        current_pos = self._get_position(physics)
        if self._prev_position is not None:
            progress = current_pos[0] - self._prev_position[0] 
            self._success_metrics['forward_progress'] += max(0, progress)
        
        # Update movement smoothness
        smoothness = self._compute_smoothness_reward(physics)
        self._success_metrics['movement_smoothness'] = (
            self._success_metrics['movement_smoothness'] * 0.9 + smoothness * 0.1
        )

    def set_instruction(self, instruction: str):
        """Set the current instruction"""
        self._current_instruction = instruction
        self._instruction_start_time = self._episode_step
        print(f" Instruction updated: '{instruction}'")

    def get_current_instruction(self) -> str:
        """Get current instruction"""
        return self._current_instruction or "stand still and balance"

    def get_success_metrics(self) -> dict:
        """Get current success metrics"""
        return self._success_metrics.copy()

    def get_episode_info(self) -> dict:
        """Get episode information"""
        return {
            'episode_step': self._episode_step,
            'instruction': self._current_instruction,
            'max_episode_steps': self._max_episode_steps,
            'success_metrics': self.get_success_metrics()
        } 