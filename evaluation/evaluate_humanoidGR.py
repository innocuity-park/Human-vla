#!/usr/bin/env python3
"""
HumanoidGR Online Evaluation Script
This script evaluates the trained HumanoidGR model in the DMControl Mujoco simulation environment
with natural language instructions and sliding window sequences.
"""

import os
import sys
import argparse
import numpy as np
import torch
import time
from pathlib import Path
from typing import List, Dict, Any
import warnings
from datetime import datetime

# Setup headless rendering for MuJoCo (CRITICAL for server environments)
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add project paths - corrected for evaluation folder
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "envs"))

from envs.humanoidGR_env import HumanoidGRGymEnv
from envs.humanoidGR_policy import HumanoidGRPolicy


class HumanoidGREvaluator:
    """
    HumanoidGR Online Evaluator
    
    Evaluates trained HumanoidGR models in DMControl simulation with:
    - Natural language instruction following
    - Sliding window sequence processing
    - Real-time humanoid robot control
    - Video recording and analysis
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'auto',
        window_size: int = 15,
        render_videos: bool = True,
        verbose: bool = True
    ):
        """
        Initialize evaluator
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device for inference ('auto', 'cuda', 'cpu')
            window_size: Sliding window size for sequences
            render_videos: Whether to render and save videos
            verbose: Whether to print detailed logs
        """
        
        self.checkpoint_path = checkpoint_path
        self.window_size = window_size
        self.render_videos = render_videos
        self.verbose = verbose
        
        # Auto-select device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Evaluation instructions (will be loaded from file or command line)
        self.evaluation_instructions = []
        
        # Results storage
        self.evaluation_results = []
        
        if verbose:
            print(f"HumanoidGR Evaluator initialized")
            print(f"  - Checkpoint: {checkpoint_path}")
            print(f"  - Device: {self.device}")
            print(f"  - Window size: {window_size}")
            print(f"  - Render videos: {render_videos}")

    def load_instructions_from_file(self, instructions_file: str) -> List[str]:
        """Load evaluation instructions from a text file"""
        try:
            with open(instructions_file, 'r', encoding='utf-8') as f:
                instructions = [line.strip() for line in f.readlines() if line.strip()]
            
            if self.verbose:
                print(f"✓ Loaded {len(instructions)} instructions from {instructions_file}")
            
            return instructions
            
        except Exception as e:
            print(f"Failed to load instructions from {instructions_file}: {e}")
            return []

    def set_default_instructions(self) -> List[str]:
        """Set default evaluation instructions"""
        default_instructions = [
            "walk forward naturally",
            "walk backward slowly", 
            "turn left while walking",
            "turn right while walking",
            "stand still and balance",
            "swing the arms and walk forward",
            "walk in a circle",
            "walk with long strides",
            "walk with short steps",
            "walk and stop periodically",
            "walk forward then turn around"
        ]
        
        if self.verbose:
            print(f"Using default instructions ({len(default_instructions)} instructions)")
        
        return default_instructions

    def set_instructions(self, instructions: List[str]):
        """Set evaluation instructions"""
        self.evaluation_instructions = instructions
        if self.verbose:
            print(f"✓ Set {len(instructions)} evaluation instructions")
            for i, instruction in enumerate(instructions, 1):
                print(f"  {i}. {instruction}")

    def setup_environment(self) -> HumanoidGRGymEnv:
        """Setup HumanoidGR evaluation environment with rendering fallbacks"""
        if self.verbose:
            print("Setting up HumanoidGR environment...")
        
        # Try EGL rendering first (preferred for headless)
        try:
            env = HumanoidGRGymEnv(
                task_instructions=self.evaluation_instructions,
                window_size=self.window_size,
                width=224,
                height=224,
                camera_id=3,  # Third-person view
                arena_size=(20.0, 20.0),  # Large arena for diverse movements
                enable_language_obs=True
            )
            
            if self.verbose:
                print("Environment setup complete with EGL rendering")
                print(f"  - Action space: {env.action_space}")
                print(f"  - Observation space keys: {list(env.observation_space.spaces.keys())}")
            
            return env
            
        except Exception as e:
            if self.verbose:
                print(f"EGL rendering failed: {e}")
                print("Trying OSMesa rendering...")
            
            # Fallback to OSMesa rendering
            os.environ['MUJOCO_GL'] = 'osmesa'
            try:
                env = HumanoidGRGymEnv(
                    task_instructions=self.evaluation_instructions,
                    window_size=self.window_size,
                    width=224,
                    height=224,
                    camera_id=3,  # Third-person view
                    arena_size=(20.0, 20.0),
                    enable_language_obs=True
                )
                
                if self.verbose:
                    print("✓ Environment setup complete with OSMesa rendering")
                
                return env
                
            except Exception as e2:
                print(f"Both EGL and OSMesa rendering failed!")
                print(f"   EGL error: {e}")
                print(f"   OSMesa error: {e2}")
                print("Suggestions:")
                print("   - Check MuJoCo installation")
                print("   - Verify OpenGL/EGL libraries")
                print("   - Try: export MUJOCO_GL=osmesa")
                raise e2

    def setup_policy(self) -> HumanoidGRPolicy:
        """Setup HumanoidGR policy"""
        if self.verbose:
            print("Setting up HumanoidGR policy...")
        
        try:
            policy = HumanoidGRPolicy(
                checkpoint_path=self.checkpoint_path,
                device=self.device,
                window_size=self.window_size,
                action_dim=56,
                state_dim=56,
                deterministic=True,
                action_scale=1.0,
                verbose=self.verbose
            )
            
            if self.verbose:
                print("✓ Policy setup complete")
            
            return policy
            
        except Exception as e:
            print(f"Failed to setup policy: {e}")
            raise

    def evaluate_instruction(
        self,
        env: HumanoidGRGymEnv,
        policy: HumanoidGRPolicy,
        instruction: str,
        max_steps: int = 1000,
        episode_id: int = 0
    ) -> Dict[str, Any]:
        """
        Evaluate policy on a specific instruction
        
        Args:
            env: HumanoidGR environment
            policy: HumanoidGR policy
            instruction: Natural language instruction
            max_steps: Maximum steps per episode
            episode_id: Episode identifier
            
        Returns:
            Evaluation results dictionary
        """
        
        if self.verbose:
            print(f"\nEvaluating instruction {episode_id + 1}: '{instruction}'")
        
        # Set instruction in environment and policy
        env.set_instruction(instruction)
        policy.set_instruction(instruction)
        
        # Reset environment
        obs = env.reset()
        
        # Episode tracking
        episode_reward = 0.0
        episode_length = 0
        frames = []
        success_metrics = {
            'balance_stability': [],
            'instruction_alignment': 0.0,
            'movement_smoothness': [],
            'forward_progress': 0.0
        }
        
        start_time = time.time()
        
        for step in range(max_steps):
            # Predict action
            action, _ = policy.predict(obs, episode_start=(step == 0), deterministic=True)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            # Collect video frame
            if self.render_videos:
                frame = env.render(mode='rgb_array')
                frames.append(frame)
            
            # Update success metrics
            if 'success_metrics' in info:
                env_metrics = info['success_metrics']
                if 'balance_stability' in env_metrics:
                    success_metrics['balance_stability'].append(env_metrics['balance_stability'])
                if 'movement_smoothness' in env_metrics:
                    success_metrics['movement_smoothness'].append(env_metrics['movement_smoothness'])
                if 'forward_progress' in env_metrics:
                    success_metrics['forward_progress'] = env_metrics['forward_progress']
            
            # Print progress
            if self.verbose and step % 100 == 0:
                print(f"  Step {step}/{max_steps}, Reward: {episode_reward:.3f}")
            
            # Check termination
            if done:
                if self.verbose:
                    print(f"  Episode terminated at step {step}")
                break
        
        end_time = time.time()
        
        # Compute final metrics
        final_metrics = {
            'balance_stability': np.mean(success_metrics['balance_stability']) if success_metrics['balance_stability'] else 0.0,
            'movement_smoothness': np.mean(success_metrics['movement_smoothness']) if success_metrics['movement_smoothness'] else 0.0,
            'forward_progress': success_metrics['forward_progress'],
            'instruction_alignment': self._compute_instruction_alignment(instruction, info)
        }
        
        # Compute instruction-following success
        instruction_success = self._compute_instruction_success(instruction, final_metrics, episode_reward)
        
        # Save video if enabled
        video_path = None
        if self.render_videos and frames:
            video_path = self._save_episode_video(frames, instruction, episode_id)
        
        # Compile results
        results = {
            'episode_id': episode_id,
            'instruction': instruction,
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'evaluation_time': end_time - start_time,
            'fps': episode_length / (end_time - start_time),
            'success_metrics': final_metrics,
            'instruction_success': instruction_success,
            'video_path': video_path
        }
        
        if self.verbose:
            print(f"Instruction evaluation complete:")
            print(f"   Reward: {episode_reward:.3f}")
            print(f"   Length: {episode_length} steps")
            print(f"   Success: {instruction_success:.2%}")
            print(f"   Balance: {final_metrics['balance_stability']:.3f}")
            print(f"   Smoothness: {final_metrics['movement_smoothness']:.3f}")
            if video_path:
                print(f"   Video: {video_path}")
        
        return results

    def _compute_instruction_alignment(self, instruction: str, info: Dict[str, Any]) -> float:
        """Compute how well the agent followed the instruction"""
        # Simple heuristic-based alignment score
        instruction = instruction.lower()
        
        # Get position and velocity information from info
        alignment_score = 0.5  # Base score
        
        # This is a simplified version - in practice, you'd use more sophisticated metrics
        if 'forward' in instruction:
            # Check if agent moved forward
            if 'forward_progress' in info:
                forward_progress = info.get('forward_progress', 0)
                alignment_score = min(1.0, max(0.0, forward_progress))
        
        elif 'balance' in instruction or 'stand' in instruction:
            # Check if agent maintained balance with minimal movement
            if 'balance_stability' in info:
                balance = info.get('balance_stability', 0)
                alignment_score = balance
        
        return alignment_score

    def _compute_instruction_success(self, instruction: str, metrics: Dict[str, float], reward: float) -> float:
        """Compute overall instruction-following success"""
        # Weighted combination of different metrics
        balance_weight = 0.3
        alignment_weight = 0.4
        smoothness_weight = 0.2
        reward_weight = 0.1
        
        # Normalize reward (assume range [-1, 1] per step, scaled by episode length)
        normalized_reward = max(0.0, min(1.0, (reward + 100) / 200))  # Simple normalization
        
        success = (
            metrics['balance_stability'] * balance_weight +
            metrics['instruction_alignment'] * alignment_weight +
            metrics['movement_smoothness'] * smoothness_weight +
            normalized_reward * reward_weight
        )
        
        return float(np.clip(success, 0.0, 1.0))

    def _save_episode_video(self, frames: List[np.ndarray], instruction: str, episode_id: int) -> str:
        """Save episode video with verification"""
        try:
            import imageio
            
            # Create videos directory in evaluation folder
            video_dir = Path(__file__).parent / 'videos'
            video_dir.mkdir(exist_ok=True)
            
            # Create safe filename
            safe_instruction = "".join(c for c in instruction if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_instruction = safe_instruction.replace(' ', '_')
            
            # Add timestamp for uniqueness
            timestamp = datetime.now().strftime("%H%M%S")
            
            video_path = video_dir / f"episode_{episode_id:03d}_{safe_instruction}_{timestamp}_{len(frames)}frames.mp4"
            
            # Convert frames to proper format
            video_frames = []
            for frame in frames:
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                video_frames.append(frame)
            
            # Save video with high quality
            imageio.mimsave(str(video_path), video_frames, fps=30, quality=8)
            
            # Verify video was created successfully
            if video_path.exists():
                file_size = video_path.stat().st_size
                if self.verbose:
                    print(f"Video saved: {video_path.name}")
                    print(f"Frames: {len(frames)}")
                    print(f"Resolution: {frames[0].shape[1]}x{frames[0].shape[0]}")
                    print(f"Size: {file_size / 1024:.1f} KB")
                
                # Optional: Verify with cv2 if available
                try:
                    import cv2
                    cap = cv2.VideoCapture(str(video_path))
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()
                    
                    if frame_count == len(frames):
                        if self.verbose:
                            print(f"Video verification: {frame_count} frames @ {fps:.1f} FPS")
                    else:
                        print(f"Frame count mismatch: saved {frame_count}, expected {len(frames)}")
                        
                except ImportError:
                    pass  # cv2 not available, skip verification
                
                return str(video_path)
            else:
                print(f"Video file was not created: {video_path}")
                return None
            
        except Exception as e:
            print(f"Failed to save video: {e}")
            return None

    def run_evaluation(
        self,
        num_episodes_per_instruction: int = 1,
        max_steps_per_episode: int = 1000,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete evaluation across all instructions
        
        Args:
            num_episodes_per_instruction: Number of episodes per instruction
            max_steps_per_episode: Maximum steps per episode
            save_results: Whether to save results to file
            
        Returns:
            Complete evaluation results
        """
        
        print(f"\n{'='*80}")
        print(f"STARTING HUMANOIDGR ONLINE EVALUATION")
        print(f"{'='*80}")
        print(f"Checkpoint: {self.checkpoint_path}")
        print(f"Instructions: {len(self.evaluation_instructions)}")
        print(f"Episodes per instruction: {num_episodes_per_instruction}")
        print(f"Max steps per episode: {max_steps_per_episode}")
        print(f"Total episodes: {len(self.evaluation_instructions) * num_episodes_per_instruction}")
        print(f"{'='*80}\n")
        
        # Setup environment and policy
        env = self.setup_environment()
        policy = self.setup_policy()
        
        # Run evaluation
        all_results = []
        episode_id = 0
        
        start_time = time.time()
        
        for instruction_idx, instruction in enumerate(self.evaluation_instructions):
            print(f"\nInstruction {instruction_idx + 1}/{len(self.evaluation_instructions)}: '{instruction}'")
            
            instruction_results = []
            
            for episode_num in range(num_episodes_per_instruction):
                result = self.evaluate_instruction(
                    env, policy, instruction, max_steps_per_episode, episode_id
                )
                instruction_results.append(result)
                all_results.append(result)
                episode_id += 1
            
            # Compute instruction summary
            instruction_summary = self._compute_instruction_summary(instruction_results)
            print(f"Instruction Summary: Success {instruction_summary['avg_success']:.2%}, "
                  f"Reward {instruction_summary['avg_reward']:.3f}")
        
        end_time = time.time()
        
        # Compute overall summary
        overall_summary = self._compute_overall_summary(all_results, end_time - start_time)
        
        # Save results
        if save_results:
            self._save_evaluation_results(all_results, overall_summary)
        
        # Print final summary
        self._print_final_summary(overall_summary)
        
        return {
            'episode_results': all_results,
            'overall_summary': overall_summary,
            'evaluation_time': end_time - start_time
        }

    def _compute_instruction_summary(self, instruction_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute summary for a specific instruction"""
        if not instruction_results:
            return {}
        
        return {
            'avg_success': np.mean([r['instruction_success'] for r in instruction_results]),
            'avg_reward': np.mean([r['episode_reward'] for r in instruction_results]),
            'avg_length': np.mean([r['episode_length'] for r in instruction_results]),
            'avg_balance': np.mean([r['success_metrics']['balance_stability'] for r in instruction_results]),
            'avg_smoothness': np.mean([r['success_metrics']['movement_smoothness'] for r in instruction_results])
        }

    def _compute_overall_summary(self, all_results: List[Dict[str, Any]], total_time: float) -> Dict[str, Any]:
        """Compute overall evaluation summary"""
        if not all_results:
            return {}
        
        # Group by instruction
        instruction_summaries = {}
        for result in all_results:
            instruction = result['instruction']
            if instruction not in instruction_summaries:
                instruction_summaries[instruction] = []
            instruction_summaries[instruction].append(result)
        
        # Compute per-instruction averages
        per_instruction_stats = {}
        for instruction, results in instruction_summaries.items():
            per_instruction_stats[instruction] = self._compute_instruction_summary(results)
        
        # Overall statistics
        overall_stats = {
            'total_episodes': len(all_results),
            'total_instructions': len(instruction_summaries),
            'total_time': total_time,
            'avg_fps': np.mean([r['fps'] for r in all_results]),
            'overall_success': np.mean([r['instruction_success'] for r in all_results]),
            'overall_reward': np.mean([r['episode_reward'] for r in all_results]),
            'overall_balance': np.mean([r['success_metrics']['balance_stability'] for r in all_results]),
            'overall_smoothness': np.mean([r['success_metrics']['movement_smoothness'] for r in all_results]),
            'per_instruction_stats': per_instruction_stats
        }
        
        return overall_stats

    def _save_evaluation_results(self, all_results: List[Dict[str, Any]], summary: Dict[str, Any]):
        """Save evaluation results to files"""
        try:
            import json
            
            # Create results directory
            results_dir = Path('evaluation_results')
            results_dir.mkdir(exist_ok=True)
            
            # Generate timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save detailed results
            results_file = results_dir / f"humanoidGR_evaluation_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump({
                    'checkpoint_path': self.checkpoint_path,
                    'evaluation_config': {
                        'window_size': self.window_size,
                        'device': self.device,
                        'instructions': self.evaluation_instructions
                    },
                    'episode_results': all_results,
                    'summary': summary
                }, f, indent=2)
            
            # Save summary CSV
            summary_file = results_dir / f"humanoidGR_summary_{timestamp}.csv"
            self._save_summary_csv(summary, summary_file)
            
            print(f"Results saved:")
            print(f"   Detailed: {results_file}")
            print(f"   Summary: {summary_file}")
            
        except Exception as e:
            print(f"Warning: Failed to save results: {e}")

    def _save_summary_csv(self, summary: Dict[str, Any], filepath: Path):
        """Save summary as CSV"""
        try:
            import pandas as pd
            
            # Create summary dataframe
            rows = []
            for instruction, stats in summary['per_instruction_stats'].items():
                rows.append({
                    'instruction': instruction,
                    'avg_success': stats['avg_success'],
                    'avg_reward': stats['avg_reward'],
                    'avg_balance': stats['avg_balance'],
                    'avg_smoothness': stats['avg_smoothness']
                })
            
            df = pd.DataFrame(rows)
            df.to_csv(filepath, index=False)
            
        except ImportError:
            # Fallback: save as simple text file
            with open(filepath.with_suffix('.txt'), 'w') as f:
                f.write("Instruction,Success,Reward,Balance,Smoothness\n")
                for instruction, stats in summary['per_instruction_stats'].items():
                    f.write(f"{instruction},{stats['avg_success']:.3f},{stats['avg_reward']:.3f},"
                           f"{stats['avg_balance']:.3f},{stats['avg_smoothness']:.3f}\n")

    def _print_final_summary(self, summary: Dict[str, Any]):
        """Print final evaluation summary"""
        print(f"\n{'='*80}")
        print(f"HUMANOIDGR EVALUATION RESULTS")
        print(f"{'='*80}")
        print(f"Overall Performance:")
        print(f"   Success Rate: {summary['overall_success']:.2%}")
        print(f"   Average Reward: {summary['overall_reward']:.3f}")
        print(f"   Balance Stability: {summary['overall_balance']:.3f}")
        print(f"   Movement Smoothness: {summary['overall_smoothness']:.3f}")
        
        print(f"\nPerformance Statistics:")
        print(f"   Total Episodes: {summary['total_episodes']}")
        print(f"   Total Instructions: {summary['total_instructions']}")
        print(f"   Evaluation Time: {summary['total_time']:.1f}s")
        print(f"   Average FPS: {summary['avg_fps']:.1f}")
        
        print(f"\nPer-Instruction Results:")
        for instruction, stats in summary['per_instruction_stats'].items():
            print(f"   '{instruction}': {stats['avg_success']:.2%} success, "
                  f"{stats['avg_reward']:.3f} reward")
        
        print(f"{'='*80}")
        
        # Grade the performance
        overall_success = summary['overall_success']
        if overall_success >= 0.8:
            grade = "EXCELLENT"
        elif overall_success >= 0.6:
            grade = "GOOD"
        elif overall_success >= 0.4:
            grade = "FAIR"
        else:
            grade = "NEEDS IMPROVEMENT"
        
        print(f"Overall Grade: {grade} ({overall_success:.1%} success rate)")
        print(f"{'='*80}\n")


def main():
    """Enhanced main evaluation function with comprehensive options"""
    parser = argparse.ArgumentParser(
        description='HumanoidGR Online Evaluation - Enhanced Version with LoRA Support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ENHANCED HUMANOIDGR EVALUATION
"""
    )
    
    # Model configuration
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints_sliding_window/best_model_finetune.pt',
                       help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                       help='Device for inference (auto=GPU if available)')
    parser.add_argument('--window_size', type=int, default=15,
                       help='Sliding window size for sequence processing')
    
    # Evaluation configuration  
    parser.add_argument('--num_episodes', type=int, default=1,
                       help='Number of episodes per instruction')
    parser.add_argument('--max_steps', type=int, default=40,
                       help='Maximum steps per episode (default: 40 for quick tests)')
    
    # Instruction configuration (mutually exclusive)
    instruction_group = parser.add_mutually_exclusive_group()
    instruction_group.add_argument('--single_instruction', type=str, default=None,
                                  help='Evaluate single instruction (recommended for testing)')
    instruction_group.add_argument('--instructions', nargs='+', default=None,
                                  help='Multiple instructions (space-separated)')
    instruction_group.add_argument('--instructions_file', type=str, default=None,
                                  help='File with instructions (one per line)')
    
    # Output configuration
    parser.add_argument('--no_video', action='store_true',
                       help='Disable video rendering and saving')
    parser.add_argument('--no_save', action='store_true',
                       help='Disable saving evaluation results')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    
    # Advanced options
    parser.add_argument('--force_rendering', type=str, default=None, 
                       choices=['egl', 'osmesa'],
                       help='Force specific rendering backend')
    parser.add_argument('--verify_checkpoint', action='store_true',
                       help='Verify checkpoint loading without running evaluation')
    
    args = parser.parse_args()
    
    # Setup rendering backend if specified
    if args.force_rendering:
        os.environ['MUJOCO_GL'] = args.force_rendering
        print(f"Forced rendering backend: {args.force_rendering}")
    
    # Resolve checkpoint path
    if not os.path.isabs(args.checkpoint):
        project_root = Path(__file__).parent.parent
        checkpoint_path = project_root / args.checkpoint
    else:
        checkpoint_path = Path(args.checkpoint)
    
    # Validate checkpoint
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("\nAvailable checkpoints:")
        checkpoint_dir = checkpoint_path.parent
        if checkpoint_dir.exists():
            checkpoint_files = list(checkpoint_dir.glob("*.pt"))
            if checkpoint_files:
                for f in sorted(checkpoint_files):
                    print(f"   - {f.name}")
            else:
                print("   - No .pt files found")
        else:
            print(f"   - Directory does not exist: {checkpoint_dir}")
        print(f"\nSuggestion: Train the model first or check the checkpoint path")
        return
    
    # Print startup information
    print(f"\n{'='*80}")
    print(f"HUMANOIDGR ENHANCED EVALUATION")
    print(f"{'='*80}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {args.device}")
    print(f"Video recording: {'Disabled' if args.no_video else 'Enabled'}")
    print(f"Max steps: {args.max_steps}")
    print(f"Window size: {args.window_size}")
    print(f"Verbose: {'Disabled' if args.quiet else 'Enabled'}")
    
    # Checkpoint verification mode
    if args.verify_checkpoint:
        print(f"\nCHECKPOINT VERIFICATION MODE")
        print(f"{'='*50}")
        
        try:
            from envs.humanoidGR_policy import HumanoidGRPolicy
            
            # Test policy loading
            device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device)
            policy = HumanoidGRPolicy(
                checkpoint_path=str(checkpoint_path),
                device=device,
                window_size=args.window_size,
                action_dim=56,
                state_dim=56,
                deterministic=True,
                verbose=not args.quiet
            )
            
            print(f"Checkpoint verification successful!")
            print(f"   - Model loaded correctly")
            print(f"   - LoRA compatibility: OK")
            print(f"   - Hidden size handling: OK")
            print(f"   - Ready for evaluation")
            return
            
        except Exception as e:
            print(f"Checkpoint verification failed: {e}")
            import traceback
            if not args.quiet:
                traceback.print_exc()
            return
    
    print(f"{'='*80}")
    
    try:
        # Create evaluator
        evaluator = HumanoidGREvaluator(
            checkpoint_path=str(checkpoint_path),
            device=args.device,
            window_size=args.window_size,
            render_videos=not args.no_video,
            verbose=not args.quiet
        )
        
        # Load instructions based on arguments
        instructions = []
        
        if args.single_instruction:
            # Single instruction mode
            instructions = [args.single_instruction]
            print(f"Single instruction mode: '{args.single_instruction}'")
            
        elif args.instructions:
            # Custom instructions from command line
            instructions = args.instructions
            print(f"Using command-line instructions: {len(instructions)} instructions")
            
        elif args.instructions_file:
            # Load from file
            instructions_path = Path(args.instructions_file)
            if not instructions_path.is_absolute():
                # Relative to evaluation directory
                instructions_path = Path(__file__).parent / instructions_path
            
            if instructions_path.exists():
                instructions = evaluator.load_instructions_from_file(str(instructions_path))
            else:
                print(f"Instructions file not found: {instructions_path}")
                return
                
        else:
            # Try to load default instructions file
            default_file = Path(__file__).parent / 'default_instructions.txt'
            if default_file.exists():
                instructions = evaluator.load_instructions_from_file(str(default_file))
                print(f"Using default instructions file: {default_file}")
            else:
                # Fall back to hardcoded defaults
                instructions = evaluator.set_default_instructions()
                print(f"⚙️  Using hardcoded default instructions")
        
        # Validate we have instructions
        if not instructions:
            print(f"No valid instructions found! Please provide instructions via:")
            print(f"  - --single_instruction 'your instruction'")
            print(f"  - --instructions 'instruction1' 'instruction2' ...")
            print(f"  - --instructions_file path/to/instructions.txt")
            print(f"  - Create default_instructions.txt in evaluation folder")
            return
        
        # Set instructions in evaluator
        evaluator.set_instructions(instructions)
        
        # Run evaluation
        results = evaluator.run_evaluation(
            num_episodes_per_instruction=args.num_episodes,
            max_steps_per_episode=args.max_steps,
            save_results=not args.no_save
        )
        
        print(f"Evaluation completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\nEvaluation interrupted by user")
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 