import os
import io
import argparse
import csv
import lmdb
from pickle import dumps, loads
import numpy as np
import torch
import h5py
from pathlib import Path
from tqdm import tqdm

# Import video processor for preprocessing during dataset creation
import sys
sys.path.append('.')
from GPTmodel.video_processor import VideoProcessor

def load_text_annotations(csv_path):
    """Load text annotations from CSV file"""
    annotations = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                # Remove .mp4 extension and clean up text
                snippet_name = row[0].replace('.mp4', '')
                text = row[1].strip()
                annotations[snippet_name] = text
    return annotations

def get_joint_positions(proprioceptive_data, joint_pos_indices):
    """Extract joint positions from proprioceptive data using indices"""
    return proprioceptive_data[:, joint_pos_indices]  # Shape: (timesteps, 56)

def create_dummy_tokens(text):
    """Create simple dummy tokens from text for compatibility (raw text stored as string)"""
    # For now, just store the raw text as a simple token representation
    # The dataset can handle raw text strings
    return text

def save_to_lmdb(output_dir, hdf5_files, text_annotations, video_dir):
    """Convert MoCapAct HDF5 data to LMDB format with EFFICIENTLY stored video frames"""
    
    # Initialize video processor for frame extraction
    video_processor = VideoProcessor(
        video_base_path=video_dir,
        image_size=(224, 224),
        normalize=False  # CRITICAL: Don't normalize, keep as uint8 [0,255]
    )
    print(f"Video processor initialized for EFFICIENT preprocessing")
    
    # Reasonable map size for efficiently stored video frames
    env = lmdb.open(output_dir, map_size=int(5e10), readonly=False, lock=False)  # 50GB max (much smaller)
    
    with env.begin(write=True) as txn:
        cur_step = 0
        cur_episode = 0
        processed_snippets = 0
        total_annotations = len(text_annotations)
        successful_video_processing = 0
        failed_video_processing = 0
        total_frame_size = 0  # Track actual storage size
        
        print(f"Processing {len(hdf5_files)} HDF5 files to find {total_annotations} annotated snippets...")
        print("Video frames will be preprocessed and stored EFFICIENTLY at timestep level")
        print("Using uint8 format for 4x storage reduction")
        
        for hdf5_file in tqdm(hdf5_files, desc="Processing HDF5 files"):
            with h5py.File(hdf5_file, 'r') as f:
                # Get joint position indices (should be the same for all files)
                joint_pos_indices = f['observable_indices/walker/joints_pos'][...]
                
                # Get all snippet names in this file (format: CMU_XXX_XX-start-end)
                all_keys = list(f.keys())
                snippet_names = [key for key in all_keys if key.startswith('CMU_') and '-' in key and key.count('-') == 2]
                
                for snippet_name in snippet_names:
                    # Check if this snippet has text annotation
                    if snippet_name not in text_annotations:
                        continue  # Skip snippets without text annotations
                    
                    text_desc = text_annotations[snippet_name]
                    video_filename = f"{snippet_name}.mp4"
                    
                    # Process only the first 10 episodes (0-9) for this snippet
                    for episode_idx in range(10):
                        episode_key = f"{snippet_name}/{episode_idx}"
                        if episode_key not in f:
                            # Skip missing episodes but don't print warning for every one
                            continue
                        
                        episode_group = f[episode_key]
                        
                        # Get trajectory data
                        mean_actions = episode_group['mean_actions'][...]  # Shape: (T, 56)
                        proprioceptive = episode_group['observations/proprioceptive'][...]  # Shape: (T+1, 2868)
                        
                        # Extract joint positions (take first T timesteps to match actions)
                        joint_positions = get_joint_positions(proprioceptive[:-1], joint_pos_indices)  # Shape: (T, 56)
                        
                        episode_length = mean_actions.shape[0]
                        
                        # CRITICAL: Preprocess video frames for this episode EFFICIENTLY
                        try:
                            # Load video frames for the entire episode trajectory
                            video_frames = video_processor.load_video_sequence_for_trajectory(
                                video_filename, episode_length, episode_length
                            )  # Shape: (T, 3, 224, 224) as uint8 [0,255]
                            
                            if video_frames.shape[0] != episode_length:
                                print(f"  Video frame count mismatch for {snippet_name}/{episode_idx}: "
                                      f"expected {episode_length}, got {video_frames.shape[0]}")
                                # Skip this episode if frame count doesn't match
                                continue
                            
                            successful_video_processing += 1
                            
                        except Exception as e:
                            print(f"Failed to process video for {snippet_name}/{episode_idx}: {e}")
                            failed_video_processing += 1
                            # Skip this episode if video processing fails
                            continue
                        
                        # Store episode-level data (instruction only, no video path needed)
                        txn.put(f'inst_token_{cur_episode}'.encode(), dumps(text_desc))
                        
                        # Store episode metadata
                        txn.put(f'episode_length_{cur_episode}'.encode(), dumps(episode_length))
                        txn.put(f'episode_start_step_{cur_episode}'.encode(), dumps(cur_step))
                        
                        # Save each timestep in this episode WITH EFFICIENTLY STORED VIDEO FRAMES
                        for t in range(episode_length):
                            # Store step-level metadata
                            txn.put(f'cur_episode_{cur_step}'.encode(), dumps(cur_episode))
                            txn.put(f'done_{cur_step}'.encode(), dumps(t == episode_length - 1))
                            txn.put(f'step_in_episode_{cur_step}'.encode(), dumps(t))  # Position within episode
                            
                            # Store humanoid-specific state (joint positions, 56-dim)
                            txn.put(f'joint_pos_{cur_step}'.encode(), dumps(joint_positions[t]))
                            
                            # Store actions (56-dim for humanoid)
                            txn.put(f'actions_{cur_step}'.encode(), dumps(mean_actions[t]))
                            
                            # CRITICAL: Store preprocessed video frame EFFICIENTLY
                            # Convert to uint8 and store as numpy array for 4x size reduction
                            if video_frames.dtype == torch.float32:
                                # Convert from float32 [0,1] to uint8 [0,255]
                                frame_data = (video_frames[t] * 255).clamp(0, 255).byte().cpu().numpy()
                            else:
                                # Already uint8, just convert to numpy
                                frame_data = video_frames[t].cpu().numpy()
                            
                            # Ensure uint8 format
                            frame_data = frame_data.astype(np.uint8)  # Shape: (3, 224, 224) as uint8
                            
                            # Track storage size for monitoring
                            frame_size = frame_data.nbytes  # Should be 3*224*224 = 150,528 bytes ≈ 0.15 MB
                            total_frame_size += frame_size
                            
                            txn.put(f'video_frame_{cur_step}'.encode(), dumps(frame_data))
                            
                            cur_step += 1
                        
                        cur_episode += 1
                        
                        # Progress reporting every 100 episodes
                        if cur_episode % 100 == 0:
                            avg_frame_size = total_frame_size / max(cur_step, 1)
                            estimated_total_size = total_frame_size / (1024**3)  # GB
                            print(f"  Progress: {cur_episode} episodes, {cur_step} timesteps")
                            print(f"  Avg frame size: {avg_frame_size:.0f} bytes ({avg_frame_size/1024:.1f} KB)")
                            print(f"  Current storage: {estimated_total_size:.2f} GB")
                    
                    processed_snippets += 1
        
        # Save final statistics
        txn.put('total_steps'.encode(), dumps(cur_step))
        txn.put('total_episodes'.encode(), dumps(cur_episode))
        txn.put('processed_snippets'.encode(), dumps(processed_snippets))
        txn.put('successful_video_processing'.encode(), dumps(successful_video_processing))
        txn.put('failed_video_processing'.encode(), dumps(failed_video_processing))
        txn.put('total_frame_storage_bytes'.encode(), dumps(total_frame_size))
    
    env.close()
    
    # Final storage analysis
    final_size_gb = total_frame_size / (1024**3)
    avg_frame_size = total_frame_size / max(cur_step, 1)
    
    print(f"Dataset creation completed!")
    print(f"Processed snippets: {processed_snippets}/{total_annotations}")
    print(f"Total episodes: {cur_episode}")
    print(f"Total steps: {cur_step}")
    print(f"Video processing: {successful_video_processing} successful, {failed_video_processing} failed")
    print(f"Storage efficiency:")
    print(f"Total frame storage: {final_size_gb:.2f} GB")
    print(f"Average frame size: {avg_frame_size:.0f} bytes ({avg_frame_size/1024:.1f} KB)")
    
    if processed_snippets < total_annotations:
        missing = total_annotations - processed_snippets
        print(f"  WARNING: {missing} annotated snippets were not found in HDF5 files!")
        print("This might explain why the LMDB is smaller than expected.")
    
    if failed_video_processing > 0:
        print(f"  WARNING: {failed_video_processing} episodes failed video processing and were skipped!")

def main():
    parser = argparse.ArgumentParser(description="Convert MoCapAct HDF5 data to LMDB format with EFFICIENTLY stored video frames")
    parser.add_argument("--hdf5_dir", default="./data/action/small", type=str, help="Directory containing HDF5 files")
    parser.add_argument("--output_dir", default="./data/humanoid_lmdb", type=str, help="Output LMDB directory") 
    parser.add_argument("--text_csv", default="./data/text/125.csv", type=str, help="CSV file with text annotations")
    parser.add_argument("--video_dir", default="./data/video/snippet_videos", type=str, help="Directory containing video files")
    parser.add_argument("--device", default="cuda:0", type=str, help="Device for CLIP model")
    
    args = parser.parse_args()
    
    # Create output directory 
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load text annotations
    print("Loading text annotations...")
    text_annotations = load_text_annotations(args.text_csv)
    print(f"Loaded {len(text_annotations)} text annotations")
    
    # Find all HDF5 files
    hdf5_files = list(Path(args.hdf5_dir).glob("*.hdf5"))
    print(f"Found {len(hdf5_files)} HDF5 files")
    
    if len(hdf5_files) == 0:
        print("ERROR: No HDF5 files found! Please check the hdf5_dir path.")
        return
    
    # Verify video directory exists
    if not os.path.exists(args.video_dir):
        print(f"ERROR: Video directory {args.video_dir} does not exist!")
        print("Video preprocessing requires access to video files.")
        return
    
    # Convert to LMDB with EFFICIENT video preprocessing
    print("Converting to LMDB format with EFFICIENT video frame preprocessing...")
    print("Using uint8 format for 4x storage reduction compared to float32")
    save_to_lmdb(args.output_dir, hdf5_files, text_annotations, args.video_dir)

if __name__ == '__main__':
    main() 