# HumanoidGR: Vision-Language-Action Model for 56-DOF Humanoid Robots


**A Vision-Language-Action (VLA) model for 56-DOF humanoid robot control**

</div>

## Overview

HumanoidGR is an advanced GPT-based autoregressive VLA model designed specifically for 56-DOF humanoid robots. It features hierarchical action encoding, multi-modal fusion, and sophisticated temporal modeling for precise humanoid robot control with natural language instruction following.

###  Key Features

- **56-DOF Humanoid Control**: Specialized hierarchical encoding for complex humanoid anatomy
- **Video Prediction**: Forward dynamics modeling with proper temporal alignment (t→t+1)
- **Language Understanding**: Natural language instruction following via CLIP integration
- **Efficient Training**: Sliding window approach with LoRA fine-tuning
- **Production Ready**: Comprehensive evaluation in DMControl Mujoco simulation

## Architecture

### Model Components

```
HumanoidGR Architecture:
├──  Video Encoder (MAE + Perceiver Resampler)
├──  Language Encoder (CLIP)  
├──  Hierarchical Action Encoder (56-DOF specific)
├──  GPT Transformer Core (with LoRA fine-tuning)
├──  Action Decoder (Hierarchical by body parts)
└──  Video Decoder (Future frame prediction)
```

### Hierarchical Action Encoding

Our 56-DOF humanoid action space is organized anatomically:

```python
Body Parts Organization:
├── Left Leg: joints [0:7]      (7 DOF)
├── Right Leg: joints [7:14]    (7 DOF) 
├── Torso/Spine: joints [14:23] (9 DOF)
├── Head/Neck: joints [23:32]   (9 DOF)
├── Left Arm: joints [32:44]    (12 DOF)
└── Right Arm: joints [44:56]   (12 DOF)
```

### Sequence Format

Each timestep contains:
```
(language, joint_pos_t, video_frame_t, patch_tokens, [video_queries], [action_queries])
```

##  Dataset Creation (`mocapact2lmdb.py`)

### Dataset Pipeline

The dataset creation process converts MoCapAct HDF5 data to LMDB format with efficient video preprocessing:

```bash
python mocapact2lmdb.py \
    --hdf5_dir ./data/action/small \
    --output_dir ./data/humanoid_lmdb \
    --text_csv ./data/text/125.csv \
    --video_dir ./data/video/snippet_videos
```

### Key Features

1. **Efficient Video Storage**: Converts video frames to uint8 format (4x size reduction)
2. **Text Annotation Integration**: Links natural language descriptions with trajectories
3. **Joint Position Extraction**: Extracts 56-DOF joint positions from proprioceptive data
4. **Episode-based Organization**: Organizes data by episodes with proper temporal alignment

### Data Format

Each episode contains:
- `joint_pos_{step}`: Joint positions (56-DOF) at timestep
- `actions_{step}`: Action targets (56-DOF) at timestep  
- `video_frame_{step}`: Preprocessed RGB frame (3×224×224, uint8)
- `inst_token_{episode}`: Natural language instruction
- `cur_episode_{step}`: Episode identifier

##  Training (`train/train_humanoidGR.py`)

### Two-Phase Training Strategy

#### Phase 1: Warmup (2 epochs)
- ❄️ Freeze CLIP, MAE, and GPT transformer
-  Train embeddings, decoders, and hierarchical components
-  Lower learning rate (lr × 0.1)
-  ~6M trainable parameters

#### Phase 2: LoRA Fine-tuning (20 epochs)  
- ❄️ Keep CLIP and MAE frozen
-  Apply LoRA to GPT attention layers
-  Full learning rate
-  ~2M additional LoRA parameters

### Sliding Window Training

The training uses a **15-timestep sliding window** approach:

1. **Input Sequence**: `[t-14:t]` (video frames + joint positions + language)
2. **Prediction**: Action at timestep `t` and video frame at `t+1`
3. **Temporal Alignment**: Critical t→t+1 prediction for forward dynamics
4. **Memory Efficiency**: ~15x more efficient than full episode processing

### Usage

```bash
# Single-GPU training
python train/train_humanoidGR.py \
    --lmdb_dir data/humanoid_lmdb \
    --video_base_path data/video/snippet_videos \
    --batch_size 2 \
    --warmup_epochs 2 \
    --finetune_epochs 20

# Multi-GPU distributed training (Recommended)
accelerate launch --config_file train/accelerate_config.yaml \
    train/train_humanoidGR.py \
    --lmdb_dir data/humanoid_lmdb \
    --per_gpu_batch_size 1 \
    --gradient_accumulation_steps 8
```

### Key Training Components

1. **HumanoidLMDBDataset**: Efficient sliding window dataset loader
2. **Hierarchical Loss Computation**: Body-part-specific action losses
3. **Advanced Parameter Groups**: Different learning rates for LoRA vs base parameters
4. **SwanLab Integration**: Comprehensive experiment tracking
5. **Sophisticated Attention Masking**: Prevents information leakage during training

##  Environment & Evaluation (`envs/` & `evaluation/`)

### Environment Components

#### 1. **HumanoidGR Environment** (`envs/humanoidGR_env.py`)
- Integrates with DMControl's CMUHumanoidPositionControlledV2020
- Provides sliding window video sequences (224×224 RGB)
- Manages natural language instruction processing
- Vision-language-action observation space

#### 2. **HumanoidGR Task** (`envs/humanoidGR_task.py`)
- Natural language instruction-based reward computation
- Hierarchical reward system for different body parts
- Episode management with instruction variation
- Balance, smoothness, and instruction-following metrics

#### 3. **HumanoidGR Policy** (`envs/humanoidGR_policy.py`)
- Interfaces trained HumanoidGR model with evaluation
- Sliding window sequence management
- Real-time action prediction with vision-language-action
- Compatible with standard evaluation protocols

### Evaluation Instructions

The system evaluates across 10 natural language instructions:

1. **"walk forward naturally"** - Natural forward locomotion
2. **"walk backward slowly"** - Controlled backward movement  
3. **"turn left while walking"** - Left turning with locomotion
4. **"turn right while walking"** - Right turning with locomotion
5. **"stand still and balance"** - Static balance maintenance
6. **"walk in a circle"** - Circular motion patterns
7. **"walk with long strides"** - Extended stride locomotion
8. **"walk with short steps"** - Compact step patterns
9. **"walk and stop periodically"** - Intermittent movement
10. **"walk forward then turn around"** - Sequential complex motions

### Evaluation Usage

```bash
# Basic evaluation
python evaluation/evaluate_humanoidGR.py \
    --checkpoint checkpoints_sliding_window/best_model_finetune.pt

# Extended evaluation with custom instructions
python evaluation/evaluate_humanoidGR.py \
    --checkpoint checkpoints_sliding_window/best_model_finetune.pt \
    --instructions "jump forward" "walk sideways" "dance in place" \
    --num_episodes 3 \
    --max_steps 2000
```

### Evaluation Metrics

1. **Instruction Following Success Rate** - How well the robot follows natural language commands
2. **Balance Stability** - Maintenance of upright posture during motion
3. **Movement Smoothness** - Quality of motion without jerky movements
4. **Forward Progress** - Spatial displacement achievements
5. **Energy Efficiency** - Minimization of excessive joint torques

##  Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU training)

### Quick Setup

```bash
# Clone repository
git clone https://github.com/your-repo/HumanoidGR.git
cd HumanoidGR

# Create conda environment
conda create -n humanoid python=3.8
conda activate humanoid

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate tensorboard
pip install clip-by-openai lmdb opencv-python imageio
pip install dm_control mujoco  # For evaluation
pip install swanlab  # For experiment tracking

# Install additional requirements
pip install -r requirements.txt
```

##  Project Structure

```
HumanoidGR/
├──  GPTmodel/                    # Core model implementation
│   ├── humanoidGR.py              # Main model (v2.0 optimized)
│   ├── action_encoder.py          # Hierarchical action encoding
│   ├── video_processor.py         # Video processing utilities
│   ├── vision_transformer.py      # Vision transformer components
│   ├── transformer_utils.py       # Transformer utilities
│   └── pretrain/                  # Pretrained weights
├──  train/                      # Training scripts
│   ├── train_humanoidGR.py       # Main distributed training script
│   ├── HumanoidLMDBDataset.py     # Sliding window dataset
│   ├── config_humanoid.json      # Training configuration
│   ├── optimal_training_config.json # Optimal training settings
│   └── accelerate_config.yaml    # Distributed training config
├──  evaluation/                 # Evaluation tools
│   ├── evaluate_humanoidGR.py    # Main evaluation script
│   ├── EVALUATION_README.md      # Detailed evaluation guide
│   ├── EVALUATION_INSTRUCTIONS.md # Evaluation instructions
│   ├── SUCCESS_FACTORS_SUMMARY.md # Success factors
│   └── default_instructions.txt  # Default instruction set
├──  envs/                      # Environment components
│   ├── humanoidGR_env.py         # DMControl environment wrapper
│   ├── humanoidGR_task.py        # Task implementation
│   ├── humanoidGR_policy.py      # Policy wrapper
│   ├── dm_control_wrapper.py     # DMControl integration
│   └── tracking.py               # Motion tracking utilities
├── 📄 mocapact2lmdb.py           # Dataset conversion script
├── 📄 requirements.txt           # Python dependencies
└── 📄 README.md                  # This file
```

##  Quick Start

### 1. Dataset Creation

```bash
# Convert MoCapAct data to LMDB format
python mocapact2lmdb.py \
    --hdf5_dir /path/to/mocapact/hdf5 \
    --output_dir data/humanoid_lmdb \
    --text_csv /path/to/text/annotations.csv \
    --video_dir /path/to/snippet/videos
```

### 2. Training

```bash
# Start distributed training
accelerate launch --config_file train/accelerate_config.yaml \
    train/train_humanoidGR.py \
    --lmdb_dir data/humanoid_lmdb \
    --per_gpu_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --warmup_epochs 2 \
    --finetune_epochs 20
```

### 3. Evaluation

```bash
# Evaluate trained model
python evaluation/evaluate_humanoidGR.py \
    --checkpoint checkpoints_sliding_window/best_model_finetune.pt \
    --num_episodes 3 \
    --max_steps 1000
```

##  Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size and sequence length
--per_gpu_batch_size 1 --gradient_accumulation_steps 16
```

#### 2. Dataset Loading Errors
```bash
# Check LMDB format and permissions
python -c "import lmdb; env = lmdb.open('data/humanoid_lmdb'); print('OK')"
```

#### 3. DMControl Installation Issues
```bash
pip install dm_control[all]
# On Linux: sudo apt-get install libosmesa6-dev
```

#### 4. Video Processing Errors
```bash
# Install video dependencies
pip install imageio[ffmpeg]
```

##  Novel Contributions

### Technical Innovations

1. **Vision-Language-Action Integration**: First application of sliding window VLA to 56-DOF humanoid control
2. **Hierarchical Action Encoding**: Body-part-specific action prediction for complex humanoid anatomy
3. **Temporal Alignment**: Proper t→t+1 prediction methodology for forward dynamics
4. **Multi-modal Fusion**: Vision + Language + Proprioception integration with attention masking
5. **Efficient Training**: Sliding window approach with LoRA fine-tuning for memory efficiency

### Key Improvements Over Previous Methods

- **Proper Temporal Prediction**: Fixed critical video prediction bug (t→t vs t→t+1)
- **Enhanced Action Chunking**: GR1-style base + chunk-specific queries
- **Sophisticated Attention**: Prevents information leakage during training
- **Efficient Parameter Loading**: Maximizes compatible pretrained weight usage
