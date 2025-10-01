Core training framework for research and model development.

Structure
- configs/: YAML configs per experiment/model
- data/: Simple dataset/dataloader helpers
- models/: Model defs and registry pattern (timm/transformers compatible)
- trainer/: Training loop, optimizers, metrics
- utils/: Seed, logging, config helpers, checkpoint I/O

Usage (example)
- Edit configs/vit_b16.yaml for your dataset paths and parameters
- Run: python tvb-ai/scripts/train.py --config tvb-ai/core/configs/vit_b16.yaml
- Outputs under tvb-ai/experiments/<run_name>/{checkpoints,logs}

Notes
- PyTorch 2.6.0 (CUDA 12.6), timm, transformers, accelerate
- The core package is independent from tvb-server. Exported checkpoints can be loaded later by API workers.

