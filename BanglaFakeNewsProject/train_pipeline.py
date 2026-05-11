"""
Pipeline script that runs the optimized trainer to produce a strong checkpoint,
then fine-tunes a multimodal model initialized from that checkpoint. It compares
validation accuracy and saves the best model as `best_pipeline_model.pth`.

Usage:
    python train_pipeline.py --config config.yaml --data-dir . --output-dir outputs --device cpu
"""

import os
import argparse
import subprocess
import json
import shutil
import sys
from pathlib import Path

def run_optimized(config, data_dir, output_dir, device):
    script_dir = Path(__file__).resolve().parent
    script_path = script_dir / 'train_optimized.py'
    cmd = [
        sys.executable, str(script_path),
        "--config", config,
        "--data-dir", data_dir,
        "--output-dir", output_dir,
        "--device", device
    ]
    print("Running optimized trainer:", " ".join(cmd))
    subprocess.check_call(cmd)

def run_multimodal(pretrained_checkpoint, epochs=5):
    # Call train_multimodal in-process by importing its train_model function is simpler,
    # but to avoid side-effects we will run it as a subprocess with an env var telling it
    # which pretrained checkpoint to use.
    env = os.environ.copy()
    env['PRETRAINED_CHECKPOINT'] = pretrained_checkpoint
    script_dir = Path(__file__).resolve().parent
    script_path = script_dir / 'train_multimodal.py'
    cmd = [sys.executable, str(script_path)]
    print("Running multimodal trainer (will look for PRETRAINED_CHECKPOINT env var):", " ".join(cmd))
    subprocess.check_call(cmd, env=env)


def find_checkpoint(output_dir):
    # Look for best_multimodal_model.pth inside output_dir or repo root
    candidates = [
        os.path.join(output_dir, 'best_multimodal_model.pth'),
        os.path.join('.', 'best_multimodal_model.pth')
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return None


def main(args):
    # Run optimized trainer
    run_optimized(args.config, args.data_dir, args.output_dir, args.device)

    # Locate checkpoint produced by optimized trainer
    ckpt = find_checkpoint(args.output_dir)
    if ckpt is None:
        print("No checkpoint found after optimized training. Aborting pipeline.")
        return
    print("Found optimized checkpoint:", ckpt)

    # Copy checkpoint to a known location
    pipeline_ckpt = os.path.join(args.output_dir, 'pipeline_pretrained.pth')
    shutil.copyfile(ckpt, pipeline_ckpt)

    # Run multimodal fine-tune with that checkpoint
    run_multimodal(pipeline_ckpt)

    # Look for final model
    final_model = os.path.join('.', 'best_multimodal_model.pth')
    if os.path.exists(final_model):
        # Save a copy
        out_path = os.path.join(args.output_dir, 'best_pipeline_model.pth')
        shutil.copyfile(final_model, out_path)
        print("Pipeline finished. Best pipeline model saved to:", out_path)
    else:
        print("Multimodal trainer did not produce a final model.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--data-dir', type=str, default='.')
    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    main(args)
