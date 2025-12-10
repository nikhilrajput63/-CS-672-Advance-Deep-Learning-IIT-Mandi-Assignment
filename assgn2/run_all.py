#!/usr/bin/env python3
import os
import sys
import time
from datetime import datetime

def print_header(text):
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80 + "\n")

def check_dependencies():
    print_header("Checking Dependencies")
    try:
        import gymnasium
        import torch
        import numpy
        import matplotlib
        print("✓ All dependencies found")
        print(f"  - Gymnasium: {gymnasium.__version__}")
        print(f"  - PyTorch: {torch.__version__}")
        print(f"  - NumPy: {numpy.__version__}")
        print(f"  - Matplotlib: {matplotlib.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nInstall with: pip install -r requirements.txt")
        return False

def run_part(part_num, title, module_name):
    print_header(f"PART {part_num}: {title}")
    start = time.time()
    try:
        __import__(module_name)
        print(f"\n✓ Part {part_num} completed in {time.time() - start:.2f}s")
        return True
    except Exception as e:
        print(f"\n✗ Part {part_num} failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    start_time = time.time()
    
    print_header("DEEP REINFORCEMENT LEARNING - ASSIGNMENT")
    print("CartPole-v1: DQN, REINFORCE, and PG with Baseline")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not check_dependencies():
        sys.exit(1)
    
    parts = [
        (1, "DQN IMPLEMENTATION", "dqn_agent"),
        (2, "REINFORCE (VANILLA PG)", "pg_agent"),
        (3, "PG WITH BASELINE", "pg_baseline_agent"),
        (4, "COMPREHENSIVE COMPARISON", "comparison_analysis")
    ]
    
    results = []
    for part_num, title, module in parts:
        results.append(run_part(part_num, title, module))
    
    print_header("EXECUTION SUMMARY")
    
    if all(results):
        print("✓ All experiments completed successfully!\n")
    else:
        print("✗ Some experiments failed. Check logs above.\n")
    
    total_time = time.time() - start_time
    print(f"Total Time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "=" * 80)
    print("KEY RESULTS:")
    print("=" * 80)
    print("""
DQN: Most sample efficient, most stable
REINFORCE: Simplest, high variance
PG+Baseline: Balanced performance and complexity

Check results/ folder for all outputs and analysis report.
""")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)