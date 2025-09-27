#!/usr/bin/env python3
"""
Compare SAC vs CSAC training results
"""
import subprocess
import time

def run_experiment(script_name, algorithm_name):
    """Run training script and measure time"""
    print(f"\n{'='*50}")
    print(f"Running {algorithm_name} experiment...")
    print(f"{'='*50}")

    start_time = time.time()
    result = subprocess.run(['python', script_name], capture_output=True, text=True)
    end_time = time.time()

    print(f"\n{algorithm_name} Output:")
    print(result.stdout)
    if result.stderr:
        print(f"{algorithm_name} Errors:")
        print(result.stderr)

    training_time = end_time - start_time
    print(f"\n{algorithm_name} completed in {training_time:.2f} seconds")
    return result.returncode == 0, training_time

def main():
    print("Starting SAC vs CSAC comparison...")

    # Run SAC experiment
    sac_success, sac_time = run_experiment('sac_example.py', 'SAC')

    # Run CSAC experiment
    csac_success, csac_time = run_experiment('csac_example.py', 'CSAC')

    # Summary
    print(f"\n{'='*50}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*50}")
    print(f"SAC:  {'✓' if sac_success else '✗'} Success, {sac_time:.2f}s")
    print(f"CSAC: {'✓' if csac_success else '✗'} Success, {csac_time:.2f}s")

    print(f"\nLogs saved to:")
    print(f"  SAC:  ./logs/sac/")
    print(f"  CSAC: ./logs/csac/")

    print(f"\nModels saved to:")
    print(f"  SAC:  ./models/sac_pendulum.zip")
    print(f"  CSAC: ./models/csac_pendulum.zip")

    print(f"\nTo view TensorBoard logs:")
    print(f"  tensorboard --logdir=./logs")

if __name__ == "__main__":
    main()