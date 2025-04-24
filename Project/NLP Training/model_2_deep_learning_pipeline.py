# ##################
# This script acts as a pipeline combining fine-tuning a pretrained DistilBERT model as well as custom layers.
# Codes in this script are executed on a Linux-based virtual machine with the following computational requirements:
# GPU:  RTX2080 Super
# vCPU:  8 
# CPU Memory: 48GB 
# GPU Memory: 8GB
# Author: Kelvin Mock
# ##################

import subprocess
import os
import sys

# Define the scripts to run (in sequence)
scripts = [
    "fine-tuning-distilBERT.py",
    "training-added-layers-distilBERT.py"
]

# Get the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Run each script in sequence
for script in scripts:
    script_path = os.path.join(current_directory, script)
    if os.path.exists(script_path):
        print(f"Running {script}...")
        result = subprocess.run([sys.executable, script_path], env=os.environ)
        print(result.stdout)
        if result.returncode != 0:
            print(f"Error running {script}: {result.stderr}")
            break
    else:
        print(f"Script {script} not found in the directory.")