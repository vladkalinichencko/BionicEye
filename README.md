# BionicEye Agent

This repository contains code for a Reinforcement Learning agent using a Vision Transformer to solve image classification tasks by active visual exploration.

---

## Quick Start

### 1. Local Machine

All commands are run from the project root directory.

```bash
# 1. Create virtual environment and activate it
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start training
python current_main_script.py
```

### 2. Remote Server (SC-09)

#### Step A: Copy files to server (run on your local machine)

This command copies your current project folder to the server. You will be prompted for the password.

```bash
scp -r . user9@87.242.102.117:~/BionicEye
```

#### Step B: Setup and Run (run on the remote server)

Connect to the server (`ssh user9@87.242.102.117`), then run these commands.

```bash
# --- First-time setup on server (only need to do this once) ---
cd ~/BionicEye
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# --- End of setup ---
```

#### Step C: Check Server Load (Important!)

Before starting a new training, check if the GPUs are already in use by someone else.

```bash
nvidia-smi
```
Look at the process list at the bottom. If you see other `python` processes consuming significant GPU memory, it's best to wait or coordinate with your colleagues.

#### Step D: Start Training

If the server is free, start the training in the background using `nohup`.

```bash
# Activate environment (if you've just reconnected)
source venv/bin/activate

# Start training in the background
nohup ./venv/bin/python current_main_script.py > training.log 2>&1 &

# You can now safely disconnect.
# To check progress, run:
tail -f training.log
```
The `nohup` command ensures the script keeps running even after you disconnect. All output will be written to `training.log`.

#### Step E: Downloading Results from Server

To copy the generated checkpoints or visualizations from the server back to your local machine, run these commands on your **local terminal**.

```bash
# To download the latest checkpoint:
scp user9@87.242.102.117:~/BionicEye/checkpoints/latest_checkpoint.pth ./checkpoints/

# To download all visualizations:
scp -r user9@87.242.102.117:~/BionicEye/visualizations/ ./visualizations/
```
