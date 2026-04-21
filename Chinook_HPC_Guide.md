# Bubble-Mapping on Chinook HPC

A guide to deploying the UNet segmentation pipeline on the UAF Chinook cluster.

---

## 1. Overview

This guide walks through every step needed to run the bubble-mapping UNet segmentation pipeline on the UAF Research Computing Systems (RCS) Chinook HPC cluster. It covers connecting to the cluster, setting up your environment, transferring data, configuring the pipeline for HPC, writing SLURM batch scripts, and monitoring your training jobs.

The guide assumes you already have a working local copy of the pipeline (on the `windows-compat` branch) and want to move to full-scale training on Chinook using the `main` branch.

---

## 2. HPC Concepts for New Users

If you have only ever run Python scripts on your own laptop or a lab workstation, the HPC environment will feel unfamiliar at first. This section explains the key concepts before you touch a single command.

### 2.1 What an HPC Cluster Actually Is

An HPC (High-Performance Computing) cluster is a large collection of computers — called **nodes** — connected over a fast network and managed as a single shared system. Many researchers from different projects use the same cluster simultaneously. Because it is shared, you cannot simply open a terminal and start running code the way you would on your laptop. Instead, you describe the work you want done and the resources you need, and a piece of software called a **job scheduler** decides when and where to run it.

Chinook's job scheduler is called **SLURM**. Almost everything you do on Chinook beyond editing files and navigating directories goes through SLURM.

### 2.2 Nodes: Login vs. Compute vs. GPU

A "node" is just an individual computer within the cluster. Chinook has three categories relevant to your work:

**Login nodes** (`chinook.alaska.edu`, `chinookgpu.alaska.edu`) are the entry points. When you SSH into Chinook, you land on a login node. They are shared by every user who is connected at that moment and have no dedicated compute resources. Think of them as the front desk — you use them to organize your work, edit files, and submit jobs. Running a Python script directly on a login node is the equivalent of doing all your work at the front desk instead of in the lab. Other users immediately feel the impact.

**CPU compute nodes** are where non-GPU work runs. SLURM allocates these to you exclusively for the duration of your job. Your preprocessing script belongs here.

**GPU compute nodes** are compute nodes that also have GPUs attached. Your UNet training belongs here. These are accessed through a separate login node (`chinookgpu.alaska.edu`) and separate SLURM partitions.

You can always tell which type of node you are on by looking at your terminal prompt. Login node prompts show the login node's name (e.g., `chinook04`, `chinookgpu`). Compute node prompts show the node's name (e.g., `n151`, `n153`, `n155`).

### 2.3 SLURM and the Job Queue

SLURM is the software that manages who runs what and when. When you submit a job with `sbatch`, SLURM places it in a **queue** (also called the job queue or job scheduler queue). It then waits for the requested resources to become available — the right number of CPUs, GPUs, and memory — before starting your job on the appropriate node.

The queue is not strictly first-come, first-served. SLURM uses a **fair-share scheduling** system that takes into account how much of the cluster each user has used recently. If you have been running many large jobs, your priority temporarily decreases to give other users a turn. This is by design and is why being conservative with resource requests (asking for only what you need) benefits everyone including yourself — smaller, accurate requests are often scheduled faster.

To see the current state of your jobs in the queue:

```bash
module load slurm
squeue -u $USER
```

The **ST** (status) column tells you what each job is doing:

| Status | Meaning |
|---|---|
| `PD` | Pending — waiting in the queue for resources |
| `R` | Running — currently executing on a compute node |
| `CG` | Completing — job has finished, SLURM is cleaning up |
| `F` | Failed — job exited with an error |
| `TO` | Timed out — job hit the wall time limit and was killed |

### 2.4 Wall Time

**Wall time** is the maximum real-world clock time your job is allowed to run, measured from the moment SLURM starts it. "Wall" refers to the clock on the wall — elapsed time, not CPU time. You set it with the `--time` flag in your SLURM script.

When your job hits the wall time limit, SLURM kills it immediately. There is no warning, no graceful shutdown, and no automatic saving of progress. If your UNet training is on epoch 98 of 100 and your wall time expires, the job ends and you lose anything not already saved to disk.

This has two practical implications for your project. First, overestimate your wall time when you are not sure how long a job will take — it is far better to request 24 hours and finish in 12 than to request 12 hours and get killed at hour 11. Second, use **checkpointing** (see Section 9.4) so that even if your job is killed, you can resume from the last saved checkpoint rather than starting over.

The maximum wall time on both GPU partitions is 2 days (48 hours). If your training genuinely needs longer than that, you will need to design your training loop to save checkpoints and resume across multiple job submissions.

### 2.5 Partitions

A **partition** is a named group of nodes within the cluster, each with its own rules about wall time limits, resource maximums, and intended use. When you submit a job, you specify which partition to use with `--partition=`. SLURM then routes your job to nodes within that partition.

The partitions relevant to your work on Chinook are:

| Partition | Node Type | Max Wall Time | Use For |
|---|---|---|---|
| `debug` | CPU | 1 hour | Testing scripts before a real run |
| `t1small` | CPU | varies | Short CPU jobs like preprocessing |
| `l40s` | GPU (L40S, 48 GB) | 2 days | GPU training and inference |
| `h100` | GPU (H100, 80 GB) | 2 days | GPU training (preferred for your pipeline) |

Run `sinfo` to see all available partitions and their current node availability.

### 2.6 Environment Variables: $HOME, $CENTER1, $ARCHIVE

On Linux systems, a word starting with `$` is an **environment variable** — a name that expands to a value. On Chinook, RCS pre-defines several that point to your storage locations so you do not have to type full paths.

| Variable | Expands To | What It Is |
|---|---|---|
| `$HOME` | Your home directory (e.g., `/home/amwelch3`) | 50 GB, backed up. Code and conda environments. |
| `$CENTER1` | Your scratch space | Large, fast, NOT backed up. Training data and outputs. |
| `$ARCHIVE` | Your tape archive | Slow, backed up. Final results only. |
| `$USER` | Your username (`amwelch3`) | Useful in scripts and commands like `squeue -u $USER` |

You can always check what a variable expands to with `echo`:

```bash
echo $HOME
echo $CENTER1
```

### 2.7 What "Backed Up" and "Not Backed Up" Mean

**$HOME is backed up** — if a disk fails or you accidentally delete a file, RCS can restore it from a backup. This does not mean you should store large data there (the 50 GB quota will stop you anyway), but it does mean your code and conda environment are relatively safe.

**$CENTER1 is NOT backed up** — if files are deleted or a disk fails, they are gone permanently. $CENTER1 is intended as working scratch space: you put data there to process it, then move your final results somewhere permanent. Treat everything in $CENTER1 as temporary. Always copy trained model checkpoints to $ARCHIVE or your own computer when a training run completes.

**$ARCHIVE is tape-backed** — data here is stored on magnetic tape for long-term retention. It is very slow to access and is not mounted on compute nodes, so you cannot read from it during a job. It is purely for archiving final outputs.

### 2.8 Modules

Chinook uses a system called **LMod** to manage software. Rather than having every piece of software installed and active at all times, software is organized into **modules** that you load on demand. When you load a module, it makes that software available in your current session.

For your work, the only module you will regularly need to load is `slurm`, which makes SLURM commands like `sbatch`, `squeue`, and `scancel` available:

```bash
module load slurm
```

You need to run this once per login session before using any SLURM commands. SLURM batch scripts should also include `module load slurm` (or `module purge` followed by `module load slurm`) near the top to ensure the correct environment inside the job.

### 2.9 Log Files and Reading Job Output

When you run a Python script on your laptop, output prints to your terminal. On Chinook, your job runs on a compute node while your terminal sits idle on the login node — the output has nowhere to go except a file. This is what the `--output=logs/train_%j.out` line in your SLURM script does: it redirects all output (both standard output and errors) to a file. The `%j` is automatically replaced by the job ID number.

To watch the output of a running job in real time:

```bash
tail -f logs/train_<JOBID>.out
```

`tail -f` prints the last few lines of a file and then keeps watching it, printing new lines as they are written. Press `Ctrl+C` to stop watching (this does not affect the job itself).

If your job fails, the log file is the first place to look. The error message is usually near the bottom. Common things to look for: `ModuleNotFoundError` (package not installed in conda env), `CUDA out of memory` (reduce batch size), or `Killed` at the very end (usually out-of-memory at the OS level — request more `--mem-per-gpu`).

### 2.10 GPU Memory vs. System RAM

There are two completely separate pools of memory on a GPU node, and the distinction matters when writing your SLURM script:

**GPU memory (VRAM)** lives on the GPU card itself and is where your model weights, activations, and data tensors live during training. The L40S has 48 GB of GPU memory; the H100 has 80 GB. You do not request this in your SLURM script — it is allocated automatically when you request a GPU. If your model or batch is too large to fit, you get a `CUDA out of memory` error.

**System RAM** is the regular computer memory (the node has 1 TB total). This is what `--mem-per-gpu=185G` controls. Your CPU processes — Python itself, the DataLoader workers reading files from disk, GDAL raster operations — use system RAM. If you exceed your system RAM allocation, your job is killed by the OS with little warning, and the log file will show `Killed` as the last line.

For your pipeline, the 185 GB per GPU recommendation from RCS is appropriate. The large satellite imagery files are the main driver of system RAM usage.

---

## 3. Chinook Quick Reference

Chinook is a condo-style HPC cluster at the UAF Geophysical Institute, running Rocky Linux 8 with SLURM as the job scheduler.

| Resource | Details |
|---|---|
| **GPU Nodes (L40S)** | 5 nodes, each: dual AMD EPYC 24-core (48 cores), 1 TB RAM, 8x NVIDIA L40S |
| **GPU Node (H100)** | 1 node: dual AMD EPYC 24-core (48 cores), 1 TB RAM, 8x NVIDIA H100 |
| **$HOME** | 50 GB quota, backed up. Store code and conda environments here. |
| **$CENTER1** | Lustre scratch (307 TB total), **NOT backed up**, 1 TB per project default. Store training data and outputs here. |
| **$ARCHIVE** | Tape-backed long-term storage. Not available on compute nodes. Use for final results archival. |

> **Important:** $CENTER1 is not backed up. Always copy final models and results to $ARCHIVE or off-cluster when training completes. Large numbers of small files slow down the Lustre parallel filesystem, so avoid extracting thousands of tiny chip files directly on $CENTER1 if possible.

---

## 4. Connecting to Chinook

### 4.1 SSH Access

You connect to Chinook via SSH at `chinook.alaska.edu` using your UA credentials.

**From Windows (PuTTY):**

1. Download and install PuTTY from the official site (putty.org).
2. Set the Host Name to `chinook.alaska.edu` and click Open.
3. Enter your UA username and password when prompted.
4. To enable graphical forwarding (for TensorBoard), go to Connection → SSH → X11 and check "Enable X11 forwarding". You also need an X server like XMing installed locally.

**From Mac/Linux:**

```bash
ssh amwelch3@chinook.alaska.edu
```

For graphical forwarding:

```bash
ssh -Y amwelch3@chinook.alaska.edu
```

Mac users need XQuartz installed for X11 forwarding.

### 4.2 Getting Your Code onto Chinook (Git — Recommended)

For your code repository, **git is the best way to get it onto Chinook** — and the best way to keep it updated as you make changes. Since your project is already on GitHub (or a similar remote), you simply clone it directly from Chinook:

```bash
# On Chinook, after logging in:
cd ~
git clone https://github.com/your-username/bubble-mapping.git
```

This puts a full copy of the repository in `~/bubble-mapping`. When you make changes to your code locally and push them to GitHub, you can pull the updates onto Chinook with:

```bash
cd ~/bubble-mapping
git pull
```

This is far better than SCP for code because it only transfers what changed, keeps a full history, and makes it impossible to accidentally overwrite the wrong file. It also means your Chinook copy and local copy stay in sync through git rather than through manual file management.

> **Note:** Chinook has outbound internet access, so `git clone` and `git pull` from public repositories like GitHub work directly. If your repository is private, you will need to either use a personal access token in the URL or set up an SSH key on Chinook and add it to GitHub.

**After cloning, switch to the correct branch:**

```bash
cd ~/bubble-mapping
git checkout main
```

### 4.3 Getting Data onto Chinook (SCP and Rsync)

Large data files — your satellite imagery, label geopackages, and preprocessed chips — cannot go through git (they are too large and binary files do not belong in git repositories). For these, use SCP or rsync.

**SCP** (Secure Copy) is the simplest option for one-time transfers. Run this from your local Windows terminal (PowerShell or Command Prompt):

```bash
# Transfer a folder of imagery to $CENTER1
scp -r "C:\Users\amwelch3\data\training_images" amwelch3@chinook.alaska.edu:/center1/your-project/data/training_images/
```

**Rsync** is better for large or repeated transfers. It checks what already exists at the destination and only transfers the differences, so if a transfer is interrupted you can resume it without re-sending everything:

```bash
# From your local machine (Mac/Linux terminal):
rsync -avz --progress /path/to/local/data/ amwelch3@chinook.alaska.edu:/center1/your-project/data/

# The trailing slash on the source matters:
# /data/   → copies the CONTENTS of data/ into the destination
# /data    → copies the data/ folder itself into the destination
```

**Globus** is recommended for very large datasets (hundreds of GB or more). It is a managed transfer service that handles interruptions automatically and can run in the background. RCS has a Globus endpoint for Chinook — contact uaf-rcs@alaska.edu for setup instructions if you need it.

> **Important:** Transfer large data directly to `$CENTER1`, not to `$HOME`. Your $HOME quota is only 50 GB. Run `echo $CENTER1` on Chinook to see your exact scratch path before constructing transfer commands.

### 4.4 Moving Files Within Chinook ($HOME, $CENTER1, $ARCHIVE)

Moving files between the three storage systems on Chinook is done manually with regular Linux copy commands — **nothing moves automatically**. In particular, files do not go to $ARCHIVE on their own. If you do not copy your results there yourself, and $CENTER1 is purged or a disk fails, they are gone.

**Copying files between storage systems:**

```bash
# Copy a trained model from scratch to archive (run from the login node after your job finishes)
cp -r $CENTER1/data/models/UNET/AE/ $ARCHIVE/bubble-mapping/models/

# Copy results to archive
cp -r $CENTER1/data/results/ $ARCHIVE/bubble-mapping/results/
```

Use `cp -r` (recursive) for directories. For large copies, `rsync` is again preferable because it can resume if interrupted and shows progress:

```bash
rsync -avz --progress $CENTER1/data/models/ $ARCHIVE/bubble-mapping/models/
```

**Important limitations of $ARCHIVE:**

- $ARCHIVE is backed by magnetic tape and is **very slow to read from**. It is write-once archival storage, not a place to actively work from.
- $ARCHIVE is **not mounted on compute nodes**. You cannot read from or write to $ARCHIVE inside a SLURM job. All archiving must be done from the login node, after your job completes.
- Reading data back from $ARCHIVE (e.g., if you need to retrieve an old model) can take minutes to hours as the tape has to be located and loaded.

**Recommended archiving workflow after a training run completes:**

```bash
# 1. Verify the job finished successfully and the model file exists
ls -lh $CENTER1/data/models/UNET/AE/

# 2. Copy to archive from the login node
cp -r $CENTER1/data/models/UNET/AE/ $ARCHIVE/bubble-mapping/models/AE_$(date +%Y%m%d)/

# 3. Optionally copy results too
cp -r $CENTER1/data/results/ $ARCHIVE/bubble-mapping/results/

# 4. Verify the archive copy before deleting anything from $CENTER1
ls -lh $ARCHIVE/bubble-mapping/models/
```

> **Tip:** The `$(date +%Y%m%d)` in the copy command appends today's date (e.g., `AE_20260410`) to the folder name, which makes it easy to tell different training runs apart in $ARCHIVE.

---

## 5. Environment Setup

Chinook uses the LMod module system for software. You need to load the `slurm` module to interact with the job scheduler. For Python environments, the recommended approach is conda (Miniconda), installed in your $HOME directory.

### 5.1 Install Miniconda

If Miniconda is not already installed in your $HOME:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init bash
```

### 5.2 Create the Conda Environment

Your project includes `pixi.toml` which defines the HPC environment (Python 3.11, PyTorch 2.5, CUDA 12.1). You can use this as a reference, but on Chinook the straightforward approach is conda:

```bash
conda create -n bubble-mapping python=3.11 -y
conda activate bubble-mapping

# Install PyTorch with CUDA (match the GPU driver on Chinook)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 \
    -c pytorch -c nvidia -y

# Install geospatial and scientific stack
conda install -c conda-forge gdal geopandas rasterio \
    scikit-learn matplotlib tensorboard tqdm \
    albumentations optuna h5py -y

# Install imgaug (pip only)
pip install imgaug
```

> **File Quota Warning:** Conda environments generate many files. Be cautious about how many environments you create. Never install conda environments in $ARCHIVE. Monitor your file count with `show_storage`.

### 5.3 Alternative: Pixi

If you prefer pixi (the tool defined in pixi.toml), you can install it and use it directly:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
cd ~/bubble-mapping
pixi install
eval "$(pixi shell-hook)"
```

Pixi reads `pixi.toml` and creates a fully reproducible environment. Either approach (conda or pixi) works. The key requirement is Python 3.11 with PyTorch and CUDA support.

---

## 6. Project Layout on Chinook

Here is the recommended file organization, using the appropriate filesystem for each type of data:

| Filesystem | Path | Contents |
|---|---|---|
| $HOME | `~/bubble-mapping/` | Code repository, config files, conda env |
| $CENTER1 | `$CENTER1/data/training_images/AE/` | Raw .tif satellite imagery |
| $CENTER1 | `$CENTER1/data/training/AE/` | Label geopackages (.gpkg) |
| $CENTER1 | `$CENTER1/data/preprocessed/` | Preprocessed patches (output of preprocessing.py) |
| $CENTER1 | `$CENTER1/data/models/` | Saved model checkpoints |
| $CENTER1 | `$CENTER1/data/logs/` | TensorBoard logs |
| $CENTER1 | `$CENTER1/data/results/` | Prediction outputs (GeoTIFF masks) |
| $ARCHIVE | `$ARCHIVE/bubble-mapping/` | Final trained models and results (backed up) |

Use symlinks to connect the code to the data without duplicating files:

```bash
cd ~/bubble-mapping
ln -s $CENTER1/data data
```

> **Q: So I just put that command into the command line and it will always read "data/" as the data folder in $CENTER1? Do I need to run that command every time I ssh onto the HPC?**
>
> You only run it **once, ever**. A symlink is a permanent filesystem entry stored in `$HOME` (which is backed up), not a session setting. After you run `ln -s $CENTER1/data data`, a file called `data` appears in `~/bubble-mapping/` and permanently points to your `$CENTER1/data` directory. Every time you log in — now or a year from now — `~/bubble-mapping/data/` automatically redirects to `$CENTER1/data/` without any extra steps. Linux handles the redirection transparently at the filesystem level.

This way, when the pipeline references `data/training_images/AE/`, it transparently reads from $CENTER1.

---

## 7. Configuration Changes for HPC

The pipeline is controlled by `config/configUnetxAE.py`. Several settings must change when moving from your local Windows machine to Chinook. Make sure you are on the **main** branch (not windows-compat), which already has most HPC-ready defaults.

### 7.1 Switch to the Main Branch

```bash
cd ~/bubble-mapping
git checkout main
```

The `main` branch has `fit_workers=8` and full training parameters. The `windows-compat` branch has `fit_workers=0` and smoke-test parameters, which are not suitable for HPC.

### 7.2 Settings to Update

Open `config/configUnetxAE.py` and make the following changes:

| Setting | Windows Value | HPC Value | Why |
|---|---|---|---|
| `REPO_PATH` | `r"C:\Users\..."` | `os.path.expanduser("~/bubble-mapping")` | Linux paths |
| `selected_GPU` | `1` | `0` | SLURM sets CUDA_VISIBLE_DEVICES; GPU 0 is always the first visible |
| `fit_workers` | `0` | `8` | Linux fork() works; 8 workers for 48-core nodes |
| `train_batch_size` | `8` | `32` | H100/L40S have enough VRAM |
| `num_epochs` | `10` | `100` | Full training run |
| `num_training_steps` | `50` | `500` | Full training run |
| `heavy_eval_steps` | `10` | `50` | Full evaluation cadence |
| `eval_mc_dropout` | `False` | `True` | Enable uncertainty estimation |
| `CPL_LOG` path | `"NUL"` | `"/dev/null"` | Linux null device |
| `postproc_workers` | `12` | `12` (or higher) | Plenty of CPU cores available |

> **Tip: GPU Selection.** On SLURM, when you request `--gres=gpu:1`, the scheduler sets `CUDA_VISIBLE_DEVICES` so that only your allocated GPU is visible. Your code should always use `selected_GPU = 0`, because GPU 0 is whatever SLURM assigned to you. Do not hardcode a specific GPU index like 7.

### 7.3 Example Config Diff

Here are the key lines to change at the top of `configUnetxAE.py`:

```python
# OLD (Windows):
REPO_PATH = r"C:\Users\amwelch3\git_repos\bubble-mapping"

# NEW (Chinook):
import os
REPO_PATH = os.path.expanduser("~/bubble-mapping")
```

And the GDAL null log path:

```python
# OLD (Windows):
gdal.SetConfigOption("CPL_LOG", "NUL")

# NEW (Linux):
gdal.SetConfigOption("CPL_LOG", "/dev/null")
```

### 7.4 Understanding and Updating Paths Throughout Your Scripts

This is one of the most common sources of confusion when moving from Windows to a Linux HPC. Paths work differently between operating systems, and any path you wrote on Windows will break on Chinook.

**The key differences:**

| Concept | Windows | Linux (Chinook) |
|---|---|---|
| Folder separator | Backslash `\` | Forward slash `/` |
| Home directory | `C:\Users\amwelch3` | `/home/amwelch3` or `~` |
| Raw string paths in Python | `r"C:\Users\..."` | `"/home/amwelch3/..."` |
| Null device | `"NUL"` | `"/dev/null"` |

**Never hardcode a full path in your scripts.** Instead, build paths dynamically so they work regardless of where the code runs:

```python
import os
from pathlib import Path

# Bad — hardcoded, breaks on any other machine:
data_dir = r"C:\Users\amwelch3\data\training_images"

# Good — works on both Windows and Linux:
data_dir = Path.home() / "data" / "training_images"

# Also good — using environment variables for Chinook-specific paths:
data_dir = Path(os.environ.get("CENTER1", Path.home())) / "data" / "training_images"
```

**Finding hardcoded paths in your scripts:**

To locate every place in your code where a Windows-style path might be hiding, run this from your repository root on Chinook:

```bash
# Find any file containing a Windows-style backslash path
grep -r "C:\\\\" . --include="*.py"

# Find any file containing your username in a path (catches most hardcoded paths)
grep -r "amwelch3" . --include="*.py"

# Find any file containing "NUL" (the Windows null device)
grep -r '"NUL"' . --include="*.py"
```

**For your pipeline specifically**, the primary path to update is `REPO_PATH` in `config/configUnetxAE.py`, which is already covered in Section 7.2. The data directories are handled through the symlink approach in Section 6 — because `~/bubble-mapping/data` is a symlink pointing to `$CENTER1/data`, your pipeline code can refer to relative paths like `data/training_images/` without needing to know the full $CENTER1 path. If you find any other scripts that reference absolute paths, apply the same `os.path.expanduser` or `pathlib.Path.home()` pattern.

---

## 8. How Your Pipeline Uses Parallel Computing

Chinook asks you to verify that your code can use parallel processing. Your bubble-mapping pipeline already does this in several ways.

### 8.1 GPU Parallelism (CUDA)

This is the primary form of parallelism in your pipeline. PyTorch automatically distributes tensor operations (convolutions, matrix multiplies) across the thousands of CUDA cores on the GPU. The UNet model in `core/UNet.py` runs entirely on GPU during both training and inference. No code changes are needed for this; PyTorch handles it when you call `model.to("cuda")`.

Your pipeline also uses **mixed-precision training** (BF16/FP16) via `torch.cuda.amp`, which doubles throughput on modern GPUs like the H100 and L40S by using lower-precision arithmetic for operations that don't need full 32-bit precision.

### 8.2 Multi-Core CPU Parallelism (DataLoader Workers)

The `fit_workers` setting controls how many CPU processes load and augment training data in parallel while the GPU is busy computing. On Chinook (48 cores per GPU node), setting `fit_workers=8` means 8 processes read patches from disk, apply augmentations (rotations, flips, brightness changes via albumentations), and feed them to the GPU. This prevents the GPU from sitting idle waiting for data.

The DataLoader also uses `pin_memory=True` and `prefetch_factor=2`, which keeps the next batches ready in GPU-accessible memory before the current batch finishes processing.

### 8.3 What This Pipeline Does NOT Need

Your pipeline does **not** use MPI (Message Passing Interface) or multi-node parallelism. It trains on a single GPU, which is the standard approach for UNet segmentation at this scale. Multi-GPU training (using `torch.nn.DataParallel` or `DistributedDataParallel`) could be added later if training is too slow on one GPU, but is not expected to be necessary. Each training run should complete in hours, not days.

> **For the SLURM partition question:** Since your pipeline uses one GPU and does not require MPI, you do not need multi-node partitions like `t1standard`. You should use the GPU partition (see Section 8). The Chinook documentation mentions that serial or single-node jobs should avoid the standard partition, which reserves 3+ nodes by default.

---

## 9. Writing SLURM Batch Scripts

Batch scripts are plain-text files that tell SLURM what resources you need and what commands to run. They use **bash** (yes, bash is the right shell). The script has two parts: `#SBATCH` directives at the top that request resources, and then regular shell commands.

> **Q: It seems like I'm just running the python scripts individually. How do I load the settings in the config file? Because currently, I run from mainUnet which imports the configuration file. Do I need to import that on each of the python scripts?**
>
> On HPC, each SLURM job runs one script directly — there is no `mainUnet.py` hub calling the others. So yes, each script that needs config values (`preprocessing.py`, `training.py`, `evaluation.py`) must import the config file itself at the top. For example:
>
> ```python
> from config.configUnetxAE import *
> ```
>
> Check each script now: if they already have this import, you are set. If they currently receive config values as arguments passed in by `mainUnet.py`, you will need to add the import directly. This is a one-time change per script. Once each script imports the config on its own, running `python training.py` from a SLURM job will pick up all the same settings you configured in `configUnetxAE.py`.
### 9.1 Preprocessing Job

Preprocessing is CPU-bound (reading rasters, creating chips) and does not need a GPU. Run it as a separate job first.

Save this as `preprocess.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=bubble-preprocess
#SBATCH --partition=t1small
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --mail-user=amwelch3@alaska.edu
#SBATCH --mail-type=END,FAIL

module purge
module load slurm

eval "$(conda shell.bash hook)"
conda activate bubble-mapping

cd ~/bubble-mapping
python preprocessing.py
```

Submit with:

```bash
module load slurm
sbatch preprocess.slurm
```

### 9.2 Training Job (GPU)

This is the main training script that needs a GPU. The key addition is `--gres=gpu:1` to request one GPU.

Save as `train.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=bubble-train
#SBATCH --partition=h100
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=185G
#SBATCH --cpus-per-gpu=6
#SBATCH --output=logs/train_%j.out
#SBATCH --mail-user=amwelch3@alaska.edu
#SBATCH --mail-type=BEGIN,END,FAIL

module purge
module load slurm

eval "$(conda shell.bash hook)"
conda activate bubble-mapping

cd ~/bubble-mapping

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"

python training.py

echo "Job finished: $(date)"
```

Submit from the GPU login node (`chinookgpu.alaska.edu`):

```bash
sbatch train.slurm
```

> **GPU Partitions:** Chinook has two GPU partitions: `h100` (NVIDIA H100, 80 GB, preferred for training) and `l40s` (NVIDIA L40S, 48 GB). See Section 14.2 for details on which to choose.

### 9.3 Evaluation Job

Evaluation also benefits from a GPU for inference speed.

Save as `eval.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=bubble-eval
#SBATCH --partition=l40s
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=185G
#SBATCH --cpus-per-gpu=6
#SBATCH --output=logs/eval_%j.out
#SBATCH --mail-user=amwelch3@alaska.edu
#SBATCH --mail-type=END,FAIL

module purge
module load slurm

eval "$(conda shell.bash hook)"
conda activate bubble-mapping

cd ~/bubble-mapping
python evaluation.py
```

> **Partition choice for evaluation:** The `l40s` partition is appropriate for inference/evaluation — it has 48 GB GPU memory which is plenty for running a trained model. Reserve the `h100` partition for full training runs.

### 9.4 Understanding the SBATCH Directives

| Directive | What It Does |
|---|---|
| `--partition=h100` or `--partition=l40s` | Routes the job to the appropriate GPU-equipped nodes (H100 for training, L40S for eval) |
| `--gpus=1` | Request 1 GPU. SLURM sets CUDA_VISIBLE_DEVICES automatically. Always specify this — if omitted, SLURM allocates ALL GPUs on the node. |
| `--mem-per-gpu=185G` | System RAM per GPU (not GPU VRAM). Covers DataLoader workers and GDAL raster operations. |
| `--cpus-per-gpu=6` | CPU cores per GPU. Feeds DataLoader workers + overhead. |
| `--time=24:00:00` | Wall time limit. Job is killed after this. Overestimate at first. |
| `--output=logs/train_%j.out` | Stdout/stderr go here. %j is replaced by the job ID. |
| `--mail-type=END,FAIL` | Email you when the job finishes or fails. |

---

## 10. Running and Monitoring Jobs

### 10.1 Essential SLURM Commands

You must load the slurm module before using any of these commands:

```bash
module load slurm
```

| Command | Purpose |
|---|---|
| `sbatch train.slurm` | Submit a batch job |
| `squeue -u $USER` | Check status of your jobs |
| `scancel <jobid>` | Cancel a running or queued job |
| `sinfo` | See available partitions and node status |
| `sacct -j <jobid> --format=Elapsed,MaxRSS` | Check resource usage after completion |
| `tail -f logs/train_<jobid>.out` | Watch live output from a running job |

### 10.2 Debug Partition First

Chinook recommends testing in the debug partition before submitting to production queues. Change your training script to use debug with a short time limit and reduced epochs to verify everything works:

```bash
#SBATCH --partition=debug
#SBATCH --time=01:00:00
```

Also temporarily set `num_epochs=2` and `num_training_steps=10` in the config. Once the debug job completes successfully, restore full parameters and submit to the GPU partition.

### 10.3 TensorBoard Monitoring

To monitor training curves (loss, Dice score) in real time, use SSH port forwarding. From your local machine:

```bash
ssh -L 6006:localhost:6006 amwelch3@chinook.alaska.edu
```

Then on Chinook (from the login node):

```bash
eval "$(conda shell.bash hook)"
conda activate bubble-mapping
tensorboard --logdir $CENTER1/data/logs/UNET/AE --port 6006
```

Open http://localhost:6006 in your local browser to see the dashboard.

> **What to watch in TensorBoard:** Both `train_loss` and `val_loss` should decrease over epochs. Dice/F1 should increase toward 0.5-0.7+. If `val_loss` rises while `train_loss` falls, the model is overfitting. If loss goes to NaN, there is a gradient explosion problem. See the TensorBoard Guide section in CLAUDE.md for detailed interpretation.

### 10.4 Checkpointing: Protecting Your Training Progress

Because SLURM will kill your job when it hits the wall time limit — with no warning and no grace period — you need a strategy for saving your progress periodically during training. This is called **checkpointing**.

A checkpoint is a snapshot of your model's weights (and optionally the optimizer state) saved to disk at a regular interval. If your job is killed after 24 hours of training, you can restart it from the last checkpoint rather than from scratch.

PyTorch saves checkpoints with `torch.save`. At minimum, save the model weights:

```python
import torch

# Save a checkpoint at the end of each epoch
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, f"$CENTER1/data/models/UNET/AE/checkpoint_epoch_{epoch}.pt")
```

To resume from a checkpoint at the start of a new job:

```python
checkpoint = torch.load('$CENTER1/data/models/UNET/AE/checkpoint_epoch_50.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

**Practical recommendations for your pipeline:**

Save a checkpoint at the end of every epoch (since your full run is 100 epochs, this means 100 checkpoint files). To avoid filling your storage, keep only the last 3 and the best-performing checkpoint:

```python
import os

# After saving checkpoint_epoch_N.pt, delete checkpoint_epoch_(N-3).pt
old = f"checkpoint_epoch_{epoch - 3}.pt"
if os.path.exists(old):
    os.remove(old)
```

Check whether your pipeline's training loop already handles checkpointing — if `configUnetxAE.py` has settings related to model saving frequency, use those rather than adding duplicate logic.

> **Q: Where do I write in this checkpoint code?**
>
> The checkpoint save logic goes inside the training loop in `training.py`, at the end of each epoch — after you compute validation metrics but before the loop moves to the next epoch. Look for the `for epoch in range(num_epochs):` loop. Near the bottom of that loop body, add the `torch.save(...)` block shown above.
>
> Before adding anything, check whether the pipeline already handles this. Search for `torch.save` or `model_state_dict` in `training.py` — if those already exist, the pipeline is already checkpointing and you should use the existing mechanism rather than duplicating it. Also check `configUnetxAE.py` for settings like `save_freq` or `checkpoint_interval`, which would control how often saves happen.
>
> If there is no existing checkpointing, the minimal addition looks like this inside the epoch loop:
>
> ```python
> # At the end of each epoch in training.py:
> import os
> checkpoint_dir = os.path.join(os.environ.get("CENTER1", "."), "data/models/UNET/AE")
> os.makedirs(checkpoint_dir, exist_ok=True)
> torch.save({
>     'epoch': epoch,
>     'model_state_dict': model.state_dict(),
>     'optimizer_state_dict': optimizer.state_dict(),
>     'loss': loss,
> }, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"))
> # Delete checkpoints older than 3 epochs to save space
> old_ckpt = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch - 3}.pt")
> if os.path.exists(old_ckpt):
>     os.remove(old_ckpt)
> ```

---

## 11. Complete Workflow Summary

Here is the end-to-end process, from first login to retrieving results:

1. Connect to Chinook via SSH and set up your conda environment (one-time setup, Sections 4-5).
2. Clone your code repository to $HOME with git, and transfer data to $CENTER1 with SCP or rsync (Sections 4.2-4.3, 6).
3. Switch to the `main` branch and update `configUnetxAE.py` for HPC (Section 7).
4. Create the logs directory: `mkdir -p ~/bubble-mapping/logs`
5. Submit the preprocessing job and wait for it to complete (Section 9.1).
6. Test training in the debug partition with reduced epochs (Section 10.2).
7. Submit the full training job to the GPU partition (Section 9.2).
8. Monitor training via TensorBoard and tail the log output (Section 10.3).
9. When training completes, submit the evaluation job (Section 9.3).
10. Copy final models and results to $ARCHIVE for safekeeping (Section 4.4).

---
> **Q: CENTER1 is not backed up, so does that mean I need to clone my repository every time I SSH onto the HPC?**
>
> No. Your code repository lives in `$HOME` (i.e., `~/bubble-mapping`), which **is** backed up and persists indefinitely between sessions. You clone once and it stays there. `$CENTER1` is where your **data** lives (satellite imagery, preprocessed chips, model checkpoints) — and yes, that is at risk if a disk fails. But that has no bearing on your code. When you SSH in on any given day, your repository is exactly where you left it.
>
> The only thing that would require re-cloning is if you deliberately deleted the folder or if there was an extraordinary storage failure in `$HOME`. Under normal circumstances, `cd ~/bubble-mapping && git pull` is all you ever need to do to update your code.

> **Q: Do I need to transfer large data files every time I ssh onto the HPC, since they're not backed up in $CENTER1? Or are you saying they should still be there barring some disk failure?**
>
> They will be there every time you log in — you transfer once and the files stay on the cluster. "Not backed up" does **not** mean the files disappear between sessions; it means there is no disaster-recovery copy. Under normal circumstances, files in `$CENTER1` persist indefinitely until you delete them or RCS purges scratch space (which they do occasionally for very old, large files — they announce this in advance).
>
> The practical implication: your satellite imagery and preprocessed chips are safe to leave in `$CENTER1` throughout a project. The only real risks are hardware failure or a purge event — both rare. The reason to copy final **model checkpoints and results** to `$ARCHIVE` (or your local machine) is to protect those specific outputs from those rare events, not because you need to re-upload your data constantly.

> **Q: Should I be running every script in debug prior to running it on the real nodes?**
>
> For any **new script** or after **significant changes**, yes — always run a quick debug job first. The debug partition allocates resources much faster (often within seconds), so you find out about crashes, missing imports, or config errors immediately rather than waiting hours only to have a GPU job fail in the first 30 seconds.
>
> The practical pattern: before submitting a real training job, temporarily set `--partition=debug`, `--time=00:30:00`, and reduce `num_epochs` to 2 and `num_training_steps` to 10 in the config. Once the debug job completes successfully, restore the full parameters and submit to the `h100` or `l40s` partition.
>
> For jobs you have already run successfully and are re-submitting with no code changes (e.g., a second full training run with different hyperparameters), you can skip the debug run — but it never hurts.

## 12. Troubleshooting

### 12.1 Common Issues

**"ModuleNotFoundError: No module named 'osgeo'"** -- GDAL is not installed in your conda environment. Run: `conda install -c conda-forge gdal`

**"CUDA out of memory"** -- Reduce `train_batch_size` (try 16 instead of 32), or request a node with more GPU memory (H100 has 80 GB vs. L40S with 48 GB).

**Job stuck in queue (PD state)** -- GPU nodes are in high demand. Check priority with `sprio -j <jobid>`. Consider reducing `--time` to improve scheduling priority. The debug partition has a max 1-hour walltime but faster turnaround.

**"import resource" error** -- You are on the `windows-compat` branch. Switch to main: `git checkout main`. The main branch has this import guarded for Linux.

**Slow data loading** -- Verify `fit_workers` is 8 (not 0). If data is on $ARCHIVE, stage it first with `batch_stage` or copy it to $CENTER1. Lustre performance degrades with many small files; consider storing preprocessed data as larger chunk files if this becomes an issue.

### 12.2 Getting Help

Contact RCS User Support at uaf-rcs@alaska.edu for Chinook-specific issues (account access, partitions, GPU availability, storage quotas). For pipeline-specific questions, refer to `pipeline_guide.md` in the repository.

---

## 13. Responsible Use for New HPC Users

This section is aimed at users who are new to HPC environments. Violating RCS policies can result in warnings, account suspension, or in serious cases, legal action. Most violations are unintentional — this section explains the most common pitfalls and how to avoid them.

### 13.1 The Most Important Concept: Login Nodes vs. Compute Nodes

This is the single most important thing to understand as a new HPC user.

When you SSH into Chinook, you land on a **login node** (e.g., `chinook.alaska.edu` or `chinookgpu.alaska.edu`). Think of the login node as the front desk of a shared office building — it is shared by every single user on the cluster at the same time. It exists only for lightweight tasks: navigating directories, editing files, writing scripts, and submitting jobs. It has no dedicated resources for your work.

**Compute nodes** are the actual workhorses — the GPU and CPU machines that SLURM allocates to you exclusively when you submit a job. Your Python training scripts, preprocessing pipelines, and anything computationally intensive must run on compute nodes, not the login node.

**What "running a job on the login node" looks like (what NOT to do):**

```bash
# You SSH in and your terminal prompt looks like this:
[amwelch3@chinook04 ~]$

# BAD — running training directly in this terminal:
python training.py       # This runs on the login node. Do NOT do this.
python preprocessing.py  # Same problem.
```

**What running a job on a compute node looks like (correct):**

```bash
# Good — submit your script to SLURM, which sends it to a compute node:
module load slurm
sbatch train.slurm

# Or for an interactive session on a compute node:
srun -p l40s --gpus=1 --mem-per-gpu=185G --cpus-per-gpu=6 --pty /bin/bash -l
# Your prompt will change to show you're on a compute node:
[amwelch3@n153 ~]$   # "n153" is a compute node — you're safe to run code here.
```

The key tell is your **terminal prompt**: if it shows `chinook04`, `chinookgpu`, or similar login node names, you are on the login node. If it shows a node name like `n151`, `n153`, or `n155`, you are on a compute node.

### 13.2 Avoiding Common Policy Violations

**Running intensive work on the login node** is the most frequent violation for new users. Any time you want to run a Python script, process data, or test your model, use `srun` for an interactive session or `sbatch` for a batch job. The only exception is very quick commands: checking a file size, editing a config, or running `squeue`.

**Leaving GPUs idle** is actively monitored on Chinook. If you request `--gpus=2` but your code only uses one GPU, RCS may terminate your job. Always verify your code is actually using the GPUs you requested (see Section 13.6), and only request the number of GPUs your code actually needs. For your UNet pipeline, `--gpus=1` is appropriate.

**Sharing your account credentials** — never give anyone your UA username or password, even a labmate. If a collaborator needs access to Chinook, they must request their own account through RCS.

**Setting insecure permissions on $HOME** — as described earlier in this guide, never run `chmod g+w ~` or `chmod o+w ~`. Only you should have write access to your home directory. If you receive a security warning from RCS about permissions, run:

```bash
chmod 700 ~         # Restrict home directory to owner only
chmod 600 ~/.ssh/*  # Restrict SSH keys
chmod 700 ~/.ssh
```

**Requesting far more resources than you need** — inflated resource requests (e.g., `--mem=1TB` for a job that uses 10 GB) prevent other users from running their jobs. Start conservatively, check your actual usage with `sacct`, and scale up only as needed.

**Storing large data in HOME** — your home directory has a 50 GB quota. Large satellite imagery and preprocessed chips must go in `CENTER1`. Running out of quota can cause your jobs to crash mid-run.

**Using $CENTER1 as a backup** — CENTER1 is not backed up. Treating it as permanent storage and losing data when files are purged is not a policy violation, but it is a very painful mistake. Always copy final models and results to `ARCHIVE`.

> **Q: My results aren't anything crazy in terms of file size — would it be better to just copy results to my local machine instead of (or maybe in addition to) $ARCHIVE?**
>
> Yes, copying to your local machine is often the better primary backup for small-to-medium results, and you should absolutely do it. Model checkpoints (a few hundred MB to a couple GB) and result GeoTIFFs are very manageable to transfer and store locally.
>
> A practical strategy: **copy to both**. Use `$ARCHIVE` as a belt-and-suspenders backup that stays on the cluster in case you need to re-run evaluation or share files with your advisor. Copy to your local machine for easy day-to-day access. To transfer from Chinook to your Windows machine, run this in PowerShell:
>
> ```bash
> scp -r amwelch3@chinook.alaska.edu:/center1/your-project/data/models/UNET/AE/ "C:\Users\amwelch3\results\"
> ```
>
> Or use rsync if you have it available (e.g., via Git Bash or WSL):
>
> ```bash
> rsync -avz --progress amwelch3@chinook.alaska.edu:/center1/your-project/data/results/ ~/results/
> ```
>
> The one reason to still archive on `$ARCHIVE` even for small files: if you step away from the project for months and your local machine changes, having a copy on the cluster means you never lose the lineage of a trained model.
### 13.3 Quick Pre-Submission Checklist

Before submitting any job, run through these:

- [ ] Am I submitting via `sbatch` or `srun` (not running Python directly on the login node)?
- [ ] Is `--gpus` set to only the number of GPUs my code actually uses (1 for this pipeline)?
- [ ] Is `--time` reasonable — not set to the maximum just in case?
- [ ] Is my training data in `$CENTER1`, not `$HOME`?
- [ ] Have I tested with the `debug` partition and reduced epochs before submitting a long job?
- [ ] Am I logged into `chinookgpu.alaska.edu` (not `chinook.alaska.edu`) for GPU jobs?

---

## 14. Accessing the GPU Nodes

GPU jobs on Chinook use a separate login node and a different set of partitions from the standard CPU cluster. This section is based on RCS's official GPU documentation.

### 14.1 Connecting to the GPU Login Node

GPU nodes use a **different login address** than the standard Chinook cluster:

```bash
ssh amwelch3@chinookgpu.alaska.edu
```

> **Important:** Do not build your conda environment on `chinook.alaska.edu` and expect it to work on the GPU nodes. The GPU nodes use AMD EPYC processors, which have a different CPU architecture than the standard Chinook login nodes. Environments compiled on the wrong architecture may fail to run. Always build and test your conda environment from `chinookgpu.alaska.edu`.

> **Q: What does "always build and test your conda environment from chinookgpu.alaska.edu" mean? Like do I need to install miniconda on both the CPU and GPU login?**
>
> No, you only install Miniconda once. Because `$HOME` is a **shared filesystem** — the exact same files are visible from every login node — your `~/miniconda3` directory and all your conda environments are accessible whether you're logged into `chinook.alaska.edu` or `chinookgpu.alaska.edu`.
>
> The important thing is **which login node you are on when you run `conda create` and `conda install`**. When conda installs packages like PyTorch, it may compile or download binaries optimized for the CPU architecture it detects. The GPU nodes use AMD EPYC processors, and if you build the environment while logged into a different architecture (the standard CPU login nodes), some binaries may not run correctly on the GPU nodes.
>
> In practice: log into `chinookgpu.alaska.edu`, and from there run your `conda create` and `conda install` commands. The environment gets created in `$HOME` (shared storage) and will be usable from anywhere — you just want the initial build to happen from the right login node. You only need to do this once. After that, `conda activate bubble-mapping` works identically from either login node.

Once connected, your prompt will show the GPU login node:

```
[amwelch3@chinookgpu ~]$
```

This is still a login node — only light tasks here. Do not run training scripts at this prompt.

> **Q: Clarify: do I ever need to "cd into" a compute node or GPU node (in a way that the shell prompt would change)? Or do I just submit jobs with the scheduler from the login node and that's it?**
>
> For **batch jobs** (your main workflow), you never interact with compute nodes directly. You stay on the login node the whole time. You write your SLURM script, run `sbatch train.slurm`, and the job runs on a compute node entirely in the background — your terminal prompt never changes. You check progress by reading the log file (`tail -f logs/train_JOBID.out`) and checking `squeue`, all from the login node.
>
> The **one exception** is an **interactive session** using `srun` (Section 14.5). This command drops you into a live shell on a compute node, and yes, your prompt does change (e.g., from `[amwelch3@chinookgpu ~]$` to `[amwelch3@n155 ~]$`). Use this when you want to:
> - Test a few lines of Python interactively on a real GPU before committing to a batch job
> - Debug a crash by re-running the failing code step by step
> - Run `nvidia-smi` to confirm your GPU is visible
>
> In day-to-day work: 90%+ of the time, `sbatch` from the login node is all you need. `srun` for interactive sessions is a useful but occasional tool, not a regular part of the workflow.

### 14.2 GPU Partitions

There are two GPU partitions on Chinook, corresponding to the two GPU types:

| Partition | GPU Type    | GPU Memory | Max Walltime | Best For                                                           |
| --------- | ----------- | ---------- | ------------ | ------------------------------------------------------------------ |
| `l40s`    | NVIDIA L40S | 48 GB      | 2 days       | Small-to-medium model training, inference, fine-tuning. FP32 only. |
| `h100`    | NVIDIA H100 | 80 GB      | 2 days       | Large model training, FP64, large-scale deep learning.             |

**For your bubble-mapping UNet pipeline:** Either partition will work, but the H100 (`--partition=h100`) is the better choice for full training runs. It has more GPU memory (80 GB vs. 48 GB), supports mixed-precision training more efficiently, and is better suited for deep learning at scale. Use the L40S for quick debug runs or inference.

### 14.3 Recommended SBATCH Flags for GPU Jobs

When submitting GPU jobs, RCS recommends using per-GPU resource specifications rather than total node resources. This is because the GPU nodes are **shared** — multiple users may run on the same node simultaneously.

| Flag | Recommended Value | What It Does |
|---|---|---|
| `--partition=` | `l40s` or `h100` | Selects the GPU type |
| `--gpus=` | `1` (for this pipeline) | Number of GPUs to allocate |
| `--mem-per-gpu=` | `185G` | System RAM per GPU (not GPU VRAM) |
| `--cpus-per-gpu=` | `6` | CPU cores per GPU |

> If you do not specify `--gpus`, SLURM defaults to allocating **all** GPUs on the node. RCS will terminate jobs that request GPUs but leave them idle, so always specify the exact number you need.

### 14.4 Updated SBATCH Template for GPU Training

Replace the training script from Section 9.2 with this corrected version that uses the proper GPU partition names and RCS-recommended resource flags:

```bash
#!/bin/bash
#SBATCH --job-name=bubble-train
#SBATCH --partition=h100
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=185G
#SBATCH --cpus-per-gpu=6
#SBATCH --output=logs/train_%j.out
#SBATCH --mail-user=amwelch3@alaska.edu
#SBATCH --mail-type=BEGIN,END,FAIL

module purge
module load slurm

eval "$(conda shell.bash hook)"
conda activate bubble-mapping

cd ~/bubble-mapping

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"

python training.py

echo "Job finished: $(date)"
```

Submit from the GPU login node:

```bash
module load slurm
sbatch train.slurm
```
> **Q: What is train.slurm? Do I need to save a bash script with the content above as "train" or "train.slurm" somewhere? Do I need to uncomment all the SBATCH lines at the top?**
>
> Yes — `train.slurm` is a plain text file you create yourself. Copy the entire script block above (starting with `#!/bin/bash`) and save it as a file named `train.slurm` in your `~/bubble-mapping/` directory. The `.slurm` extension is just a convention to help you recognize it; you could name it `train.sh` or `submit_training.slurm` — the name doesn't matter to SLURM, only the contents do.
>
> **Do NOT remove the `#` from the `#SBATCH` lines.** This is the most common source of confusion: even though lines starting with `#` are normally comments in bash, SLURM is specifically looking for the pattern `#SBATCH` to read your resource directives. If you remove the `#`, SLURM ignores those lines and your job runs with default (often wrong) settings. Leave them exactly as shown.
>
> To create the file on Chinook, you can use a terminal text editor:
>
> ```bash
> cd ~/bubble-mapping
> nano train.slurm     # or: vi train.slurm
> ```
>
> Paste in the script content, save, and then submit it with:
>
> ```bash
> module load slurm
> sbatch train.slurm
> ```
>
> You will want similar files for `preprocess.slurm` and `eval.slurm` (see Sections 9.1 and 9.3). Keep all three in your `~/bubble-mapping/` directory.
### 14.5 Interactive GPU Session

To test code interactively on a GPU compute node before submitting a batch job:

```bash
# On the L40S partition:
srun -p l40s --gpus=1 --mem-per-gpu=185G --cpus-per-gpu=6 --pty /bin/bash -l

# On the H100 partition:
srun -p h100 --gpus=1 --mem-per-gpu=185G --cpus-per-gpu=6 --pty /bin/bash -l
```

Your prompt will change to show a compute node name (e.g., `n155`), confirming you are on a GPU compute node. From here you can activate your conda environment and run Python interactively.

### 14.6 Verifying Your GPU is Actually Being Used

RCS monitors for jobs that request GPUs but do not use them. Before submitting a long training run, verify your code is actually reaching the GPU using these two methods:

**Method 1 — nvidia-smi** (run from inside an interactive session or at the start of your batch script):

```bash
nvidia-smi
```

This shows all GPUs allocated to you and their current utilization. If `GPU-Util` is 0% during training, something is wrong — your code is running on CPU.

**Method 2 — PyTorch check** (add to the top of your batch script):

```python
# checkGPUs.py
import torch
print("CUDA available:", str(torch.cuda.is_available()))
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_properties(i).name)
```

Expected output on an H100 node with 1 GPU allocated:

```
CUDA available: True
NVIDIA H100 80GB HBM3
```

If `CUDA available: False`, your PyTorch installation may not have CUDA support, or you may need to rebuild your conda environment on `chinookgpu.alaska.edu` (see Section 14.1).

The existing training script already includes a PyTorch CUDA check at startup — make sure to review the log output early in your job to confirm the GPU is visible before waiting hours for a result.

---

## 15. Quick Reference Card

Copy-paste commands for the most common operations:

```bash
# Load SLURM (required before any job commands)
module load slurm

# Activate environment
eval "$(conda shell.bash hook)"
conda activate bubble-mapping

# Submit jobs
sbatch preprocess.slurm
sbatch train.slurm
sbatch eval.slurm

# Monitor
squeue -u $USER                    # Job status
tail -f logs/train_<JOBID>.out     # Live output
scancel <JOBID>                    # Cancel a job
show_storage                       # Check disk quotas

# Copy results to archive
cp -r $CENTER1/data/models/UNET/AE $ARCHIVE/bubble-mapping/models/
```
