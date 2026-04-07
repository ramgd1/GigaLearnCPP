# GigaLearnCPP

A high-performance C++ Rocket League bot training framework built on top of [RocketSim](https://github.com/ZealanL/RocketSim) and PPO reinforcement learning.

---

## Disclaimers

This guide walks through installing and running GigaLearnCPP from scratch.

If you do not use the following, parts of this guide may not apply directly:

| Requirement | Notes |
|---|---|
| **Visual Studio 2022** | Recommended. Must include C++ Desktop Development workload. |
| **CUDA 12.8 / 12.9** | Recommended. Other CUDA 12.x (12.6+) versions may work with the correct LibTorch build. Not required for CPU-only. |
| **Windows** | Recommended. Linux is possible but not covered here. |
| **Python 3.11** | Required for metrics via wandb. |
| **Git** | Required to clone. |

---

## Step 1 — Clone the Repository

Open a command prompt and run:

```bat
cd %USERPROFILE%\Downloads
git clone https://github.com/ramgd1/GigaLearnCPP --recurse-submodules
```

> Git must be installed and available in PATH.
> Download: https://git-scm.com/download/win

---

## Step 2 — Install CUDA (GPU only)

If you have an NVIDIA GPU:

1. Download CUDA 12.8: https://developer.nvidia.com/cuda-12-8-0-download-archive
2. Select: **Windows → x64 → Your OS version → exe (local) → Download**
3. Run the installer and follow the prompts.

Skip this step if you plan to run on CPU only (pass `--cpu` at launch).

---

## Step 3 — Download LibTorch

1. Go to: https://pytorch.org/get-started/locally/
2. Select:
   - **Build:** Stable
   - **OS:** Windows
   - **Package:** LibTorch
   - **Language:** C++ / Java
   - **Compute Platform:** CUDA 12.8 (or CPU if no GPU)
3. Download the **Release** zip (not Debug, not Nightly).

---

## Step 4 — Place LibTorch into the Project

Extract the zip and place the `libtorch` folder at:

```
GigaLearnCPP\GigaLearnCPP\libtorch\
```

The folder must contain `libtorch\share\cmake\Torch\TorchConfig.cmake` — that's how CMake finds it.

---

## Step 5 — Open in VS2022

1. Make sure VS2022 is installed with the **Desktop development with C++** workload.
2. Open VS2022 → **Open a local folder** → select the `GigaLearnCPP` root folder.

---

## Step 6 — Configure the Build

1. Click the config dropdown at the top (usually `x64-Debug`) → **Manage Configurations**
2. Click the green `+` → select **x64-Debug** as base → this creates `x64-Debug (2)`
3. Rename it to: `x64-RelWithDebInfo`
4. Set **Configuration type:** `RelWithDebInfo`
5. Close `CMakeSettings.json` and click **Save**
6. Switch the dropdown to **x64-RelWithDebInfo** — CMake will reconfigure (normal)

![Build config gif](readmeres/1222(1).gif)

> **Tip:** You can also use **Release** for maximum performance (no debug symbols).

---

## Step 7 — Build

In VS2022: **Build → Build All** (or press `Ctrl+Shift+B`).

Build output lands in:
```
GigaLearnCPP\out\build\x64-RelWithDebInfo\
```

---

## Step 8 — Prepare the Build Output Folder

Before running, make sure the following exist in the build output folder:

| Item | Required? |
|---|---|
| `collision_meshes\` | **Mandatory** — RocketSim needs it to simulate |
| `RocketSimVis` (or similar visualizer) | Optional — for rendering |

The `collision_meshes` folder is in the repo root and gets found automatically at startup (the code searches `./`, `../`, `../../`).

---

## Running

Navigate to the build output folder and run the executable. Common flags:

```bat
# CPU only (works on any machine, no GPU needed)
GigaLearnBot.exe --cpu

# GPU (requires CUDA and a supported LibTorch build)
GigaLearnBot.exe --gpu

# Start fresh (ignore existing checkpoints)
GigaLearnBot.exe --no-load

# Load from a specific checkpoint folder
GigaLearnBot.exe --checkpoint path\to\checkpoints

# Override number of parallel game instances
GigaLearnBot.exe --num-games 256

# Enable rendering (slows training significantly)
GigaLearnBot.exe --render --render-timescale 4.0

# Transfer learn from a checkpoint
GigaLearnBot.exe --tl path\to\teacher_checkpoint

# Enable wandb metrics (requires Python + wandb installed)
GigaLearnBot.exe --send-metrics

# Disable reward metrics logging
GigaLearnBot.exe --no-add-rewards
```

See the [Wiki](WIKI.md) for a full breakdown of all flags, config options, rewards, and more.

---

## Tips

- **Reconfigure CMake:** `Project → Configure Cache` in VS2022
- **CMake broken:** Delete the `.vs` folder — VS2022 will regenerate everything
- **RTX 50 / Blackwell GPUs:** The prebuilt LibTorch doesn't include `sm_120`. Use `--cpu` or build LibTorch from source with the correct `TORCH_CUDA_ARCH_LIST`.
- **Checkpoints** are saved automatically under `checkpoints\` in the build folder every 1M timesteps (configurable).
