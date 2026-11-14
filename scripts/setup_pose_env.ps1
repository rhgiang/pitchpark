# Creates .venv-pose (torch lives ONLY here; the main .venv and src/pitchsim never import torch).
# Why a separate venv: the 3D lifters (GVHMR / HMR 2.0) need torch + CUDA 12.8 (the RTX 5060 Ti is
# Blackwell, sm_120, which only cu128+ wheels support).  Run from the repo root:
#   powershell -File scripts/setup_pose_env.ps1
$ErrorActionPreference = "Stop"
$env:UV_HTTP_TIMEOUT = "300"   # the torch wheel is 2.7 GB; the default timeout drops on slow links
# Python 3.12 from python.org: uv-managed pythons are blocked by Application Control on this machine.
uv venv .venv-pose --python "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
$py = ".venv-pose\Scripts\python.exe"
uv pip install --python $py torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python $py pip setuptools wheel
# chumpy (needed by SMPL .pkl loading) only builds without build isolation
uv pip install --python $py --no-build-isolation chumpy
uv pip install --python $py numpy scipy opencv-python einops timm lightning hydra-core hydra-zen hydra_colorlog rich smplx trimesh gdown pytorch-lightning tqdm joblib scikit-image yacs
& $py -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
