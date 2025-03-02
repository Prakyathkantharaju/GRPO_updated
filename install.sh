#!/bin/bash
set -e

# Set terminal color
echo 'export TERM=xterm-256color' >> ~/.bashrc

# Update and install system dependencies
echo "Updating system packages..."
sudo apt update
sudo apt install -y python3-pip python3-dev build-essential git curl

# Install uv
echo "Installing uv package manager..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH for this session
export PATH="$HOME/.cargo/bin:$PATH"

# Create virtual environment
echo "Creating virtual environment..."
python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
uv venv .venv --python="$python_version"

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install PyTorch ecosystem
echo "Installing PyTorch ecosystem..."
uv pip install torch torchvision torchaudio

# Install dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

# Install flash-attention with special flags
echo "Installing flash-attention with special flags..."
uv pip install flash-attn --no-build-isolation

# Install additional build dependencies
echo "Installing additional build dependencies..."
uv pip install ninja packaging

# Install Jupyter notebook environment
echo "Installing Jupyter environment..."
uv pip install jupyter ipykernel

# Register the kernel
echo "Registering Jupyter kernel..."
python -m ipykernel install --user --name=grpo-env --display-name="GRPO Environment"

# Verify installations
echo "Verifying installations..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import trl; print(f'TRL: {trl.__version__}')"
python -c "import vllm; print(f'VLLM: {vllm.__version__}')" || echo "VLLM not installed correctly"
python -c "import flash_attn; print(f'Flash Attention: {flash_attn.__version__}')" || echo "Flash Attention not installed correctly"

echo "Installation complete! Activate the environment with: source .venv/bin/activate" 