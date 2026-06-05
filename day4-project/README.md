# Day 4 Project

Small grammar-correction CLI that uses a T5 model.

## Prerequisites
- Python 3.8+ installed
- Git (optional)

## Setup (recommended)
1. Create a virtual environment and install dependencies (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Activate the environment and install dependencies (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Or run the provided setup script which creates the venv and installs packages:

```powershell
.\setup_venv.ps1
```

2. Run the app:

```powershell
.\.venv\Scripts\python.exe main.py
```

Notes:
- The project uses `torch`, `nltk`, and `transformers`. Installing `torch` may require picking the correct wheel for your platform/GPU — see https://pytorch.org/ for the recommended install command.
- `main.py` will download required NLTK data on first run.
