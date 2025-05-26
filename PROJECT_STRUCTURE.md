# Project Structure - Retinopati DiabeTEST

## 📁 Core Application Files
```
├── app.py                           # Main Streamlit application
├── fold_4_best_model.pth           # Trained DenseNet121-LSTM model (48.4MB)
├── vocabulary_corrected.json       # Medical vocabulary (22 words)
└── train.csv                       # Training data reference
```

## 📁 Testing & Validation
```
├── test_image_prediction.py        # End-to-end image testing
├── validate_system.py              # System validation script
└── image/                          # Test fundus images (7 samples)
    ├── 0a09aa7356c0.png
    ├── 0a38b552372d.png
    ├── 0a4e1a29ffff.png
    ├── 0a61bddab956.png
    ├── 0a74c92e287c.png
    ├── 0a85a1e8f9e9.png
    └── 0a9ec1e99ce4.png
```

## 📁 Configuration & Documentation
```
├── .streamlit/
│   └── config.toml                 # Streamlit configuration (reduces warnings)
├── pyproject.toml                  # Dependencies and project config (for uv)
├── requirements.txt                # Dependencies for pip install
├── uv.lock                         # Dependency lock file
├── start_app.sh                    # Optimized startup script
├── README.md                       # Project overview
├── USAGE_GUIDE.md                  # Detailed usage instructions
├── TROUBLESHOOTING.md              # Error solutions and warnings guide
└── PROJECT_STRUCTURE.md            # This file
```

## 📁 Environment Files
```
├── .python-version                 # Python version specification
├── .venv/                          # Virtual environment
└── .git/                           # Git repository
```

## 🚀 Quick Start

### Option 1: Using startup script (recommended - reduces warnings)
```bash
# Run with optimized settings
./start_app.sh
```

### Option 2: Using uv
```bash
# Run the application
uv run streamlit run app.py --server.port 8501 --server.address 0.0.0.0

# Validate system
uv run python validate_system.py

# Test with sample image
uv run python test_image_prediction.py
```

### Option 3: Using pip and Python
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

# Validate system
python validate_system.py

# Test with sample image
python test_image_prediction.py
```