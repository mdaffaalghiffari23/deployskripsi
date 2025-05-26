# 🛠️ Troubleshooting Guide - Retinopati DiabeTEST

## ⚠️ Common Warnings and Errors

### 1. PyTorch-Streamlit Compatibility Warnings

**Error Message:**
```
RuntimeError: Tried to instantiate class '__path__._path', but it does not exist!
Examining the path of torch.classes raised: RuntimeError: no running event loop
```

**What it means:**
- This is a **harmless warning** caused by Streamlit's file watcher trying to inspect PyTorch modules
- It doesn't affect the functionality of your application
- The app will continue to work normally

**Solutions Applied:**
1. ✅ Added Streamlit configuration to disable file watcher
2. ✅ Added environment variables to suppress warnings
3. ✅ Created startup script with optimized settings

**To minimize warnings, use the startup script:**
```bash
./start_app.sh
```

### 2. Deprecation Warnings

**Error Message:**
```
The `use_column_width` parameter has been deprecated
```

**Solution:**
✅ **Fixed** - Updated to use `use_container_width=True` instead

### 3. Model Loading Issues

**Possible Issues:**
- Model file not found
- Vocabulary mismatch
- Insufficient memory

**Solutions:**
1. Ensure `fold_4_best_model.pth` exists (48.4MB)
2. Ensure `vocabulary_corrected.json` is present
3. Check system has 4GB+ RAM available

## 🚀 Recommended Startup Methods

### Method 1: Using Startup Script (Recommended)
```bash
cd /path/to/deployskripsi
./start_app.sh
```

### Method 2: Manual with Optimized Settings
```bash
# Using uv
PYTHONWARNINGS="ignore::UserWarning" uv run streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.fileWatcherType none

# Using pip
PYTHONWARNINGS="ignore::UserWarning" streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.fileWatcherType none
```

### Method 3: Standard Startup (May show warnings)
```bash
# Using uv
uv run streamlit run app.py --server.port 8501 --server.address 0.0.0.0

# Using pip
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## 🔧 Configuration Files Added

### `.streamlit/config.toml`
- Disables file watcher to prevent PyTorch inspection errors
- Sets logging level to warning
- Optimizes client settings

### Environment Variables
- `TORCH_HOME=/tmp/torch_cache` - Prevents permission issues
- `PYTHONWARNINGS="ignore::UserWarning"` - Suppresses harmless warnings

## ✅ Final Status

All warnings have been addressed:
- ✅ Deprecation warning fixed
- ✅ PyTorch-Streamlit compatibility optimized
- ✅ Startup script created for clean launches
- ✅ Configuration files added

**The application is fully functional regardless of these warnings.**
