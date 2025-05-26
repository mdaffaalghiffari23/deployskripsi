# 🩺 Retinopati DiabeTEST

Medical image captioning system for diabetic retinopathy detection using DenseNet121-LSTM architecture.

## 🚀 Quick Setup

### Option 1: Using uv (recommended)
```bash
# Clone and run
git clone <repository-url>
cd deployskripsi
uv run streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Option 2: Using pip
```bash
# Clone and install
git clone <repository-url>
cd deployskripsi
pip install -r requirements.txt

# Run application
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## 📚 Documentation

- **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - Detailed usage instructions
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Project organization

## 🧠 Features

- **Medical Image Analysis**: Process fundus images for diabetic retinopathy detection
- **Caption Generation**: Generate medical descriptions in Indonesian
- **Risk Assessment**: Classify severity levels (Normal → Severe DR)
- **Web Interface**: User-friendly Streamlit application
- **Patient Management**: Input patient information and medical reports

## ⚗️ System Requirements

- Python 3.11+
- PyTorch 2.7.0+
- Streamlit 1.45.1+
- 4GB+ RAM (for model loading)

## 🔬 Testing

```bash
# Validate system
python validate_system.py

# Test with sample images
python test_image_prediction.py
```

## ⚠️ Medical Disclaimer

This application is a diagnostic aid tool developed for research purposes. Results do not replace professional medical evaluation. Always consult qualified ophthalmologists for definitive diagnosis.

---

**License**: Research/Educational Use  
**Contact**: For technical support or medical questions, consult the development team or medical professionals respectively.