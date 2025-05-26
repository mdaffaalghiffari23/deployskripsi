# Retinopati DiabeTEST - Usage Guide

## 🚀 Quick Start

### Running the Application

1. **Start the Streamlit app:**
   ```bash
   cd /home/tulus-setiawan/Documents/Tulus/deployskripsi
   uv run streamlit run app.py --server.port 8501 --server.address 0.0.0.0
   ```

2. **Access the application:**
   - Open your browser and go to: `http://localhost:8501`
   - The app will load with the medical interface

### Using the Interface

1. **Patient Information:**
   - Enter patient name
   - Select patient birth date
   - Choose the attending eye doctor

2. **Image Upload:**
   - Click "Unggah Citra Fundus"
   - Select a fundus image (JPG, JPEG, or PNG)
   - Supported formats: `.jpg`, `.jpeg`, `.png`

3. **Analysis:**
   - Click "Prediksi" button
   - Wait for the analysis to complete
   - View the results and medical recommendations

## 🔧 Technical Details

### Model Architecture
- **Vision Model:** DenseNet121 (pre-trained on ImageNet)
- **Language Model:** LSTM for caption generation
- **Input Size:** 224x224 pixels
- **Vocabulary Size:** 22 words (specialized medical terms)

### Medical Classifications
The system can detect and classify:

1. **Grade 0 - Normal:** "Tidak ada lesi"
2. **Grade 1 - Mild DR:** "Mikroaneurisma, hemoragi ringan"
3. **Grade 2 - Moderate DR:** "Eksudat keras, venous beading"
4. **Grade 3 - Severe DR:** "Hemoragi banyak, IRMA, eksudat lunak"
5. **Grade 4 - Proliferative DR:** "Neovaskularisasi, vitreous hemorrhage, fibrovascular proliferation"

### Key Files
- `app.py` - Main Streamlit application
- `fold_4_best_model.pth` - Trained model weights
- `vocabulary_corrected.json` - Medical vocabulary
- `train.csv` - Training data with captions

## 🩺 Medical Terminology

### Indonesian Medical Terms
- **Mikroaneurisma** - Microaneurysms
- **Hemoragi** - Hemorrhages
- **Eksudat keras** - Hard exudates
- **Eksudat lunak** - Soft exudates
- **Venous beading** - Venous beading
- **Neovaskularisasi** - Neovascularization
- **IRMA** - Intraretinal microvascular abnormalities

## ⚠️ Important Notes

### Medical Disclaimer
- This application is a diagnostic aid tool developed for research purposes
- Results do not replace professional medical evaluation
- Consult an ophthalmologist for definitive diagnosis
- Seek immediate medical attention if experiencing vision problems

### Technical Requirements
- Python 3.11+
- PyTorch with CPU/GPU support
- Streamlit for web interface
- PIL for image processing
- NLTK for text processing

## 🛠️ Troubleshooting

### Common Issues

1. **Model Loading Error:**
   - Ensure `fold_4_best_model.pth` exists
   - Check `vocabulary_corrected.json` is present
   - Verify all dependencies are installed

2. **Image Processing Error:**
   - Use clear, well-lit fundus images
   - Ensure image format is supported
   - Check image is not corrupted

3. **Streamlit Configuration Error:**
   - Make sure `st.set_page_config()` is the first Streamlit command
   - Restart the application if needed

### Testing the Setup
Run the test scripts to verify everything works:
```bash
# Test model loading and vocabulary
uv run python test_model.py

# Test with real image
uv run python test_image_prediction.py
```

## 📊 Model Performance
- The model was trained using K-fold cross-validation
- Achieves good performance on diabetic retinopathy grading
- Optimized for Indonesian medical terminology

## 🔄 Development Workflow

### For Updates
1. Update model: Replace `fold_4_best_model.pth`
2. Update vocabulary: Modify `vocabulary_corrected.json`
3. Test changes: Run test scripts
4. Deploy: Restart Streamlit app

### For New Features
1. Modify `app.py` for UI changes
2. Update medical classification logic as needed
3. Test thoroughly with various image types
4. Document any changes

---

**Contact:** For technical support or medical questions, consult the development team or medical professionals respectively.
