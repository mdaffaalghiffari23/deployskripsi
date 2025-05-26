# -*- coding: utf-8 -*-
"""
Streamlit app for Diabetic Retinopathy Detection and Caption Generation
"""

import streamlit as st

# Page configuration - MUST be first Streamlit command
st.set_page_config(page_title="Deteksi dan Captioning Retinopati Diabetik", layout="centered")

# Suppress PyTorch warnings and optimize imports
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
os.environ['TORCH_HOME'] = '/tmp/torch_cache'

from PIL import Image
from datetime import date
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import pandas as pd
import itertools
import nltk
import json
import re
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

# Import the model class
class DenseNetLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(DenseNetLSTM, self).__init__()

        # Load pretrained DenseNet121
        densenet = models.densenet121(weights='IMAGENET1K_V1')
        # Remove the final fully connected layer
        self.densenet = nn.Sequential(*list(densenet.children())[:-1])

        # Freeze DenseNet parameters (optional, set to False for fine-tuning)
        for param in self.densenet.parameters():
            param.requires_grad = False

        # Add an adaptive pooling layer to ensure consistent feature size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # DenseNet121 output features
        densenet_output_features = 1024

        # Feature transformation layers
        self.fc1 = nn.Linear(densenet_output_features, embed_size)
        self.bn1d = nn.BatchNorm1d(embed_size)

        # Word embedding
        self.wordembed = nn.Embedding(vocab_size, embed_size)

        # LSTM for caption generation
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)

        # Output layer
        self.fc2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, images, captions):
        """Training forward pass"""
        # Extract features from image using DenseNet
        features = self.densenet(images)

        # Apply adaptive pooling and flatten
        features = self.adaptive_pool(features).squeeze(3).squeeze(2)

        # Apply feature transformation
        features = self.bn1d(self.fc1(features))

        # Prepare captions (remove <end> token for input)
        captions = captions[:, :-1]

        # Embed word indices
        embed = self.wordembed(captions)

        # Concatenate image features with word embeddings
        inputs = torch.cat((features.unsqueeze(1), embed), 1)

        # Pass through LSTM
        lstmout, hidden = self.lstm(inputs)

        # Get outputs
        outputs = self.fc2(lstmout)

        return outputs

    def sample(self, inputs, states=None, max_len=50):
        """Inference forward pass"""
        # Extract features from image using DenseNet
        feat = self.densenet(inputs)

        # Apply adaptive pooling and flatten
        feat = self.adaptive_pool(feat).squeeze(3).squeeze(2)

        # Apply feature transformation
        feat = self.bn1d(self.fc1(feat))

        # Initialize input with image features
        features = feat.unsqueeze(1)

        predicted_sentence = []

        # Generate caption word by word
        for i in range(max_len):
            lstm_out, states = self.lstm(features, states)
            lstm_out = lstm_out.squeeze(1)
            outputs = self.fc2(lstm_out)
            target = outputs.max(1)[1]
            predicted_sentence.append(target.item())

            # Stop if <end> token is predicted
            if target.item() == 2:  # <end> token
                break

            # Update input for next step
            features = self.wordembed(target).unsqueeze(1)

        return predicted_sentence

def load_vocabulary_from_json(vocab_path):
    """Load vocabulary from pre-saved JSON file"""
    try:
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
        
        wordtoid = vocab_data['wordtoid']
        # Convert string keys to int for idtoword
        idtoword = {int(k): v for k, v in vocab_data['idtoword'].items()}
        
        return wordtoid, idtoword
    
    except Exception as e:
        st.error(f"Error loading vocabulary from {vocab_path}: {e}")
        # Fallback minimal vocabulary
        wordtoid = {'<unknown>': 0, '<start>': 1, '<end>': 2, '<pad>': 3,
                   'tidak': 4, 'ada': 5, 'lesi': 6, 'mikroaneurisma': 7, 'hemoragi': 8}
        idtoword = {v: k for k, v in wordtoid.items()}
        return wordtoid, idtoword

def build_vocabulary_from_csv(csv_path):
    """Build vocabulary from training CSV file using exact training process"""
    try:
        df = pd.read_csv(csv_path, usecols=['caption'])
        img_caption = df['caption'].tolist()
        
        # Follow exact training process from densenet121_lstm.py
        imgcaption_dict = {}
        for i, caption in enumerate(img_caption):
            # Remove special characters and normalize
            cleaned_caption = re.sub(r'[^a-zA-Z0-9\s]', '', caption).strip().lower().replace('  ', ' ')
            imgcaption_dict[f'img_{i}'] = cleaned_caption

        # Tokenize captions
        imgcaption_wordtoken = []
        for caption in imgcaption_dict.values():
            imgcaption_wordtoken.append(nltk.word_tokenize(caption))

        # Create vocabulary exactly as in training
        alltokens = itertools.chain.from_iterable(imgcaption_wordtoken)
        wordtoid = {token: idx for idx, token in enumerate(set(alltokens))}

        alltokens = itertools.chain.from_iterable(imgcaption_wordtoken)
        idtoword = [token for idx, token in enumerate(set(alltokens))]

        # Add special tokens
        wordtoid['<unknown>'] = 0
        wordtoid['<start>'] = 1
        wordtoid['<end>'] = 2
        wordtoid['<pad>'] = 3

        # Adjust token IDs (add 4 to existing tokens)
        for token in list(wordtoid.keys()):
            if token not in ['<unknown>', '<start>', '<end>', '<pad>']:
                old_id = wordtoid[token]
                wordtoid[token] = old_id + 4

        # Create ID to word mapping
        idtoword_dict = {}
        cnt = 4
        for token in idtoword:
            idtoword_dict[cnt] = token
            cnt += 1
        idtoword_dict[0] = '<unknown>'
        idtoword_dict[1] = '<start>'
        idtoword_dict[2] = '<end>'
        idtoword_dict[3] = '<pad>'
        
        return wordtoid, idtoword_dict
    
    except Exception as e:
        st.error(f"Error building vocabulary: {e}")
        # Fallback minimal vocabulary
        wordtoid = {'<unknown>': 0, '<start>': 1, '<end>': 2, '<pad>': 3,
                   'tidak': 4, 'ada': 5, 'lesi': 6, 'mikroaneurisma': 7, 'hemoragi': 8}
        idtoword = {v: k for k, v in wordtoid.items()}
        return wordtoid, idtoword

@st.cache_resource
def load_model_and_tokenizer(model_path, device='cpu'):
    """Load the trained model and tokenizer"""
    try:
        # Try to load vocabulary from JSON file first
        vocab_path = "vocabulary_corrected.json"
        if os.path.exists(vocab_path):
            wordtoid, idtoword = load_vocabulary_from_json(vocab_path)
        else:
            # Build vocabulary from training data as fallback
            csv_path = "train.csv"
            if os.path.exists(csv_path):
                wordtoid, idtoword = build_vocabulary_from_csv(csv_path)
            else:
                st.warning("No vocabulary file found, using minimal vocabulary")
                wordtoid = {'<unknown>': 0, '<start>': 1, '<end>': 2, '<pad>': 3,
                           'tidak': 4, 'ada': 5, 'lesi': 6, 'mikroaneurisma': 7, 'hemoragi': 8}
                idtoword = {v: k for k, v in wordtoid.items()}
        
        # Model hyperparameters (should match training)
        embed_size = 256
        hidden_size = 512
        vocab_size = len(wordtoid)
        
        # Initialize model
        model = DenseNetLSTM(embed_size, hidden_size, vocab_size)
        
        # Load trained weights
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    # Checkpoint contains model_state_dict
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    # Checkpoint contains state_dict
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    # Checkpoint is direct state dict
                    model.load_state_dict(checkpoint)
            else:
                # Checkpoint is direct state dict (not wrapped in dict)
                model.load_state_dict(checkpoint)
                
        else:
            st.error(f"Model file {model_path} not found!")
            return None, None
        
        model.eval()
        model.to(device)
        
        # Show setup complete message only when everything is loaded successfully
        st.success("Setup Complete ✅")
        
        return model, idtoword
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def generate_caption(model, idtoword, input_tensor, max_len=20):
    """Generate caption for an image"""
    try:
        with torch.no_grad():
            # Generate token sequence
            predicted_tokens = model.sample(input_tensor, max_len=max_len)
            
            # Convert tokens to words
            caption_words = []
            for token_id in predicted_tokens:
                if token_id in idtoword:
                    word = idtoword[token_id]
                    if word not in ['<start>', '<end>', '<pad>', '<unknown>']:
                        caption_words.append(word)
                
            caption = ' '.join(caption_words) if caption_words else "Tidak dapat menghasilkan caption"
            return caption.capitalize()
    
    except Exception as e:
        st.error(f"Error generating caption: {e}")
        return "Error generating caption"

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, idtoword = load_model_and_tokenizer("fold_4_best_model.pth", device=device)

# Image transform (must match preprocessing during training)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

st.title("🩺 Retinopati DiabeTEST")

# Check if model loaded successfully
if model is None or idtoword is None:
    st.error("Failed to load model. Please check if fold_4_best_model.pth and train.csv are available.")
    st.stop()

# Input Form
with st.form("fundus_form"):
    nama_pasien = st.text_input("Nama Pasien")
    tanggal_lahir = st.date_input("Tanggal Lahir Pasien", min_value=date(1900, 1, 1))
    dokter = st.selectbox("Dokter Mata yang Bertugas", ["Dr Andi", "Dr Anggun"])
    uploaded_image = st.file_uploader("Unggah Citra Fundus", type=["jpg", "jpeg", "png"])

    submitted = st.form_submit_button("Prediksi")

# Output
if submitted and uploaded_image is not None:
    # Show processing message
    with st.spinner('Menganalisis citra fundus...'):
        try:
            # Load and display image
            image = Image.open(uploaded_image).convert("RGB")
            st.image(image, caption="Citra Fundus yang Diunggah", use_container_width=True)

            # Calculate age
            today = date.today()
            usia = today.year - tanggal_lahir.year - ((today.month, today.day) < (tanggal_lahir.month, tanggal_lahir.day))

            # Transform image
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            # Generate caption
            caption = generate_caption(model, idtoword, input_tensor)

            # Medical recommendation based on caption
            caption_lower = caption.lower()
            if "tidak ada lesi" in caption_lower:
                rekomendasi = "Mata Normal - Tidak terdeteksi tanda-tanda retinopati diabetik"
                status_color = "green"
            elif any(keyword in caption_lower for keyword in ["mikroaneurisma", "hemoragi ringan"]):
                rekomendasi = "Retinopati Diabetik Ringan - Perlu monitoring berkala"
                status_color = "orange"
            elif any(keyword in caption_lower for keyword in ["eksudat keras", "venous beading"]):
                rekomendasi = "Retinopati Diabetik Sedang - Perlu konsultasi lebih lanjut"
                status_color = "orange"
            elif any(keyword in caption_lower for keyword in ["neovaskularisasi", "vitreous hemorrhage", "fibrovascular"]):
                rekomendasi = "Retinopati Diabetik Proliferatif - Perlu penanganan segera"
                status_color = "red"
            else:
                rekomendasi = "Perlu evaluasi lebih lanjut oleh dokter spesialis mata"
                status_color = "blue"

            # Display results
            st.subheader("📝 Hasil Analisis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Nama Pasien:** {nama_pasien}")
                st.write(f"**Usia Pasien:** {usia} tahun")
                st.write(f"**Dokter Mata:** {dokter}")
            
            with col2:
                st.write(f"**Tanggal Analisis:** {today.strftime('%d/%m/%Y')}")
                st.write(f"**Model:** DenseNet121-LSTM")
                st.write(f"**Status Analisis:** Selesai")

            st.write(f"**Deskripsi Temuan:** _{caption}_")
            
            # Display recommendation with color coding
            if status_color == "green":
                st.success(f"**Rekomendasi:** {rekomendasi}")
            elif status_color == "orange":
                st.warning(f"**Rekomendasi:** {rekomendasi}")
            elif status_color == "red":
                st.error(f"**Rekomendasi:** {rekomendasi}")
            else:
                st.info(f"**Rekomendasi:** {rekomendasi}")
            
            # Additional information
            st.subheader("ℹ️ Informasi Tambahan")
            st.write("""
            **Catatan Penting:**
            - Hasil analisis ini adalah bantuan diagnostik dan tidak menggantikan penilaian klinis dokter
            - Untuk diagnosis definitif, diperlukan pemeriksaan oftalmologi lengkap
            - Segera konsultasi ke dokter spesialis mata jika terdapat gejala gangguan penglihatan
            
            **Tentang Retinopati Diabetik:**
            - Komplikasi diabetes yang menyerang pembuluh darah retina
            - Dapat menyebabkan kebutaan jika tidak ditangani dengan baik
            - Deteksi dini sangat penting untuk mencegah komplikasi serius
            """)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses gambar: {str(e)}")
            st.write("Silakan coba lagi dengan gambar yang berbeda atau hubungi administrator.")

elif submitted and uploaded_image is None:
    st.warning("Silakan unggah citra fundus terlebih dahulu.")

# Sidebar with additional information
with st.sidebar:
    st.header("📋 Panduan Penggunaan")
    st.write("""
    1. **Isi data pasien** dengan lengkap
    2. **Unggah citra fundus** dalam format JPG, JPEG, atau PNG
    3. **Klik tombol Prediksi** untuk memulai analisis
    4. **Lihat hasil** dan rekomendasi yang diberikan
    """)
    
    st.header("🔧 Spesifikasi Teknis")
    st.write(f"""
    - **Model:** DenseNet121 + LSTM
    - **Device:** {device.type.upper()}
    - **Input Size:** 224x224 pixels
    - **Vocabulary Size:** {len(idtoword) if idtoword else 'N/A'}
    """)
    
    st.header("⚠️ Disclaimer")
    st.write("""
    Aplikasi ini adalah alat bantu diagnostik yang dikembangkan untuk tujuan penelitian. 
    Hasil analisis tidak menggantikan penilaian medis profesional.
    """)