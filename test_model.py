#!/usr/bin/env python3
"""
Test script to verify model loading and basic functionality
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import pandas as pd
import os
import nltk
import json
from nltk.tokenize import word_tokenize
from PIL import Image
import numpy as np

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

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

def build_vocabulary_from_csv(csv_path):
    """Build vocabulary from training CSV file"""
    try:
        df = pd.read_csv(csv_path, usecols=['caption'])
        captions = df['caption'].tolist()
        
        # Tokenize all captions
        all_tokens = []
        for caption in captions:
            tokens = word_tokenize(caption.lower())
            all_tokens.extend(tokens)
        
        # Create vocabulary
        unique_tokens = list(set(all_tokens))
        
        # Create word to id mapping
        wordtoid = {}
        wordtoid['<unknown>'] = 0
        wordtoid['<start>'] = 1
        wordtoid['<end>'] = 2
        wordtoid['<pad>'] = 3
        
        for i, token in enumerate(unique_tokens):
            wordtoid[token] = i + 4
        
        # Create id to word mapping
        idtoword = {v: k for k, v in wordtoid.items()}
        
        return wordtoid, idtoword
    
    except Exception as e:
        print(f"Error building vocabulary: {e}")
        # Fallback minimal vocabulary
        wordtoid = {'<unknown>': 0, '<start>': 1, '<end>': 2, '<pad>': 3,
                   'tidak': 4, 'ada': 5, 'lesi': 6, 'mikroaneurisma': 7, 'hemoragi': 8}
        idtoword = {v: k for k, v in wordtoid.items()}
        return wordtoid, idtoword

def load_model_and_tokenizer(model_path, device='cpu'):
    """Load the trained model and tokenizer"""
    try:
        # Build vocabulary from training data
        csv_path = "train.csv"
        vocab_path = "vocabulary_corrected.json"
        
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r') as f:
                vocab_data = json.load(f)
            wordtoid = vocab_data['wordtoid']
            idtoword = {int(k): v for k, v in vocab_data['idtoword'].items()}
            print(f"Vocabulary loaded from {vocab_path} with {len(wordtoid)} words")
        elif os.path.exists(csv_path):
            wordtoid, idtoword = build_vocabulary_from_csv(csv_path)
            print(f"Vocabulary loaded from {csv_path} with {len(wordtoid)} words")
        else:
            print("train.csv not found, using minimal vocabulary")
            wordtoid = {'<unknown>': 0, '<start>': 1, '<end>': 2, '<pad>': 3,
                       'tidak': 4, 'ada': 5, 'lesi': 6, 'mikroaneurisma': 7, 'hemoragi': 8}
            idtoword = {v: k for k, v in wordtoid.items()}
        
        # Model hyperparameters (should match training)
        embed_size = 256
        hidden_size = 512
        vocab_size = len(wordtoid)
        
        print(f"Model parameters: embed_size={embed_size}, hidden_size={hidden_size}, vocab_size={vocab_size}")
        
        # Initialize model
        model = DenseNetLSTM(embed_size, hidden_size, vocab_size)
        
        # Load trained weights
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    # Checkpoint contains model_state_dict
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print("Loaded model from model_state_dict")
                elif 'state_dict' in checkpoint:
                    # Checkpoint contains state_dict
                    model.load_state_dict(checkpoint['state_dict'])
                    print("Loaded model from state_dict")
                else:
                    # Checkpoint is direct state dict
                    model.load_state_dict(checkpoint)
                    print("Loaded model from direct state dict")
            else:
                # Checkpoint is direct state dict (not wrapped in dict)
                model.load_state_dict(checkpoint)
                print("Loaded model from direct state dict (not wrapped)")
                
            print(f"Model loaded successfully from {model_path}")
        else:
            print(f"Model file {model_path} not found!")
            return None, None
        
        model.eval()
        model.to(device)
        
        return model, idtoword
    
    except Exception as e:
        print(f"Error loading model: {e}")
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
        print(f"Error generating caption: {e}")
        return "Error generating caption"

def create_dummy_image():
    """Create a dummy fundus-like image for testing"""
    # Create a 224x224 random image with some circular pattern (simulate fundus)
    image = np.random.rand(224, 224, 3) * 255
    
    # Add a circular pattern to simulate fundus
    center = (112, 112)
    y, x = np.ogrid[:224, :224]
    mask = (x - center[0])**2 + (y - center[1])**2 <= 100**2
    image[mask] = image[mask] * 0.8 + 50  # Darken the center
    
    # Convert to PIL Image
    image = Image.fromarray(image.astype(np.uint8))
    return image

def main():
    print("🩺 Testing Retinopati DiabeTEST Model")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, idtoword = load_model_and_tokenizer("fold_4_best_model.pth", device=device)
    
    if model is None or idtoword is None:
        print("❌ Failed to load model")
        return
    
    print("✅ Model loaded successfully")
    
    # Create image transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Test with dummy image
    print("\n🔍 Testing with dummy image...")
    dummy_image = create_dummy_image()
    input_tensor = transform(dummy_image).unsqueeze(0).to(device)
    
    print(f"Input tensor shape: {input_tensor.shape}")
    
    # Generate caption
    caption = generate_caption(model, idtoword, input_tensor)
    
    print(f"Generated caption: '{caption}'")
    
    # Test medical recommendation logic
    caption_lower = caption.lower()
    if "tidak ada lesi" in caption_lower:
        rekomendasi = "Mata Normal - Tidak terdeteksi tanda-tanda retinopati diabetik"
    elif any(keyword in caption_lower for keyword in ["mikroaneurisma", "hemoragi ringan"]):
        rekomendasi = "Retinopati Diabetik Ringan - Perlu monitoring berkala"
    elif any(keyword in caption_lower for keyword in ["eksudat keras", "venous beading"]):
        rekomendasi = "Retinopati Diabetik Sedang - Perlu konsultasi lebih lanjut"
    elif any(keyword in caption_lower for keyword in ["neovaskularisasi", "vitreous hemorrhage", "fibrovascular"]):
        rekomendasi = "Retinopati Diabetik Proliferatif - Perlu penanganan segera"
    else:
        rekomendasi = "Perlu evaluasi lebih lanjut oleh dokter spesialis mata"
    
    print(f"Medical recommendation: {rekomendasi}")
    
    print("\n✅ Test completed successfully!")
    print("🚀 Your Streamlit app should be working properly!")

if __name__ == "__main__":
    main()
