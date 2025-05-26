#!/usr/bin/env python3
"""
Test script to verify image prediction with a real fundus image
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import json
import os
from PIL import Image

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

def load_vocabulary_from_json(vocab_path):
    """Load vocabulary from pre-saved JSON file"""
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    
    wordtoid = vocab_data['wordtoid']
    # Convert string keys to int for idtoword
    idtoword = {int(k): v for k, v in vocab_data['idtoword'].items()}
    
    return wordtoid, idtoword

def generate_caption(model, idtoword, input_tensor, max_len=20):
    """Generate caption for an image"""
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

def main():
    print("🩺 Testing Real Image Prediction")
    print("=" * 40)
    
    # Load vocabulary
    wordtoid, idtoword = load_vocabulary_from_json("vocabulary_corrected.json")
    print(f"Vocabulary loaded with {len(wordtoid)} words")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model parameters
    embed_size = 256
    hidden_size = 512
    vocab_size = len(wordtoid)
    
    # Initialize and load model
    model = DenseNetLSTM(embed_size, hidden_size, vocab_size)
    checkpoint = torch.load("fold_4_best_model.pth", map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    model.to(device)
    print("✅ Model loaded successfully")
    
    # Create image transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Test with available images
    image_folder = "image"
    if os.path.exists(image_folder):
        image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if image_files:
            # Test with first available image
            test_image_path = os.path.join(image_folder, image_files[0])
            print(f"\n🔍 Testing with image: {test_image_path}")
            
            # Load and process image
            image = Image.open(test_image_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            print(f"Input tensor shape: {input_tensor.shape}")
            
            # Generate caption
            caption = generate_caption(model, idtoword, input_tensor)
            print(f"Generated caption: '{caption}'")
            
            # Medical recommendation
            caption_lower = caption.lower()
            if "tidak ada lesi" in caption_lower:
                recommendation = "Mata Normal - Tidak terdeteksi tanda-tanda retinopati diabetik"
            elif any(keyword in caption_lower for keyword in ["mikroaneurisma", "hemoragi ringan"]):
                recommendation = "Retinopati Diabetik Ringan - Perlu monitoring berkala"
            elif any(keyword in caption_lower for keyword in ["eksudat keras", "venous beading"]):
                recommendation = "Retinopati Diabetik Sedang - Perlu konsultasi lebih lanjut"
            elif any(keyword in caption_lower for keyword in ["neovaskularisasi", "vitreous hemorrhage", "fibrovascular"]):
                recommendation = "Retinopati Diabetik Proliferatif - Perlu penanganan segera"
            else:
                recommendation = "Perlu evaluasi lebih lanjut oleh dokter spesialis mata"
            
            print(f"Medical recommendation: {recommendation}")
            
        else:
            print("❌ No image files found in the image folder")
    else:
        print("❌ Image folder not found")
    
    print("\n✅ Test completed!")
    print("🚀 Your Streamlit app is ready for use!")
    print("📱 Access it at: http://localhost:8501")

if __name__ == "__main__":
    main()
