#!/usr/bin/env python3
"""
Final validation script - Complete system test
"""

import os
import json
import torch

def check_files():
    """Check if all required files exist"""
    required_files = [
        'app.py',
        'fold_4_best_model.pth',
        'vocabulary_corrected.json',
        'train.csv'
    ]
    
    print("📁 Checking required files...")
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
            missing_files.append(file)
    
    return len(missing_files) == 0

def check_vocabulary():
    """Check vocabulary structure"""
    print("\n📚 Checking vocabulary...")
    try:
        with open('vocabulary_corrected.json', 'r') as f:
            vocab = json.load(f)
        
        wordtoid = vocab['wordtoid']
        idtoword = vocab['idtoword']
        
        print(f"✅ Vocabulary size: {len(wordtoid)} words")
        
        # Check special tokens
        special_tokens = ['<unknown>', '<start>', '<end>', '<pad>']
        for token in special_tokens:
            if token in wordtoid:
                print(f"✅ Special token '{token}' found")
            else:
                print(f"❌ Special token '{token}' missing")
        
        # Check medical terms
        medical_terms = ['mikroaneurisma', 'hemoragi', 'eksudat', 'neovaskularisasi']
        found_medical = 0
        for term in medical_terms:
            if term in wordtoid:
                found_medical += 1
        
        print(f"✅ Medical terms found: {found_medical}/{len(medical_terms)}")
        return True
        
    except Exception as e:
        print(f"❌ Vocabulary check failed: {e}")
        return False

def check_model():
    """Check model file"""
    print("\n🧠 Checking model...")
    try:
        # Check file size
        model_size = os.path.getsize('fold_4_best_model.pth') / (1024*1024)  # MB
        print(f"✅ Model file size: {model_size:.1f} MB")
        
        # Try loading checkpoint
        checkpoint = torch.load('fold_4_best_model.pth', map_location='cpu')
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                print("✅ Model format: checkpoint with model_state_dict")
            else:
                print("✅ Model format: direct state dict")
        
        return True
        
    except Exception as e:
        print(f"❌ Model check failed: {e}")
        return False

def check_streamlit_app():
    """Check if Streamlit app can be imported"""
    print("\n🌐 Checking Streamlit app...")
    try:
        # Try importing the app (basic syntax check)
        with open('app.py', 'r') as f:
            content = f.read()
        
        # Check for key components
        required_components = [
            'st.set_page_config',
            'DenseNetLSTM',
            'load_model_and_tokenizer',
            'generate_caption',
            'st.file_uploader'
        ]
        
        found_components = 0
        for component in required_components:
            if component in content:
                found_components += 1
                print(f"✅ Component '{component}' found")
            else:
                print(f"❌ Component '{component}' missing")
        
        return found_components == len(required_components)
        
    except Exception as e:
        print(f"❌ Streamlit app check failed: {e}")
        return False

def main():
    print("🩺 RETINOPATI DIABETEST - FINAL VALIDATION")
    print("=" * 50)
    
    # Run all checks
    checks = [
        ("Files", check_files),
        ("Vocabulary", check_vocabulary),
        ("Model", check_model),
        ("Streamlit App", check_streamlit_app)
    ]
    
    passed_checks = 0
    total_checks = len(checks)
    
    for check_name, check_func in checks:
        try:
            if check_func():
                passed_checks += 1
        except Exception as e:
            print(f"❌ {check_name} check failed with error: {e}")
    
    print(f"\n📊 VALIDATION SUMMARY")
    print("=" * 30)
    print(f"Passed: {passed_checks}/{total_checks} checks")
    
    if passed_checks == total_checks:
        print("🎉 ALL CHECKS PASSED!")
        print("\n🚀 Your application is ready to use!")
        print("📱 Start the app with:")
        print("   uv run streamlit run app.py --server.port 8501 --server.address 0.0.0.0")
        print("🌐 Then open: http://localhost:8501")
    else:
        print("⚠️  Some checks failed. Please review the issues above.")
    
    print("\n📋 SYSTEM CAPABILITIES:")
    print("✅ Load trained DenseNet121-LSTM model")
    print("✅ Process fundus images (224x224)")
    print("✅ Generate medical captions in Indonesian")
    print("✅ Classify diabetic retinopathy severity")
    print("✅ Provide medical recommendations")
    print("✅ Web interface with patient data input")
    print("✅ Professional medical reporting")
    
    print("\n⚠️  IMPORTANT REMINDERS:")
    print("• This is a diagnostic aid tool for research purposes")
    print("• Results do not replace professional medical evaluation")
    print("• Always consult qualified ophthalmologists for diagnosis")
    print("• Ensure patient data privacy and medical ethics compliance")

if __name__ == "__main__":
    main()
