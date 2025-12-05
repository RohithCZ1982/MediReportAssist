#!/usr/bin/env python3
"""
Setup verification script
Checks if all dependencies and requirements are met
"""

import sys
import os
from pathlib import Path

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

def check_python_version():
    """Check Python version"""
    if sys.version_info < (3, 8):
        print("[FAIL] Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"[OK] Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'chromadb',
        'sentence_transformers',
        'pypdf2',
        'docx',
        'requests'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'docx':
                __import__('docx')
            elif package == 'pypdf2':
                # PyPDF2 package is imported as PyPDF2 (capital letters)
                __import__('PyPDF2')
            else:
                __import__(package)
            print(f"[OK] {package} is installed")
        except ImportError:
            print(f"[FAIL] {package} is NOT installed")
            missing.append(package)
    
    return len(missing) == 0, missing

def check_ollama():
    """Check if Ollama is accessible"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            data = response.json()
            models = [m['name'] for m in data.get('models', [])]
            print(f"[OK] Ollama is running")
            print(f"  Available models: {', '.join(models) if models else 'none'}")
            
            default_model = "llama3.2"
            # Check if model exists (with or without :latest tag)
            model_found = any(
                m == default_model or 
                m == f"{default_model}:latest" or 
                m.startswith(f"{default_model}:")
                for m in models
            )
            
            if model_found:
                print(f"[OK] Default model '{default_model}' is available")
            else:
                print(f"[WARN] Default model '{default_model}' not found")
                print(f"  Run: ollama pull {default_model}")
            return True
        else:
            print(f"[WARN] Ollama responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("[WARN] Ollama is not running or not accessible")
        print("  Start it with: ollama serve")
        print("  Or install from: https://ollama.ai")
        return False
    except Exception as e:
        print(f"⚠ Error checking Ollama: {e}")
        return False

def check_directories():
    """Check if required directories exist"""
    dirs = ['static', 'uploads']
    all_exist = True
    
    for dir_name in dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"[OK] Directory '{dir_name}' exists")
        else:
            print(f"[INFO] Directory '{dir_name}' does not exist (will be created on first run)")
            dir_path.mkdir(exist_ok=True)
            all_exist = False
    
    return True

def main():
    print("=" * 60)
    print("Patient Discharge Instructions Assistant - Setup Check")
    print("=" * 60)
    print()
    
    all_ok = True
    
    # Check Python version
    print("Checking Python version...")
    if not check_python_version():
        all_ok = False
    print()
    
    # Check dependencies
    print("Checking dependencies...")
    deps_ok, missing = check_dependencies()
    if not deps_ok:
        all_ok = False
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("  Install with: pip install -r requirements.txt")
    print()
    
    # Check directories
    print("Checking directories...")
    check_directories()
    print()
    
    # Check Ollama (optional but recommended)
    print("Checking Ollama (optional)...")
    ollama_ok = check_ollama()
    if not ollama_ok:
        print("  Note: The system will use fallback responses without Ollama")
    print()
    
    # Summary
    print("=" * 60)
    if all_ok and ollama_ok:
        print("[SUCCESS] All checks passed! You're ready to start.")
        print("\nStart the server with: python start.py")
    elif all_ok:
        print("[SUCCESS] Core setup is complete!")
        print("[WARN] Ollama is recommended for best results")
        print("\nStart the server with: python start.py")
    else:
        print("[FAIL] Some issues found. Please fix them before starting.")
    print("=" * 60)

if __name__ == "__main__":
    main()


