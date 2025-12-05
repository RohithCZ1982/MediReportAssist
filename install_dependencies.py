#!/usr/bin/env python3
"""
Step-by-step dependency installer
Helps install packages individually to identify and handle compatibility issues
"""

import subprocess
import sys
import platform

def run_command(cmd, description):
    """Run a pip install command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Installing: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Successfully installed: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install: {description}")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("="*60)
    print("Patient Discharge Instructions Assistant - Dependency Installer")
    print("="*60)
    print(f"\nPython version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    
    # Check Python version
    if sys.version_info >= (3, 13):
        print("\n⚠️  WARNING: You're using Python 3.13")
        print("   Some packages may not have pre-built wheels yet.")
        print("   Consider using Python 3.11 or 3.12 for better compatibility.")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Installation cancelled.")
            return
    
    # Upgrade pip first
    print("\n" + "="*60)
    print("Step 1: Upgrading pip, setuptools, and wheel")
    print("="*60)
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], check=False)
    
    # Core packages (usually work on all Python versions)
    core_packages = [
        (["fastapi"], "FastAPI web framework"),
        (["uvicorn[standard]"], "Uvicorn ASGI server"),
        (["python-multipart"], "Multipart form data support"),
        (["aiofiles"], "Async file operations"),
        (["pydantic"], "Data validation"),
        (["requests"], "HTTP requests library"),
    ]
    
    print("\n" + "="*60)
    print("Step 2: Installing core packages")
    print("="*60)
    
    failed_packages = []
    for package, description in core_packages:
        cmd = [sys.executable, "-m", "pip", "install"] + package
        if not run_command(cmd, description):
            failed_packages.append((package, description))
    
    # Document processing packages
    doc_packages = [
        (["pypdf2"], "PDF processing"),
        (["python-docx"], "DOCX processing"),
    ]
    
    print("\n" + "="*60)
    print("Step 3: Installing document processing packages")
    print("="*60)
    
    for package, description in doc_packages:
        cmd = [sys.executable, "-m", "pip", "install"] + package
        if not run_command(cmd, description):
            failed_packages.append((package, description))
    
    # ChromaDB (may need special handling)
    print("\n" + "="*60)
    print("Step 4: Installing ChromaDB (vector database)")
    print("="*60)
    
    chromadb_installed = False
    
    # Try normal installation first
    if run_command([sys.executable, "-m", "pip", "install", "chromadb"], "ChromaDB"):
        chromadb_installed = True
    else:
        # Try with no build isolation
        print("\nTrying ChromaDB with --no-build-isolation flag...")
        if run_command([sys.executable, "-m", "pip", "install", "chromadb", "--no-build-isolation"], "ChromaDB (no build isolation)"):
            chromadb_installed = True
        else:
            failed_packages.append((["chromadb"], "ChromaDB vector database"))
    
    # Sentence transformers (will pull torch)
    print("\n" + "="*60)
    print("Step 5: Installing sentence-transformers (embeddings)")
    print("="*60)
    
    sentence_transformers_installed = False
    
    # Try normal installation
    if run_command([sys.executable, "-m", "pip", "install", "sentence-transformers"], "Sentence Transformers"):
        sentence_transformers_installed = True
    else:
        # Try installing torch separately first
        print("\nTrying to install torch separately first...")
        run_command([sys.executable, "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu"], "PyTorch (CPU)")
        
        # Try installing dependencies separately
        print("\nInstalling sentence-transformers dependencies...")
        run_command([sys.executable, "-m", "pip", "install", "transformers", "tokenizers", "safetensors", "huggingface-hub"], "Transformers dependencies")
        
        # Try sentence-transformers again
        if run_command([sys.executable, "-m", "pip", "install", "sentence-transformers"], "Sentence Transformers (retry)"):
            sentence_transformers_installed = True
        else:
            failed_packages.append((["sentence-transformers"], "Sentence Transformers"))
    
    # Summary
    print("\n" + "="*60)
    print("Installation Summary")
    print("="*60)
    
    if not failed_packages:
        print("\n✓ All packages installed successfully!")
        print("\nNext steps:")
        print("1. Set up Ollama: ollama serve")
        print("2. Pull a model: ollama pull llama3.2")
        print("3. Run: python check_setup.py")
        print("4. Start server: python start.py")
    else:
        print(f"\n⚠️  {len(failed_packages)} package(s) failed to install:")
        for package, description in failed_packages:
            print(f"   - {description} ({package[0]})")
        
        print("\nTroubleshooting options:")
        print("1. Use Python 3.11 or 3.12 (recommended)")
        print("2. See INSTALL.md for detailed troubleshooting")
        print("3. Install packages manually one by one")
        
        if not chromadb_installed or not sentence_transformers_installed:
            print("\nNote: The system can work with minimal dependencies,")
            print("      but RAG features will be limited.")
            print("      Install: pip install fastapi uvicorn python-multipart")
            print("               pip install aiofiles pydantic requests")
            print("               pip install pypdf2 python-docx")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInstallation cancelled by user.")
        sys.exit(1)


