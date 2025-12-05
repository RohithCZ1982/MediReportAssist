"""
Create a custom Ollama model for medical discharge instructions
This script creates a Modelfile and sets up a custom model optimized for medical assistance
"""

from pathlib import Path
import subprocess
import sys
import os

def create_modelfile():
    """Create Modelfile for custom medical assistant model"""
    
    modelfile_content = '''FROM llama3.2

SYSTEM """You are a specialized medical assistant focused on helping patients understand their discharge instructions. Your role is to:

1. Interpret medical discharge summaries accurately
2. Translate complex medical terminology into simple, patient-friendly language
3. Provide clear, actionable instructions about medications, diet, activities, and follow-up care
4. Identify and highlight important warnings, contraindications, and emergency signs
5. Answer questions with empathy and reassurance while maintaining medical accuracy

Guidelines:
- Always base answers on the provided discharge summary context
- Use simple language that patients without medical training can understand
- Include specific details: medication names, dosages, timing, and frequencies
- Format information clearly with bullet points or numbered lists when appropriate
- If information is missing from the context, clearly state this
- Never provide medical advice beyond what's in the discharge summary
- Encourage patients to contact their healthcare provider for clarification when needed

Response Style:
- Professional yet warm and reassuring
- Clear and concise
- Well-structured with proper formatting
- Focused on actionable patient instructions

When answering questions about:
- Medications: Include name, dosage, frequency, timing, duration, and any special instructions
- Diet: List specific foods to eat/avoid, meal timing, and duration of restrictions
- Activities: Specify what's allowed/prohibited, intensity levels, and duration
- Follow-up care: Provide appointment details, dates, times, and contact information
- Symptoms/Warnings: Clearly state what to watch for, when to seek help, and emergency signs
- Recovery timeline: Provide expected recovery duration and milestones
"""
'''
    
    modelfile_path = Path("Modelfile")
    modelfile_path.write_text(modelfile_content)
    print(f"✅ Created Modelfile at {modelfile_path.absolute()}")
    return modelfile_path

def create_custom_model(model_name="mediassist"):
    """Create custom Ollama model from Modelfile"""
    
    modelfile_path = Path("Modelfile")
    if not modelfile_path.exists():
        print("❌ Modelfile not found. Creating it now...")
        create_modelfile()
    
    print(f"\n📦 Creating custom Ollama model '{model_name}'...")
    print("This may take a few minutes as it downloads the base model if not already present.\n")
    
    try:
        # Check if Ollama is available
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            raise FileNotFoundError("Ollama not found")
        
        # Create the model
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        
        if result.returncode == 0:
            print(f"✅ Successfully created custom model '{model_name}'!")
            print(f"\n📝 To use this model, set the environment variable:")
            print(f"   Windows (PowerShell): $env:LLM_MODEL=\"{model_name}\"")
            print(f"   Windows (CMD): set LLM_MODEL={model_name}")
            print(f"   Linux/Mac: export LLM_MODEL={model_name}")
            print(f"\n🧪 Test the model with: ollama run {model_name}")
            return True
        else:
            print(f"❌ Error creating model: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("❌ Ollama not found. Please install Ollama first:")
        print("   1. Visit https://ollama.ai")
        print("   2. Download and install Ollama")
        print("   3. Run this script again")
        return False
    except subprocess.TimeoutExpired:
        print("❌ Timeout: Model creation took too long")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_model(model_name="mediassist"):
    """Test the custom model"""
    
    print(f"\n🧪 Testing model '{model_name}'...")
    
    try:
        result = subprocess.run(
            ["ollama", "run", model_name, "What should patients know about discharge instructions?"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ Model is working!")
            print(f"\nSample response:\n{result.stdout[:200]}...")
            return True
        else:
            print(f"❌ Model test failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

def update_env_file(model_name="mediassist"):
    """Create/update .env file with model name"""
    
    env_file = Path(".env")
    env_content = ""
    
    if env_file.exists():
        env_content = env_file.read_text()
        # Remove existing LLM_MODEL line if present
        lines = [line for line in env_content.split('\n') 
                if not line.startswith('LLM_MODEL=')]
        env_content = '\n'.join(lines)
    
    # Add new LLM_MODEL
    if env_content and not env_content.endswith('\n'):
        env_content += '\n'
    env_content += f'LLM_MODEL={model_name}\n'
    
    env_file.write_text(env_content)
    print(f"✅ Updated .env file with LLM_MODEL={model_name}")

def main():
    """Main function"""
    
    print("=" * 60)
    print("Medical Assistant Custom Model Creator")
    print("=" * 60)
    print("\nThis script will:")
    print("1. Create a Modelfile with medical-focused system prompt")
    print("2. Create a custom Ollama model named 'mediassist'")
    print("3. Test the model")
    print("4. Update .env file with the model name")
    print("\n" + "=" * 60 + "\n")
    
    # Ask for model name
    model_name = input("Enter model name (default: mediassist): ").strip() or "mediassist"
    
    # Step 1: Create Modelfile
    print("\n[Step 1/4] Creating Modelfile...")
    create_modelfile()
    
    # Step 2: Create model
    print("\n[Step 2/4] Creating custom Ollama model...")
    if not create_custom_model(model_name):
        print("\n❌ Failed to create model. Please check the errors above.")
        sys.exit(1)
    
    # Step 3: Test model
    print("\n[Step 3/4] Testing model...")
    test_model(model_name)
    
    # Step 4: Update .env
    print("\n[Step 4/4] Updating .env file...")
    update_env_file(model_name)
    
    print("\n" + "=" * 60)
    print("✅ Setup Complete!")
    print("=" * 60)
    print(f"\nYour custom model '{model_name}' is ready to use.")
    print("\nTo use it in your application:")
    print(f"1. Restart your application: python app.py")
    print(f"2. The .env file has been updated with LLM_MODEL={model_name}")
    print(f"3. Or set it manually: $env:LLM_MODEL=\"{model_name}\" (PowerShell)")
    print("\nTo test manually:")
    print(f"   ollama run {model_name}")

if __name__ == "__main__":
    main()

