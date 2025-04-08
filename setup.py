#!/usr/bin/env python3
import subprocess
import os
import sys

def check_dependencies():
    """Check if required system dependencies are installed"""
    dependencies = {
        "ollama": "Ollama is not installed. Please install from https://ollama.ai/",
    }
    
    for cmd, msg in dependencies.items():
        try:
            subprocess.run([cmd, "--version"], capture_output=True)
        except FileNotFoundError:
            print(f"Error: {msg}")
            sys.exit(1)

def setup_virtual_env():
    """Create and activate virtual environment"""
    if not os.path.exists("venv"):
        subprocess.run([sys.executable, "-m", "venv", "venv"])
    
    # Activate virtual environment
    if sys.platform == "win32":
        activate_script = "venv\\Scripts\\activate"
    else:
        activate_script = "source venv/bin/activate"
    
    print(f"To activate virtual environment, run: {activate_script}")

def install_requirements():
    """Install Python dependencies"""
    pip_cmd = "venv\\Scripts\\pip" if sys.platform == "win32" else "venv/bin/pip"
    subprocess.run([pip_cmd, "install", "-r", "requirements.txt"])

def setup_project_structure():
    """Create necessary directories and files"""
    directories = ["chroma_db", "models", "logs"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def main():
    print("Setting up RAG project...")
    
    # Check system dependencies
    print("Checking dependencies...")
    check_dependencies()
    
    # Setup virtual environment
    print("Setting up virtual environment...")
    setup_virtual_env()
    
    # Install requirements
    print("Installing Python dependencies...")
    install_requirements()
    
    # Setup project structure
    print("Creating project directories...")
    setup_project_structure()
    
    print("\nSetup complete! Follow these steps to run the project:")
    print("\n1. Start Ollama server:")
    print("   ollama serve")
    print("\n2. Create and pull the model:")
    print("   ollama create llama3.2 -f llama3.2.modelfile")
    print("\n3. Activate the virtual environment:")
    if sys.platform == "win32":
        print("   .\\venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("\n4. Run the Streamlit app:")
    print("   streamlit run app.py")

if __name__ == "__main__":
    main()