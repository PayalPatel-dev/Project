#!/usr/bin/env python3
"""
Setup script to create virtual environment and install dependencies
Usage: python setup.py
"""
import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, shell=False):
    """Run a command and return success status"""
    try:
        if shell:
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        else:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def setup_environment():
    """Setup virtual environment and install dependencies"""
    print("=" * 80)
    print("Setting up MIMIC-IV Analysis Environment")
    print("=" * 80)
    
    venv_name = "venv"
    venv_path = Path(venv_name)
    
    # Step 1: Create virtual environment
    print(f"\n1️⃣  Creating virtual environment '{venv_name}'...")
    if venv_path.exists():
        print(f"   ⚠️  Virtual environment already exists")
    else:
        success, output = run_command([sys.executable, "-m", "venv", venv_name])
        if success:
            print(f"   ✓ Virtual environment created")
        else:
            print(f"   ❌ Failed to create virtual environment")
            print(output)
            return False
    
    # Determine pip path based on OS
    if os.name == 'nt':  # Windows
        pip_path = venv_path / "Scripts" / "pip.exe"
        python_path = venv_path / "Scripts" / "python.exe"
    else:  # Unix/Linux/Mac
        pip_path = venv_path / "bin" / "pip"
        python_path = venv_path / "bin" / "python"
    
    # Step 2: Upgrade pip
    print(f"\n2️⃣  Upgrading pip...")
    success, output = run_command([str(python_path), "-m", "pip", "install", "--upgrade", "pip"])
    if success:
        print(f"   ✓ Pip upgraded")
    else:
        print(f"   ⚠️  Warning: Could not upgrade pip")
    
    # Step 3: Install dependencies
    print(f"\n3️⃣  Installing dependencies...")
    dependencies = [
        "kagglehub",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn"
    ]
    
    for dep in dependencies:
        print(f"   Installing {dep}...", end=" ")
        success, output = run_command([str(pip_path), "install", dep, "-q"])
        if success:
            print("✓")
        else:
            print("❌")
            print(f"   Error: {output}")
    
    # Step 4: Create activation instructions
    print("\n" + "=" * 80)
    print("✓ Setup Complete!")
    print("=" * 80)
    
    print("\n📋 Next steps:")
    print("\n1. Activate the virtual environment:")
    if os.name == 'nt':  # Windows
        print(f"   .\\{venv_name}\\Scripts\\activate")
    else:  # Unix/Linux/Mac
        print(f"   source {venv_name}/bin/activate")
    
    print("\n2. Run the explorer script:")
    print("   python mimic_explorer.py")
    
    print("\n3. To deactivate when done:")
    print("   deactivate")
    
    return True

if __name__ == "__main__":
    try:
        success = setup_environment()
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)