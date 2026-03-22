"""
Build script to package the Trading Bot into a single executable (.exe).
Uses PyInstaller to bundle dependencies like PyTorch and SQLAlchemy.
"""
import subprocess
import os
import sys

def build_exe():
    print("Starting build process for TradingBot.exe...")
    
    # Path to the main entry point
    main_script = "main.py"
    
    if not os.path.exists(main_script):
        print(f"Error: {main_script} not found.")
        return

    # Define the PyInstaller command
    # --onefile: Bundle into a single executable
    # --name: Name of the output file
    # --hidden-import: Ensure dynamic dependencies are caught
    # --collect-all: Bundles large packages correctly
    cmd = [
        "pyinstaller",
        "--onefile",
        "--name", "TradingBot",
        "--hidden-import", "sqlalchemy.ext.declarative",
        "--hidden-import", "sqlalchemy.orm",
        "--hidden-import", "torch",
        "--collect-all", "torch",
        "--collect-all", "matplotlib",
        main_script
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run PyInstaller
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\nBuild Successful!")
            print("The executable can be found in the 'dist' folder as 'TradingBot.exe'.")
            print("Note: You will still need to place your '.env' file in the same folder as the .exe.")
        else:
            print(f"\nBuild failed with exit code: {result.returncode}")
            
    except Exception as e:
        print(f"An error occurred during the build: {e}")

if __name__ == "__main__":
    build_exe()
