"""
Launch App - Works from models directory
"""

import os
import sys

# Add parent directory to path and change to it
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
os.chdir(parent_dir)

def main():
    print("🚀 Launching Enhanced Bangla Fake News Detection App...")
    print(f"📁 Working from: {os.getcwd()}")
    
    try:
        # Import and run app
        import app
        print("✅ App loaded successfully!")
        
        # The app will launch automatically when imported
        
    except Exception as e:
        print(f"❌ Failed to launch app: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()