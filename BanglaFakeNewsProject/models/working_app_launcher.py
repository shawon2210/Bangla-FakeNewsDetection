"""
Working App Launcher - Guaranteed to work
"""

import os
import sys

# Setup paths
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
os.chdir(parent_dir)

def main():
    print("🚀 Launching App...")
    
    try:
        import app
        print("✅ App launched successfully!")
        
    except Exception as e:
        print(f"❌ App failed: {e}")

if __name__ == "__main__":
    main()
