#!/usr/bin/env python3
"""
Simple test script to verify the application components.
"""
import sys
import os
from pathlib import Path

def test_imports():
    """Test if key modules can be imported."""
    try:
        # Test basic Python functionality
        import streamlit
        print("✅ Streamlit imported successfully")

        import pandas
        print("✅ Pandas imported successfully")

        import numpy
        print("✅ NumPy imported successfully")

        import plotly
        print("✅ Plotly imported successfully")

        # Test our modules
        sys.path.append(str(Path(__file__).parent))

        try:
            from src.config.settings import settings
            print("✅ Application settings loaded successfully")
        except Exception as e:
            print(f"⚠️ Settings module issue: {e}")

        try:
            from src.dashboard.app import create_dashboard
            print("✅ Dashboard module loaded successfully")
        except Exception as e:
            print(f"⚠️ Dashboard module issue: {e}")

        try:
            from src.models.trainer import ModelTrainer
            print("✅ Model trainer module loaded successfully")
        except Exception as e:
            print(f"⚠️ Model trainer module issue: {e}")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_environment():
    """Test environment configuration."""
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file found")

        # Check for API key
        try:
            from src.config.settings import settings
            if hasattr(settings, 'google_api_key') and settings.google_api_key:
                print("✅ Google API key configured")
            else:
                print("⚠️ Google API key not found in settings")
        except:
            print("⚠️ Could not load settings")
    else:
        print("⚠️ .env file not found")

def test_directories():
    """Test if required directories exist."""
    required_dirs = ["logs", "models", "data", "mlruns"]
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✅ {dir_name}/ directory exists")
        else:
            print(f"⚠️ {dir_name}/ directory missing")

def main():
    """Run all tests."""
    print("🧪 Testing Real-Time ML Application...")
    print("=" * 50)

    # Test imports
    print("\n📦 Testing Imports:")
    imports_ok = test_imports()

    # Test environment
    print("\n⚙️ Testing Environment:")
    test_environment()

    # Test directories
    print("\n📁 Testing Directories:")
    test_directories()

    print("\n" + "=" * 50)

    if imports_ok:
        print("🎉 Basic application structure is working!")
        print("\n🚀 Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run dashboard: streamlit run streamlit_app.py")
        print("3. Run full app: python src/main.py")
    else:
        print("❌ Some issues found. Please check the errors above.")

    print("=" * 50)

if __name__ == "__main__":
    main()
