#!/usr/bin/env python3
"""
Startup script for the Real-Time ML Data Analysis Application.
This script handles environment setup and launches the application.
"""
import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_python_version():
    """Check if Python version is compatible."""
    required_version = (3, 8)
    current_version = sys.version_info[:2]

    if current_version < required_version:
        logger.error(f"❌ Python {required_version[0]}.{required_version[1]}+ required. Found Python {current_version[0]}.{current_version[1]}")
        sys.exit(1)

    logger.info(f"✅ Python version: {current_version[0]}.{current_version[1]}.{current_version[2]}")


def setup_virtual_environment():
    """Set up virtual environment if it doesn't exist."""
    venv_path = Path("venv")

    if not venv_path.exists():
        logger.info("📦 Creating virtual environment...")
        try:
            subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
            logger.info("✅ Virtual environment created")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to create virtual environment: {e}")
            sys.exit(1)
    else:
        logger.info("✅ Virtual environment already exists")


def install_dependencies():
    """Install Python dependencies."""
    requirements_path = Path("requirements.txt")

    if not requirements_path.exists():
        logger.error("❌ requirements.txt not found")
        sys.exit(1)

    logger.info("📦 Installing dependencies...")

    # Activate virtual environment and install dependencies
    if os.name == 'nt':  # Windows
        activate_script = "venv/Scripts/activate.bat"
        pip_command = "venv/Scripts/pip"
    else:  # Unix/Linux
        activate_script = "venv/bin/activate"
        pip_command = "venv/bin/pip"

    try:
        # Upgrade pip first
        subprocess.run([pip_command, "install", "--upgrade", "pip"], check=True)

        # Install requirements
        subprocess.run([pip_command, "install", "-r", "requirements.txt"], check=True)

        logger.info("✅ Dependencies installed successfully")

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install dependencies: {e}")
        sys.exit(1)


def check_environment_variables():
    """Check and validate environment variables."""
    env_file = Path(".env")

    if not env_file.exists():
        logger.warning("⚠️ .env file not found. Creating from template...")
        env_example = Path(".env.example")

        if env_example.exists():
            import shutil
            shutil.copy(env_example, env_file)
            logger.info("✅ Created .env file from template")
            logger.info("🔧 Please edit .env file with your API keys and configuration")
        else:
            logger.error("❌ .env.example not found")
            sys.exit(1)

    # Check for required environment variables
    required_vars = ["GOOGLE_API_KEY"]
    missing_vars = []

    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        logger.warning(f"⚠️ Missing environment variables: {', '.join(missing_vars)}")
        logger.info("🔧 Please set these variables in your .env file")

    return len(missing_vars) == 0


def create_necessary_directories():
    """Create necessary directories for the application."""
    directories = [
        "logs",
        "models",
        "data",
        "mlruns",
        "tests"
    ]

    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"📁 Created/verified directory: {directory}")


def main():
    """Main startup function."""
    logger.info("🚀 Starting Real-Time ML Data Analysis Application setup...")

    try:
        # Step 1: Check Python version
        check_python_version()

        # Step 2: Set up virtual environment
        setup_virtual_environment()

        # Step 3: Install dependencies
        install_dependencies()

        # Step 4: Check environment variables
        env_ok = check_environment_variables()

        # Step 5: Create necessary directories
        create_necessary_directories()

        # Step 6: Show next steps
        logger.info("✅ Setup completed successfully!")

        print("\n" + "="*60)
        print("🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n📋 Next Steps:")
        print("1. 🔧 Edit .env file with your API keys")
        print("2. 🚀 Run: python src/main.py")
        print("3. 📊 Access dashboard at: http://localhost:8501")
        print("4. 🔗 API available at: http://localhost:8000")
        print("\n🔧 Available Commands:")
        print("• python src/main.py          # Start full application")
        print("• streamlit run streamlit_app.py  # Start dashboard only")
        print("• python -m pytest tests/     # Run tests")
        print("\n📚 For more information, see README.md and processes.md")
        print("="*60)

        if not env_ok:
            print("\n⚠️  Please configure your .env file before running the application!")
            print("   Required: GOOGLE_API_KEY")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("👋 Setup interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
