"""
Setup script for the Real-Time ML Application.
This script verifies all components and prepares for deployment.
"""
import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check Python version."""
    print("🐍 Checking Python version...")
    required = (3, 8)
    current = sys.version_info[:2]

    if current >= required:
        print(f"✅ Python {current[0]}.{current[1]}.{current[2]}")
        return True
    else:
        print(f"❌ Python {required[0]}.{required[1]}+ required")
        return False

def check_files():
    """Check if all required files exist."""
    print("\n📁 Checking required files...")

    required_files = [
        "requirements.txt",
        "streamlit_app.py",
        ".env",
        "README.md",
        "src/main.py",
        "src/config/settings.py",
        "src/models/trainer.py",
        "src/dashboard/app.py",
        "src/data/ingestion.py"
    ]

    missing = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing.append(file_path)

    return len(missing) == 0, missing

def check_git_repository():
    """Check Git repository setup."""
    print("\n🔗 Checking Git repository...")

    try:
        # Check if we're in a git repository
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, cwd="."
        )

        if result.returncode == 0:
            print("✅ Git repository detected")

            # Check remote origin
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                capture_output=True, text=True, cwd="."
            )

            if result.returncode == 0:
                remote_url = result.stdout.strip()
                print(f"✅ Remote origin: {remote_url}")

                # Check if it matches expected repository
                expected_url = "https://github.com/Timi2001/Analytics-Hub"
                if expected_url in remote_url:
                    print("✅ Repository matches expected configuration")
                    return True
                else:
                    print(f"⚠️ Repository URL doesn't match expected: {expected_url}")
                    return False
            else:
                print("⚠️ No remote origin configured")
                return False
        else:
            print("⚠️ Not in a git repository")
            return False

    except Exception as e:
        print(f"⚠️ Error checking git: {e}")
        return False

def check_environment_variables():
    """Check environment configuration."""
    print("\n⚙️ Checking environment variables...")

    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found")
        return False

    print("✅ .env file exists")

    # Check for required variables
    required_vars = ["GOOGLE_API_KEY"]
    missing_vars = []

    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Show first few characters for verification
            masked = value[:8] + "..." if len(value) > 8 else value
            print(f"✅ {var}: {masked}")
        else:
            print(f"❌ {var}: Not set")
            missing_vars.append(var)

    if missing_vars:
        print(f"\n❌ Missing environment variables: {', '.join(missing_vars)}")
        return False

    return True

def create_deployment_package():
    """Create deployment package for Streamlit Cloud."""
    print("\n📦 Creating deployment package...")

    try:
        # Create a deployment requirements file
        deployment_reqs = """streamlit
pandas
numpy
plotly
scikit-learn
google-generativeai
python-dotenv
pydantic
"""

        with open("requirements.deploy.txt", "w") as f:
            f.write(deployment_reqs)

        print("✅ Created requirements.deploy.txt")

        # Create Streamlit configuration
        streamlit_config = """[global]

# If false, makes your Streamlit script run as a module.
developmentMode = false

# Allows you to type a variable or string by itself in a single line of Python code to write it to the app.
dataFrameSerialization = "legacy"

[server]

headless = true
port = 8501
address = "0.0.0.0"

"""

        with open(".streamlit/config.toml", "w") as f:
            f.write(streamlit_config)

        print("✅ Created Streamlit configuration")
        return True

    except Exception as e:
        print(f"❌ Error creating deployment package: {e}")
        return False

def main():
    """Main setup verification function."""
    print("🔍 Verifying Real-Time ML Application Setup")
    print("=" * 60)

    all_checks_passed = True

    # Check Python version
    if not check_python_version():
        all_checks_passed = False

    # Check required files
    files_ok, missing = check_files()
    if not files_ok:
        print(f"\n❌ Missing files: {', '.join(missing)}")
        all_checks_passed = False

    # Check Git repository
    if not check_git_repository():
        all_checks_passed = False

    # Check environment variables
    if not check_environment_variables():
        all_checks_passed = False

    # Create deployment package
    if not create_deployment_package():
        all_checks_passed = False

    print("\n" + "=" * 60)

    if all_checks_passed:
        print("🎉 ALL CHECKS PASSED!")
        print("✅ Application is ready for deployment")

        print("\n📋 Deployment Checklist:")
        print("1. ✅ All required files present")
        print("2. ✅ GitHub repository configured")
        print("3. ✅ Environment variables set")
        print("4. ✅ Deployment package created")
        print("5. ✅ Streamlit configuration ready")

        print("\n🚀 Ready for Streamlit Community Cloud deployment!")

        print("\n🔧 Next Steps:")
        print("1. Commit and push code to GitHub:")
        print("   git add .")
        print("   git commit -m 'Complete real-time ML application'")
        print("   git push origin main")
        print("")
        print("2. Deploy to Streamlit Cloud:")
        print("   - Go to https://share.streamlit.io")
        print("   - Connect your GitHub repository")
        print("   - Set main file: streamlit_app.py")
        print("   - Add environment variables")
        print("   - Deploy!")

    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print("\n🔧 Common fixes:")
        print("• Install Python 3.8+")
        print("• Set up GitHub repository")
        print("• Configure environment variables in .env")
        print("• Ensure all files are present")

    print("=" * 60)
    return all_checks_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
