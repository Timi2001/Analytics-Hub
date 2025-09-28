"""
Streamlit App for deployment to Streamlit Community Cloud.
This is the main entry point for the deployed application.
"""
import streamlit as st
from src.dashboard.app import create_dashboard
import os
import sys

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure page
st.set_page_config(
    page_title="Analytics Hub - Real-Time ML Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-online {
        background-color: #28a745;
    }
    .status-offline {
        background-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main function for the Streamlit app."""
    try:
        # Create and run dashboard
        dashboard = create_dashboard()

        # Render header
        st.markdown('<h1 class="main-header">🚀 Analytics Hub</h1>', unsafe_allow_html=True)
        st.markdown("### Real-Time Machine Learning Data Analysis Platform")

        # Status indicators
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("""
                <div class="metric-card">
                    <h3>📊 System Status</h3>
                    <span class="status-indicator status-online"></span>Online
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
                <div class="metric-card">
                    <h3>🤖 AI Powered</h3>
                    <h2>Gemini</h2>
                </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
                <div class="metric-card">
                    <h3>🔄 Real-Time</h3>
                    <h4>Streaming</h4>
                </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown("""
                <div class="metric-card">
                    <h3>📈 Live Data</h3>
                    <h2>Analytics</h2>
                </div>
            """, unsafe_allow_html=True)

        # Run the dashboard
        dashboard.run()

    except Exception as e:
        st.error(f"❌ Application Error: {e}")
        st.error("Please check the application configuration and try again.")

        # Show troubleshooting information
        with st.expander("🔧 Troubleshooting"):
            st.write("**Common Issues:**")
            st.write("• Check if all required environment variables are set")
            st.write("• Verify API keys are valid")
            st.write("• Ensure all dependencies are installed")
            st.write("• Check application logs for detailed error information")

if __name__ == "__main__":
    main()
