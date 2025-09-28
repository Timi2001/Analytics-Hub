"""
Main entry point for the Real-Time ML Data Analysis Application.
"""
import uvicorn
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from src.config.settings import settings
from src.data.ingestion import DataIngestionService
from src.models.trainer import ModelTrainer
from src.dashboard.app import create_dashboard


def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""

    # Create FastAPI application
    app = FastAPI(
        title="Analytics Hub - Real-Time ML Data Analysis Application",
        description="A sophisticated real-time machine learning platform with live data processing",
        version="1.0.0",
        debug=settings.debug,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Root endpoint with application information."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Real-Time ML Data Analysis Application</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .container { max-width: 800px; margin: 0 auto; }
                .status { background: #f0f8ff; padding: 20px; border-radius: 8px; margin: 20px 0; }
                .success { color: #28a745; }
                .info { color: #17a2b8; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 Real-Time ML Data Analysis Application</h1>
                <div class="status">
                    <h2 class="success">✅ Application Status: Running</h2>
                    <p class="info">Environment: {settings.app_env}</p>
                    <p class="info">Debug Mode: {str(settings.debug)}</p>
                    <p class="info">API Server: http://{settings.host}:{settings.port}</p>
                    <p class="info">Dashboard: http://{settings.streamlit_address}:{settings.streamlit_port}</p>
                </div>
                <div>
                    <h3>🔧 Available Services:</h3>
                    <ul>
                        <li><strong>Data Ingestion:</strong> Real-time data processing pipeline</li>
                        <li><strong>Model Training:</strong> Dynamic machine learning model training</li>
                        <li><strong>Live Dashboard:</strong> Interactive real-time visualizations</li>
                        <li><strong>API Endpoints:</strong> RESTful API for data operations</li>
                    </ul>
                </div>
                <div>
                    <h3>📊 Key Features:</h3>
                    <ul>
                        <li>Real-time data streaming and processing</li>
                        <li>Dynamic model training and updates</li>
                        <li>Interactive dashboard with live updates</li>
                        <li>Support for multiple data sources</li>
                        <li>Automated model versioning and management</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "environment": settings.app_env,
            "version": "1.0.0"
        }

    @app.get("/api/status")
    async def api_status():
        """API status with service information."""
        return {
            "api_status": "running",
            "data_ingestion": "initialized",
            "model_trainer": "ready",
            "dashboard": "available",
            "streaming": "active"
        }

    return app


async def startup_services():
    """Initialize and start background services."""
    try:
        # Initialize data ingestion service
        data_service = DataIngestionService()
        await data_service.initialize()

        # Initialize model trainer
        model_trainer = ModelTrainer()
        await model_trainer.initialize()

        print("✅ All services initialized successfully")

    except Exception as e:
        print(f"❌ Error initializing services: {e}")
        raise


async def main():
    """Main application startup."""
    print("🚀 Starting Real-Time ML Data Analysis Application...")

    # Create FastAPI application
    app = create_application()

    # Initialize services
    await startup_services()

    # Start Streamlit dashboard in background
    import threading
    def run_dashboard():
        try:
            dashboard_app = create_dashboard()
            print("📊 Streamlit dashboard starting...")
            # Dashboard will be accessible at http://localhost:8501
        except Exception as e:
            print(f"❌ Error starting dashboard: {e}")

    # Start dashboard in background thread
    dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
    dashboard_thread.start()

    # Start FastAPI server
    config = uvicorn.Config(
        app,
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )

    server = uvicorn.Server(config)

    print(f"🌐 API Server starting at http://{settings.host}:{settings.port}")
    print(f"📊 Dashboard available at http://{settings.streamlit_address}:{settings.streamlit_port}")
    print("🔧 Configuration loaded from environment variables")
    # Start server
    await server.serve()


if __name__ == "__main__":
    # Run the application
    asyncio.run(main())
