"""
Real-time data ingestion service for the ML application.
"""
import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
from pydantic import BaseModel
import websockets
import redis.asyncio as redis
from kafka import KafkaProducer, KafkaConsumer

from src.config.settings import settings


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataPoint(BaseModel):
    """Data point model for real-time ingestion."""
    timestamp: datetime
    source: str
    data_type: str
    features: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class DataIngestionService:
    """Service for real-time data ingestion and processing."""

    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.kafka_producer: Optional[KafkaProducer] = None
        self.websocket_clients: List[websockets.WebSocketServerProtocol] = []
        self.is_running = False

    async def initialize(self):
        """Initialize the data ingestion service."""
        try:
            # Initialize Redis client
            self.redis_client = redis.from_url(
                settings.redis_url,
                password=settings.redis_password,
                decode_responses=True
            )

            # Test Redis connection
            await self.redis_client.ping()
            logger.info("✅ Redis connection established")

            # Initialize Kafka producer
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=settings.kafka_bootstrap_servers,
                client_id=settings.kafka_client_id,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            logger.info("✅ Kafka producer initialized")

            # Create data streams
            await self._create_data_streams()

            logger.info("🚀 Data ingestion service initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize data ingestion service: {e}")
            raise

    async def _create_data_streams(self):
        """Create necessary data streams in Redis."""
        try:
            # Create streams for different data types
            streams = [
                "raw_data_stream",
                "processed_data_stream",
                "feature_stream",
                "model_input_stream"
            ]

            for stream in streams:
                await self.redis_client.xgroup_create(
                    stream, "consumers", mkstream=True
                )

            logger.info("✅ Data streams created successfully")

        except Exception as e:
            logger.warning(f"⚠️ Error creating data streams: {e}")

    async def ingest_data(self, data_point: DataPoint) -> bool:
        """
        Ingest a single data point into the system.

        Args:
            data_point: The data point to ingest

        Returns:
            bool: True if ingestion was successful
        """
        try:
            # Convert to dictionary
            data_dict = data_point.dict()

            # Add to Redis stream
            await self.redis_client.xadd(
                "raw_data_stream",
                data_dict
            )

            # Send to Kafka for further processing
            await self._send_to_kafka("raw_data", data_dict)

            # Process data asynchronously
            asyncio.create_task(self._process_data(data_dict))

            logger.debug(f"📥 Data ingested: {data_point.source}")
            return True

        except Exception as e:
            logger.error(f"❌ Error ingesting data: {e}")
            return False

    async def _send_to_kafka(self, topic: str, data: Dict[str, Any]):
        """Send data to Kafka topic."""
        try:
            self.kafka_producer.send(topic, data)
            self.kafka_producer.flush()
        except Exception as e:
            logger.error(f"❌ Error sending to Kafka: {e}")

    async def _process_data(self, data: Dict[str, Any]):
        """Process raw data and extract features."""
        try:
            # Validate data
            validated_data = await self._validate_data(data)

            # Extract features
            features = await self._extract_features(validated_data)

            # Store processed data
            processed_point = {
                "timestamp": datetime.now(),
                "original_id": data.get("timestamp"),
                "features": features,
                "processed_at": datetime.now().isoformat()
            }

            # Add to processed data stream
            await self.redis_client.xadd(
                "processed_data_stream",
                processed_point
            )

            # Send to model input stream
            await self.redis_client.xadd(
                "model_input_stream",
                processed_point
            )

            logger.debug("🔄 Data processed successfully")

        except Exception as e:
            logger.error(f"❌ Error processing data: {e}")

    async def _validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate incoming data."""
        # Basic validation logic
        required_fields = ["timestamp", "source", "data_type", "features"]

        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        return data

    async def _extract_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from raw data."""
        features = data.get("features", {})

        # Add derived features
        if "numerical_features" in features:
            numerical = features["numerical_features"]

            # Statistical features
            if isinstance(numerical, list):
                features["stats"] = {
                    "mean": sum(numerical) / len(numerical) if numerical else 0,
                    "min": min(numerical) if numerical else 0,
                    "max": max(numerical) if numerical else 0,
                    "count": len(numerical)
                }

        return features

    async def ingest_from_file(self, file_path: str, data_format: str = "auto") -> int:
        """
        Ingest data from a file (CSV, JSON, Excel).

        Args:
            file_path: Path to the file
            data_format: Format of the file ("csv", "json", "excel", "auto")

        Returns:
            int: Number of records ingested
        """
        try:
            count = 0

            if data_format == "auto":
                if file_path.endswith('.csv'):
                    data_format = "csv"
                elif file_path.endswith('.json'):
                    data_format = "json"
                elif file_path.endswith(('.xlsx', '.xls')):
                    data_format = "excel"

            if data_format == "csv":
                df = pd.read_csv(file_path)
                count = await self._ingest_dataframe(df)

            elif data_format == "json":
                with open(file_path, 'r') as f:
                    data = json.load(f)
                count = await self._ingest_json_data(data)

            elif data_format == "excel":
                df = pd.read_excel(file_path)
                count = await self._ingest_dataframe(df)

            logger.info(f"📁 Ingested {count} records from {file_path}")
            return count

        except Exception as e:
            logger.error(f"❌ Error ingesting from file {file_path}: {e}")
            return 0

    async def _ingest_dataframe(self, df: pd.DataFrame) -> int:
        """Ingest data from pandas DataFrame."""
        count = 0

        for _, row in df.iterrows():
            try:
                # Convert row to data point
                data_point = DataPoint(
                    timestamp=datetime.now(),
                    source="file_ingestion",
                    data_type="tabular",
                    features=row.to_dict()
                )

                success = await self.ingest_data(data_point)
                if success:
                    count += 1

            except Exception as e:
                logger.warning(f"⚠️ Error ingesting row {count}: {e}")

        return count

    async def _ingest_json_data(self, data: Any) -> int:
        """Ingest data from JSON structure."""
        count = 0

        try:
            if isinstance(data, list):
                for item in data:
                    data_point = DataPoint(
                        timestamp=datetime.now(),
                        source="json_ingestion",
                        data_type="json",
                        features=item
                    )

                    success = await self.ingest_data(data_point)
                    if success:
                        count += 1

            elif isinstance(data, dict):
                data_point = DataPoint(
                    timestamp=datetime.now(),
                    source="json_ingestion",
                    data_type="json",
                    features=data
                )

                if await self.ingest_data(data_point):
                    count = 1

        except Exception as e:
            logger.error(f"❌ Error ingesting JSON data: {e}")

        return count

    async def get_stream_data(self, stream_name: str, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent data from a stream."""
        try:
            # Read from Redis stream
            messages = await self.redis_client.xrevrange(
                stream_name,
                count=count
            )

            return [message[1] for message in messages]

        except Exception as e:
            logger.error(f"❌ Error getting stream data: {e}")
            return []

    def start_websocket_server(self, port: int = 8765):
        """Start WebSocket server for real-time data streaming."""
        async def websocket_handler(websocket, path):
            """Handle WebSocket connections."""
            self.websocket_clients.append(websocket)
            logger.info(f"🔗 WebSocket client connected: {len(self.websocket_clients)} total")

            try:
                while True:
                    # Send recent data to client
                    recent_data = await self.get_stream_data("processed_data_stream", 5)

                    if recent_data:
                        await websocket.send(json.dumps(recent_data))

                    await asyncio.sleep(1)  # Send updates every second

            except websockets.exceptions.ConnectionClosed:
                logger.info("🔌 WebSocket client disconnected")
                if websocket in self.websocket_clients:
                    self.websocket_clients.remove(websocket)

        # Start WebSocket server
        start_server = websockets.serve(websocket_handler, "localhost", port)
        logger.info(f"📡 WebSocket server started on port {port}")

        return start_server

    async def shutdown(self):
        """Shutdown the data ingestion service."""
        try:
            self.is_running = False

            if self.kafka_producer:
                self.kafka_producer.close()

            if self.redis_client:
                await self.redis_client.close()

            # Close WebSocket connections
            for client in self.websocket_clients:
                await client.close()

            logger.info("🛑 Data ingestion service shutdown complete")

        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")
