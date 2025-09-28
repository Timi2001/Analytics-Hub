#!/usr/bin/env python3
"""
Test script for data ingestion with the provided survey data.
"""
import asyncio
import pandas as pd
from pathlib import Path
from src.data.ingestion import DataIngestionService
from src.config.settings import settings

async def test_data_ingestion():
    """Test data ingestion with the survey datasets."""
    print("🧪 Testing Data Ingestion Pipeline")
    print("=" * 50)

    # Initialize the data ingestion service
    data_service = DataIngestionService()

    try:
        # Test configuration
        print("✅ Configuration loaded successfully")
        print(f"   - Google API Key: {settings.google_api_key[:20]}...")
        print(f"   - Debug Mode: {settings.debug}")
        print(f"   - Environment: {settings.app_env}")

        # Test data files
        data_files = [
            "c:/Users/owner/Downloads/Survey Response (Students) - Sheet1.csv",
            "c:/Users/owner/Downloads/Survey Response (Alumni_staff) - Sheet1.csv"
        ]

        total_records = 0

        for file_path in data_files:
            if Path(file_path).exists():
                print(f"\n📁 Processing: {Path(file_path).name}")

                # Read and preview the data
                df = pd.read_csv(file_path)
                print(f"   - Shape: {df.shape}")
                print(f"   - Columns: {len(df.columns)}")
                print(f"   - Sample columns: {list(df.columns[:5])}...")

                # Test ingestion
                records_ingested = await data_service.ingest_from_file(file_path, "csv")
                print(f"   - Records ingested: {records_ingested}")

                total_records += records_ingested

                # Show data quality info
                print(f"   - Missing values: {df.isnull().sum().sum()}")
                print(f"   - Data types: {df.dtypes.nunique()} unique types")
            else:
                print(f"❌ File not found: {file_path}")

        print("\n📊 Ingestion Summary:")
        print(f"   - Total records processed: {total_records}")
        print(f"   - Files processed: {len(data_files)}")
        print("   - Average records per file: {:.1f}".format(total_records / len(data_files) if data_files else 0))

        # Test data retrieval
        print("\n🔍 Testing Data Retrieval:")
        recent_data = await data_service.get_stream_data("raw_data_stream", 5)
        print(f"   - Recent records in stream: {len(recent_data)}")

        if recent_data:
            print("   - Sample record keys:", list(recent_data[0].keys())[:5])

        return True

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

    finally:
        # Cleanup
        await data_service.shutdown()

def analyze_datasets():
    """Analyze the structure of the survey datasets."""
    print("\n📈 Dataset Analysis")
    print("=" * 50)

    data_files = [
        "c:/Users/owner/Downloads/Survey Response (Students) - Sheet1.csv",
        "c:/Users/owner/Downloads/Survey Response (Alumni_staff) - Sheet1.csv"
    ]

    for file_path in data_files:
        if Path(file_path).exists():
            print(f"\n📋 {Path(file_path).name}:")

            df = pd.read_csv(file_path)

            # Basic info
            print(f"   - Shape: {df.shape}")
            print(f"   - Total responses: {len(df)}")

            # Column analysis
            print(f"   - Numeric columns: {len(df.select_dtypes(include=['number']).columns)}")
            print(f"   - Categorical columns: {len(df.select_dtypes(include=['object']).columns)}")

            # Sample data
            print("   - Sample data (first 2 rows):")
            for i, (_, row) in enumerate(df.head(2).iterrows()):
                print(f"     Row {i+1}: {dict(row)[:3]}...")  # Show first 3 columns

            # Unique values in categorical columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                print(f"   - Sample categorical column '{categorical_cols[0]}' unique values: {df[categorical_cols[0]].unique()[:5]}")

if __name__ == "__main__":
    print("🎯 Analytics-Hub Data Ingestion Test")
    print("Testing with University of Abuja Survey Data")

    # Analyze datasets first
    analyze_datasets()

    # Test ingestion
    success = asyncio.run(test_data_ingestion())

    if success:
        print("\n🎉 Data ingestion test completed successfully!")
        print("\n🚀 Next steps:")
        print("1. Train ML models on this data")
        print("2. Create visualizations")
        print("3. Deploy to Streamlit Cloud")
    else:
        print("\n❌ Data ingestion test failed")
