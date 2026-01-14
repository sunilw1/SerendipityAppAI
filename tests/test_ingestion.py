"""
Tests for Data Ingestion
=========================

Tests for the data ingestion pipeline.
"""

import pytest
from pathlib import Path

from app.data.ingestion import (
    CSVIngestionPipeline,
    generate_event_hash,
    generate_event_id,
)
from app.models.schemas import RawTrackingEvent


class TestEventIdGeneration:
    """Tests for event ID generation."""
    
    def test_generate_event_id(self):
        """Test event ID format."""
        event_id = generate_event_id(123, 456, 789)
        assert event_id == "u123_t456_p789"
    
    def test_generate_event_id_zero(self):
        """Test event ID with zero values."""
        event_id = generate_event_id(0, 0, 0)
        assert event_id == "u0_t0_p0"


class TestEventHashGeneration:
    """Tests for event hash generation."""
    
    def test_generate_event_hash(self, sample_raw_event: RawTrackingEvent):
        """Test hash generation produces consistent results."""
        hash1 = generate_event_hash(sample_raw_event)
        hash2 = generate_event_hash(sample_raw_event)
        
        assert hash1 == hash2
        assert len(hash1) == 16
    
    def test_different_events_different_hashes(
        self,
        sample_raw_events: list[RawTrackingEvent]
    ):
        """Test different events produce different hashes."""
        hashes = [generate_event_hash(e) for e in sample_raw_events]
        
        # All hashes should be unique
        assert len(hashes) == len(set(hashes))


class TestCSVIngestionPipeline:
    """Tests for CSV ingestion pipeline."""
    
    def test_pipeline_initialization(self):
        """Test pipeline can be initialized."""
        pipeline = CSVIngestionPipeline()
        assert pipeline.batch_size > 0
    
    def test_pipeline_with_custom_path(self, tmp_path: Path):
        """Test pipeline with custom file path."""
        test_file = tmp_path / "test.csv"
        pipeline = CSVIngestionPipeline(file_path=test_file)
        
        assert pipeline.file_path == test_file
    
    def test_validate_missing_file(self, tmp_path: Path):
        """Test validation fails for missing file."""
        pipeline = CSVIngestionPipeline(file_path=tmp_path / "nonexistent.csv")
        
        with pytest.raises(Exception):
            pipeline.validate_file()
    
    def test_get_ingestion_stats(self):
        """Test ingestion stats are returned."""
        pipeline = CSVIngestionPipeline()
        stats = pipeline.get_ingestion_stats()
        
        assert "total_rows" in stats
        assert "successful_rows" in stats
        assert "failed_rows" in stats
