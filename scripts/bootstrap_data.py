#!/usr/bin/env python3
"""
Bootstrap Data Script
======================

Processes the dataset and prepares it for use.

Usage:
    python scripts/bootstrap_data.py [--users USER_IDS] [--limit LIMIT]

This script:
1. Validates the dataset file
2. Processes all or selected users
3. Computes baselines
4. Reports statistics
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.data.ingestion import CSVIngestionPipeline
from app.services.intelligence_service import get_intelligence_service

setup_logging()
logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap Serendipity dataset for processing"
    )
    parser.add_argument(
        "--users",
        type=str,
        default=None,
        help="Comma-separated user IDs to process (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit events per user",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show dataset statistics, don't process",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Serendipity AI Backend - Data Bootstrap")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = CSVIngestionPipeline()
    
    try:
        pipeline.validate_file()
        print(f"\n✓ Dataset found: {settings.dataset_file_path}")
    except Exception as e:
        print(f"\n✗ Dataset error: {e}")
        sys.exit(1)
    
    # Get statistics
    print("\nDataset Statistics:")
    print("-" * 40)
    stats = pipeline.get_file_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    if args.stats_only:
        print("\n[Stats-only mode, not processing]")
        return
    
    # Determine users to process
    all_users = pipeline.get_user_ids()
    
    if args.users:
        user_ids = [int(u.strip()) for u in args.users.split(",")]
        # Validate user IDs
        invalid = set(user_ids) - set(all_users)
        if invalid:
            print(f"\n✗ Invalid user IDs: {invalid}")
            print(f"  Valid users: {all_users}")
            sys.exit(1)
    else:
        user_ids = all_users
    
    print(f"\nProcessing {len(user_ids)} users: {user_ids}")
    print("-" * 40)
    
    # Get service
    service = get_intelligence_service()
    
    # Process each user
    total_events = 0
    total_trips = 0
    total_time = 0
    
    for user_id in user_ids:
        print(f"\nProcessing user {user_id}...")
        start = time.time()
        
        try:
            trips, baseline = service.process_user_data(user_id, args.limit)
            
            user_events = sum(len(t.events) for t in trips)
            user_time = time.time() - start
            
            total_events += user_events
            total_trips += len(trips)
            total_time += user_time
            
            print(f"  ✓ Trips: {len(trips)}")
            print(f"  ✓ Events: {user_events}")
            print(f"  ✓ Baseline mature: {baseline.is_mature}")
            print(f"  ✓ Typical confidence: {baseline.typical_confidence:.3f}")
            print(f"  ✓ Time: {user_time:.2f}s")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Processing Complete")
    print("=" * 60)
    print(f"  Total users: {len(user_ids)}")
    print(f"  Total trips: {total_trips}")
    print(f"  Total events: {total_events}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Events/second: {total_events / total_time:.0f}")


if __name__ == "__main__":
    main()
