#!/usr/bin/env python3
"""
Run Baselines Script
=====================

Computes and displays baselines for all users.

Usage:
    python scripts/run_baselines.py [--user USER_ID]
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.logging import setup_logging, get_logger
from app.data.ingestion import CSVIngestionPipeline
from app.services.intelligence_service import get_intelligence_service
from app.services.baseline_service import get_baseline_service

setup_logging()
logger = get_logger(__name__)


def format_baseline(baseline_summary: dict) -> str:
    """Format baseline for display."""
    lines = []
    lines.append(f"User ID: {baseline_summary['user_id']}")
    lines.append(f"  Created: {baseline_summary['created_at']}")
    lines.append(f"  Mature: {baseline_summary['is_mature']}")
    lines.append(f"  Trips: {baseline_summary['trips_analyzed']}")
    lines.append(f"  Points: {baseline_summary['points_analyzed']}")
    lines.append(f"  Date Range: {baseline_summary['date_range_days']} days")
    lines.append(f"  Typical Confidence: {baseline_summary['typical_confidence']}")
    
    lines.append(f"\n  Speed Baseline:")
    lines.append(f"    Mean: {baseline_summary['speed']['mean_ms']:.2f} m/s")
    lines.append(f"    P95: {baseline_summary['speed']['p95_ms']:.2f} m/s")
    
    lines.append(f"\n  Activity Distribution:")
    for activity, pct in sorted(
        baseline_summary['activity_distribution'].items(),
        key=lambda x: -x[1]
    ):
        lines.append(f"    {activity}: {pct*100:.1f}%")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compute and display user baselines"
    )
    parser.add_argument(
        "--user",
        type=int,
        default=None,
        help="Specific user ID to analyze",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    
    args = parser.parse_args()
    
    # Get services
    intelligence_svc = get_intelligence_service()
    baseline_svc = get_baseline_service()
    pipeline = CSVIngestionPipeline()
    
    # Determine users
    if args.user:
        user_ids = [args.user]
    else:
        user_ids = pipeline.get_user_ids()
    
    print(f"Computing baselines for {len(user_ids)} users...\n")
    
    all_baselines = []
    
    for user_id in user_ids:
        try:
            # Process user if not already done
            if not baseline_svc.has_baseline(user_id):
                print(f"Processing user {user_id}...")
                intelligence_svc.process_user_data(user_id)
            
            # Get baseline
            summary = baseline_svc.get_baseline_summary(user_id)
            
            if summary:
                all_baselines.append(summary)
                
                if not args.json:
                    print("-" * 50)
                    print(format_baseline(summary))
                    print()
            
        except Exception as e:
            print(f"Error for user {user_id}: {e}")
    
    if args.json:
        print(json.dumps(all_baselines, indent=2, default=str))
    else:
        print("=" * 50)
        print(f"Computed {len(all_baselines)} baselines")


if __name__ == "__main__":
    main()
