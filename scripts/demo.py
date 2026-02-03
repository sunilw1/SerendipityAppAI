#!/usr/bin/env python3
"""
Serendipity AI Backend - Demo Script
======================================

This script demonstrates all Phase 1 capabilities for client presentation.

Run with: python scripts/demo.py
Generate report: python scripts/demo.py --report

Features demonstrated:
1. Dataset loading and statistics
2. Data processing pipeline
3. Confidence scoring
4. Baseline learning
5. Location intelligence output
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from io import StringIO
from typing import Optional, List, Tuple, TextIO

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.logging import setup_logging
from app.data.ingestion import CSVIngestionPipeline
from app.services.intelligence_service import get_intelligence_service

# Suppress verbose logging for demo
import logging
logging.getLogger().setLevel(logging.WARNING)


class DemoOutput:
    """Handles output to console and/or file."""
    
    def __init__(self, file_path: Optional[Path] = None):
        self.file_path = file_path
        self.buffer = StringIO()
        self.file_handle: Optional[TextIO] = None
        
        if file_path:
            self.file_handle = open(file_path, 'w', encoding='utf-8')
    
    def print(self, text: str = ""):
        """Print to console and optionally to file."""
        print(text)
        self.buffer.write(text + "\n")
        if self.file_handle:
            self.file_handle.write(text + "\n")
    
    def close(self):
        """Close file handle if open."""
        if self.file_handle:
            self.file_handle.close()
    
    def get_content(self) -> str:
        """Get all buffered content."""
        return self.buffer.getvalue()


# Global output handler
out = DemoOutput()


def print_header(title: str):
    """Print a section header."""
    out.print("\n" + "=" * 70)
    out.print(f"  {title}")
    out.print("=" * 70)


def print_subheader(title: str):
    """Print a subsection header."""
    out.print(f"\n--- {title} ---")


def demo_dataset_overview() -> bool:
    """Demonstrate dataset loading and overview."""
    print_header("1. DATASET OVERVIEW")
    
    pipeline = CSVIngestionPipeline()
    
    try:
        pipeline.validate_file()
        out.print("✓ Dataset file validated successfully")
    except Exception as e:
        out.print(f"✗ Dataset error: {e}")
        return False
    
    stats = pipeline.get_file_stats()
    
    out.print(f"\n  File: {stats['file_path']}")
    out.print(f"  Size: {stats['file_size_mb']} MB")
    out.print(f"  Total Records: {stats['total_rows']:,}")
    out.print(f"  Unique Users: {stats['unique_users']}")
    out.print(f"  Unique Trips: {stats['unique_trips']}")
    
    user_ids = pipeline.get_user_ids()
    out.print(f"\n  User IDs: {user_ids}")
    
    return True


def demo_data_processing(user_id: int, limit: int = 5000):
    """Demonstrate the data processing pipeline."""
    print_header("2. DATA PROCESSING PIPELINE")
    
    service = get_intelligence_service()
    
    out.print(f"\nProcessing data for User {user_id}...")
    out.print("  Step 1: Loading raw events from CSV")
    out.print("  Step 2: Normalizing timestamps and coordinates")
    out.print("  Step 3: Cleaning data (removing duplicates, detecting drift)")
    out.print("  Step 4: Computing features (distances, speeds, stops)")
    out.print("  Step 5: Scoring confidence for each point")
    out.print("  Step 6: Learning baseline behavior")
    
    start = time.time()
    trips, baseline = service.process_user_data(user_id, limit=limit)
    duration = time.time() - start
    
    total_events = sum(len(t.events) for t in trips)
    
    out.print(f"\n✓ Processing complete in {duration:.2f} seconds")
    out.print(f"  Trips processed: {len(trips)}")
    out.print(f"  Events processed: {total_events:,}")
    out.print(f"  Processing speed: {total_events/duration:.0f} events/second")
    
    return trips, baseline


def demo_confidence_scoring(trips):
    """Demonstrate confidence scoring."""
    print_header("3. CONFIDENCE SCORING")
    
    # Collect all confidence scores
    all_scores = []
    high_conf = 0
    medium_conf = 0
    low_conf = 0
    
    for trip in trips:
        for event in trip.events:
            score = event.confidence.overall
            all_scores.append(score)
            
            if score >= 0.75:
                high_conf += 1
            elif score >= 0.5:
                medium_conf += 1
            else:
                low_conf += 1
    
    total = len(all_scores)
    avg_score = sum(all_scores) / total if total > 0 else 0
    
    out.print("\n  Confidence Score Distribution:")
    out.print(f"  ├── High (≥0.75):   {high_conf:5,} ({high_conf/total*100:5.1f}%)" if total else "")
    out.print(f"  ├── Medium (0.5-0.75): {medium_conf:5,} ({medium_conf/total*100:5.1f}%)" if total else "")
    out.print(f"  └── Low (<0.5):     {low_conf:5,} ({low_conf/total*100:5.1f}%)" if total else "")
    out.print(f"\n  Average Confidence: {avg_score:.3f}")
    
    # Show sample events with different confidence levels
    print_subheader("Sample Events by Confidence Level")
    
    # Find high confidence example
    for trip in trips[:3]:
        for event in trip.events[:50]:
            conf = event.confidence
            if conf.overall >= 0.85:
                out.print(f"\n  HIGH CONFIDENCE EVENT (score: {conf.overall:.3f})")
                out.print(f"    Location: ({event.event.location.lat:.6f}, {event.event.location.lon:.6f})")
                out.print(f"    GPS Accuracy: {event.event.gps_accuracy_meters}m")
                out.print(f"    Speed: {event.event.speed_ms:.1f} m/s")
                out.print(f"    Activity: {event.event.activity_type.value}")
                out.print(f"    Component Scores:")
                out.print(f"      GPS Accuracy:  {conf.gps_accuracy_score:.2f}")
                out.print(f"      Speed Valid:   {conf.speed_validity_score:.2f}")
                out.print(f"      Temporal:      {conf.temporal_consistency_score:.2f}")
                break
        break
    
    # Find low confidence example
    for trip in trips:
        for event in trip.events:
            conf = event.confidence
            if 0.3 <= conf.overall <= 0.5:
                out.print(f"\n  LOW CONFIDENCE EVENT (score: {conf.overall:.3f})")
                out.print(f"    Location: ({event.event.location.lat:.6f}, {event.event.location.lon:.6f})")
                out.print(f"    GPS Accuracy: {event.event.gps_accuracy_meters}m")
                out.print(f"    Quality Flags: {event.event.quality_flags}")
                flags_str = str(conf.flags[:3]) + "..." if len(conf.flags) > 3 else str(conf.flags)
                out.print(f"    Confidence Flags: {flags_str}")
                break
        else:
            continue
        break


def demo_baseline_learning(baseline):
    """Demonstrate baseline learning."""
    print_header("4. BASELINE BEHAVIOR LEARNING")
    
    out.print(f"\n  User ID: {baseline.user_id}")
    out.print(f"  Baseline Status: {'MATURE ✓' if baseline.is_mature else 'DEVELOPING'}")
    out.print(f"  Data Points Analyzed: {baseline.points_analyzed:,}")
    out.print(f"  Trips Analyzed: {baseline.trips_analyzed}")
    out.print(f"  Date Range: {baseline.date_range_days:.1f} days")
    
    print_subheader("Speed Patterns")
    speed = baseline.speed_baseline
    out.print(f"  Mean Speed: {speed.mean:.2f} m/s ({speed.mean * 3.6:.1f} km/h)")
    out.print(f"  Median Speed: {speed.median:.2f} m/s")
    out.print(f"  Max Speed: {speed.max_value:.2f} m/s ({speed.max_value * 3.6:.1f} km/h)")
    out.print(f"  95th Percentile: {speed.p95:.2f} m/s")
    
    print_subheader("Activity Distribution")
    sorted_activities = sorted(
        baseline.activity_distribution.items(),
        key=lambda x: -x[1]
    )
    for activity, pct in sorted_activities:
        bar = "█" * int(pct * 40)
        out.print(f"  {activity:15} {pct*100:5.1f}% {bar}")
    
    out.print(f"\n  Typical Confidence: {baseline.typical_confidence:.3f}")


def demo_trip_analysis(trips):
    """Demonstrate trip-level analysis."""
    print_header("5. TRIP INTELLIGENCE")
    
    if not trips:
        out.print("  No trips to analyze")
        return
    
    # Show details for first few trips
    for i, trip in enumerate(trips[:3]):
        summary = trip.summary
        
        out.print(f"\n  TRIP {summary.trip_id}")
        out.print(f"  ├── Duration: {summary.duration_seconds/60:.1f} minutes")
        out.print(f"  ├── Distance: {summary.distance_meters/1000:.2f} km")
        out.print(f"  ├── Points: {summary.total_points} (valid: {summary.valid_points})")
        out.print(f"  ├── Avg Confidence: {summary.average_confidence:.3f}")
        out.print(f"  ├── Primary Activity: {summary.primary_activity.value}")
        out.print(f"  ├── Start: ({summary.start_location.lat:.4f}, {summary.start_location.lon:.4f})")
        out.print(f"  └── End: ({summary.end_location.lat:.4f}, {summary.end_location.lon:.4f})")


def demo_data_quality(service, user_id):
    """Demonstrate data quality reporting."""
    print_header("6. DATA QUALITY REPORT")
    
    report = service.get_data_quality_report(user_id)
    
    out.print(f"\n  Total Events: {report.total_events:,}")
    out.print(f"  Valid Events: {report.valid_events:,} ({report.validity_rate*100:.1f}%)")
    out.print(f"  Flagged Events: {report.flagged_events:,}")
    
    print_subheader("Quality Flags Detected")
    if report.flag_counts:
        sorted_flags = sorted(
            report.flag_counts.items(),
            key=lambda x: -x[1]
        )
        for flag, count in sorted_flags[:10]:
            pct = count / report.total_events * 100
            out.print(f"  {flag:35} {count:6,} ({pct:5.2f}%)")
    else:
        out.print("  No quality issues detected!")
    
    print_subheader("Confidence Statistics")
    out.print(f"  Mean: {report.confidence_mean:.3f}")
    out.print(f"  Std Dev: {report.confidence_std:.3f}")
    out.print(f"  Min: {report.confidence_min:.3f}")
    out.print(f"  Max: {report.confidence_max:.3f}")


def demo_intelligence_output(service, user_id):
    """Demonstrate the final intelligence output."""
    print_header("7. LOCATION INTELLIGENCE OUTPUT")
    
    out.print("\n  This is the standardized output exposed via API:")
    out.print("  (Clean, confidence-scored, no raw data)")
    
    intelligence, total = service.get_location_intelligence(
        user_id=user_id,
        min_confidence=0.7,
        limit=5,
    )
    
    out.print(f"\n  Total high-confidence events: {total}")
    out.print(f"\n  Sample Intelligence Records (confidence ≥ 0.7):")
    
    for i, loc in enumerate(intelligence[:5], 1):
        out.print(f"\n  [{i}] Event: {loc.event_id}")
        out.print(f"      Time: {loc.timestamp}")
        out.print(f"      Location: ({loc.latitude:.6f}, {loc.longitude:.6f})")
        out.print(f"      Accuracy: {loc.accuracy_meters}m")
        out.print(f"      Speed: {loc.speed_ms:.1f} m/s")
        out.print(f"      Activity: {loc.activity}")
        out.print(f"      Confidence: {loc.confidence:.3f} ({loc.confidence_level})")
        out.print(f"      Quality Issues: {'Yes' if loc.has_quality_issues else 'No'}")


def demo_api_endpoints():
    """Show available API endpoints."""
    print_header("8. API ENDPOINTS (Ready for Integration)")
    
    out.print("""
  HEALTH CHECKS
  ├── GET  /api/v1/health          - Full health check
  ├── GET  /api/v1/health/live     - Liveness probe
  └── GET  /api/v1/health/ready    - Readiness probe

  DATA INGESTION
  ├── POST /api/v1/ingest/process/user/{user_id}  - Process user data
  ├── POST /api/v1/ingest/process/trip/{trip_id}  - Process single trip
  ├── GET  /api/v1/ingest/dataset/stats           - Dataset statistics
  ├── GET  /api/v1/ingest/dataset/users           - List users
  └── GET  /api/v1/ingest/dataset/trips           - List trips

  LOCATION INTELLIGENCE
  ├── GET  /api/v1/intelligence/location/{user_id}      - Get locations
  ├── GET  /api/v1/intelligence/quality/{user_id}       - Quality report
  ├── GET  /api/v1/intelligence/baseline/{user_id}      - User baseline
  ├── GET  /api/v1/intelligence/trip/{trip_id}/summary  - Trip summary
  └── GET  /api/v1/intelligence/confidence/distribution/{user_id}

  Documentation: http://localhost:8000/docs (Swagger UI)
    """)


def print_summary():
    """Print final summary."""
    print_header("PHASE 1 DELIVERABLES - COMPLETE ✓")
    out.print("""
  ✓ Real-time data ingestion pipeline
  ✓ Data cleaning & normalization
  ✓ Feature engineering (speed, distance, acceleration, stops)
  ✓ Confidence scoring system (0-1 scale, multi-factor)
  ✓ Baseline behavior learning (observe-only mode)
  ✓ Clean intelligence APIs (RESTful, versioned)
  ✓ Production-ready architecture (FastAPI, MySQL, Docker)
  ✓ Extensible for Phase 2+ (anomaly detection, NVIDIA acceleration)
    """)
    
    out.print("  Ready for Phase 2: Anomaly Detection & Alert System")
    out.print("\n" + "=" * 70 + "\n")


def run_demo(interactive: bool = True, limit: int = 5000):
    """Run the complete demo."""
    global out
    
    out.print("\n" + "╔" + "═" * 68 + "╗")
    out.print("║" + " " * 68 + "║")
    out.print("║" + "  SERENDIPITY AI BACKEND - PHASE 1 DEMO".center(68) + "║")
    out.print("║" + "  Family Safety Geolocation Intelligence".center(68) + "║")
    out.print("║" + " " * 68 + "║")
    out.print("╚" + "═" * 68 + "╝")
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    out.print(f"\n  Date: {timestamp}")
    out.print("  Phase: 1 - Tracking Refinement & Data Intelligence Foundation")
    
    # Step 1: Dataset overview
    if not demo_dataset_overview():
        out.print("\n✗ Cannot proceed without dataset")
        return False
    
    # Get pipeline for user selection
    pipeline = CSVIngestionPipeline()
    user_ids = pipeline.get_user_ids()
    user_id = user_ids[0]  # Use first user for demo
    
    out.print(f"\n  Demo will use User ID: {user_id}")
    if interactive:
        input("\n  Press Enter to continue...")
    
    # Step 2: Data processing
    trips, baseline = demo_data_processing(user_id, limit=limit)
    if interactive:
        input("\n  Press Enter to continue...")
    
    # Step 3: Confidence scoring
    demo_confidence_scoring(trips)
    if interactive:
        input("\n  Press Enter to continue...")
    
    # Step 4: Baseline learning
    demo_baseline_learning(baseline)
    if interactive:
        input("\n  Press Enter to continue...")
    
    # Step 5: Trip analysis
    demo_trip_analysis(trips)
    if interactive:
        input("\n  Press Enter to continue...")
    
    # Step 6: Data quality
    service = get_intelligence_service()
    demo_data_quality(service, user_id)
    if interactive:
        input("\n  Press Enter to continue...")
    
    # Step 7: Intelligence output
    demo_intelligence_output(service, user_id)
    if interactive:
        input("\n  Press Enter to continue...")
    
    # Step 8: API endpoints
    demo_api_endpoints()
    
    # Summary
    print_summary()
    
    return True


def generate_markdown_report(content: str, output_path: Path):
    """Generate a markdown report from demo output."""
    report = f"""# Serendipity AI Backend - Phase 1 Completion Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Version:** 1.0.0  
**Status:** ✅ Phase 1 Complete

---

## Executive Summary

Phase 1 of the Serendipity AI Backend has been successfully completed. The system provides:

- **Real-time GPS data ingestion** with validation and cleaning
- **Intelligent confidence scoring** for every location point
- **Baseline behavior learning** to understand normal patterns
- **Production-ready APIs** for frontend integration

---

## Demo Output

```
{content}
```

---

## Technical Specifications

### Architecture
- **Framework:** FastAPI (Python 3.10+)
- **Database:** MySQL 8.0
- **Containerization:** Docker + Docker Compose
- **API Version:** v1 (versioned endpoints)

### Data Pipeline
1. **Ingestion:** CSV/real-time event loading
2. **Validation:** Coordinate bounds, timestamp integrity
3. **Normalization:** UTC timestamps, standardized units
4. **Cleaning:** Duplicate removal, drift detection
5. **Feature Engineering:** Speed, distance, acceleration, stops
6. **Confidence Scoring:** Multi-factor 0-1 score

### Confidence Score Components
| Component | Weight | Description |
|-----------|--------|-------------|
| GPS Accuracy | 25% | Raw accuracy from device |
| Speed Validity | 20% | Physical plausibility |
| Temporal Consistency | 20% | Time ordering |
| Activity Match | 15% | Speed vs activity type |
| Signal Continuity | 20% | Gap detection |

---

## Next Steps (Phase 2)

- [ ] Anomaly detection models
- [ ] Alert system integration
- [ ] NVIDIA GPU acceleration
- [ ] Triton Inference Server deployment

---

*This report was automatically generated by the Serendipity AI Backend demo system.*
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📄 Markdown report saved to: {output_path}")


def main():
    """Main entry point with argument parsing."""
    global out
    
    parser = argparse.ArgumentParser(
        description="Serendipity AI Backend - Phase 1 Demo"
    )
    parser.add_argument(
        '--report', '-r',
        action='store_true',
        help='Generate a report file (non-interactive mode)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='phase1_demo_report',
        help='Output file name (without extension)'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=5000,
        help='Limit events to process (default: 5000)'
    )
    
    args = parser.parse_args()
    
    # Determine output paths
    output_dir = Path(__file__).parent.parent / "reports"
    output_dir.mkdir(exist_ok=True)
    
    if args.report:
        # Non-interactive mode with file output
        txt_path = output_dir / f"{args.output}.txt"
        md_path = output_dir / f"{args.output}.md"
        
        out = DemoOutput(txt_path)
        print(f"\n📝 Running demo in report mode...")
        print(f"   Output will be saved to: {output_dir}/")
        
        success = run_demo(interactive=False, limit=args.limit)
        
        if success:
            # Generate markdown report
            generate_markdown_report(out.get_content(), md_path)
            print(f"📄 Text output saved to: {txt_path}")
            print(f"\n✅ Reports generated successfully!")
            print(f"   Share these files with your client:")
            print(f"   - {txt_path.name} (raw output)")
            print(f"   - {md_path.name} (formatted report)")
        
        out.close()
    else:
        # Interactive mode
        out = DemoOutput()
        run_demo(interactive=True, limit=args.limit)


if __name__ == "__main__":
    main()
