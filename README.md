# Serendipity AI Backend

> Production-grade AI intelligence layer for family safety geolocation tracking.

## Overview

This backend system provides AI-powered location intelligence for the Serendipity family safety application. It processes GPS tracking data to:

- **Improve Tracking Accuracy** - Clean and validate GPS coordinates
- **Score Confidence** - Rate reliability of each location point (0-1)
- **Learn Behavior Baselines** - Understand normal user patterns
- **Produce Intelligence** - Standardized, enriched location data

### Phase 1 Focus

- Observe raw signals
- Clean and normalize data
- Learn baseline behavior (observe-only)
- Produce confidence-scored intelligence

No alerts, risk decisions, or frontend impact in Phase 1.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI                               │
│  ┌─────────────┬──────────────┬─────────────────────────┐   │
│  │   Health    │   Ingest     │     Intelligence        │   │
│  │   Routes    │   Routes     │       Routes            │   │
│  └─────────────┴──────────────┴─────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                     Services Layer                           │
│  ┌───────────────────────┬──────────────────────────────┐   │
│  │  Intelligence Service │    Baseline Service          │   │
│  └───────────────────────┴──────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                    Processing Pipeline                       │
│  ┌─────────┬───────────┬──────────┬────────────┬────────┐   │
│  │ Ingest  │ Normalize │  Clean   │  Features  │ Score  │   │
│  └─────────┴───────────┴──────────┴────────────┴────────┘   │
├─────────────────────────────────────────────────────────────┤
│                     Data Layer                               │
│  ┌───────────────────────┬──────────────────────────────┐   │
│  │   CSV / File Storage  │    PostgreSQL (Phase 2+)    │   │
│  └───────────────────────┴──────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- pip or poetry

### Installation

```bash
# Clone the repository
cd serendipity-ai-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp env.example .env
```

### Running the Server

```bash
# Development mode with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or using the module directly
python -m app.main
```

### API Documentation

Once running, access:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## API Endpoints

### Health Checks

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Full health check |
| `/api/v1/health/live` | GET | Liveness probe |
| `/api/v1/health/ready` | GET | Readiness probe |

### Data Ingestion

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/ingest/process/user/{user_id}` | POST | Process all user data |
| `/api/v1/ingest/process/trip/{trip_id}` | POST | Process single trip |
| `/api/v1/ingest/dataset/stats` | GET | Get dataset statistics |
| `/api/v1/ingest/dataset/users` | GET | List users in dataset |
| `/api/v1/ingest/dataset/trips` | GET | List trips in dataset |

### Location Intelligence

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/intelligence/location/{user_id}` | GET | Get location intelligence |
| `/api/v1/intelligence/quality/{user_id}` | GET | Get data quality report |
| `/api/v1/intelligence/baseline/{user_id}` | GET | Get user baseline |
| `/api/v1/intelligence/trip/{trip_id}/summary` | GET | Get trip summary |

## Project Structure

```
serendipity-ai-backend/
├── app/
│   ├── main.py                # FastAPI entry point
│   ├── api/
│   │   ├── deps.py            # Dependency injection
│   │   └── v1/
│   │       ├── router.py      # API router
│   │       └── routes/
│   │           ├── health.py
│   │           ├── ingest.py
│   │           └── intelligence.py
│   ├── core/
│   │   ├── config.py          # Configuration
│   │   ├── constants.py       # Constants & enums
│   │   └── logging.py         # Structured logging
│   ├── data/
│   │   ├── ingestion.py       # Data loading
│   │   ├── normalization.py   # Normalization
│   │   ├── cleaning.py        # Cleaning
│   │   └── validation.py      # Validation
│   ├── features/
│   │   ├── engineering.py     # Feature computation
│   │   └── confidence.py      # Confidence scoring
│   ├── models/
│   │   ├── schemas.py         # Pydantic models
│   │   └── baseline.py        # Baseline learning
│   ├── services/
│   │   ├── intelligence_service.py
│   │   └── baseline_service.py
│   ├── db/
│   │   ├── session.py         # Database session
│   │   └── repositories.py    # Data repositories
│   └── utils/
│       ├── geo.py             # Geospatial utils
│       ├── time.py            # Time utils
│       └── metrics.py         # Statistics utils
├── scripts/
│   ├── bootstrap_data.py      # Data bootstrap
│   └── run_baselines.py       # Baseline computation
├── tests/
│   ├── conftest.py            # Test fixtures
│   ├── test_ingestion.py
│   ├── test_confidence.py
│   └── test_baselines.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── dataset/                   # Data files
├── requirements.txt
├── env.example
└── README.md
```

## Data Processing Pipeline

### 1. Ingestion
- Load raw tracking events from CSV
- Preserve strict ordering (user → trip → timestamp)
- Handle missing/invalid fields

### 2. Normalization
- Standardize timestamps to UTC
- Normalize units (m/s, meters)
- Map activity types to enums
- Generate unique event IDs

### 3. Cleaning
- Remove duplicate events
- Detect GPS drift
- Flag unrealistic jumps (teleportation)
- Mark low-quality GPS points

### 4. Feature Engineering
- Time deltas between points
- Distance (Haversine)
- Calculated speed vs reported
- Acceleration
- Stop detection
- Gap detection

### 5. Confidence Scoring
Each point receives a score (0-1) based on:
- GPS accuracy
- Speed validity
- Acceleration validity
- Temporal consistency
- Activity consistency
- Signal continuity

## Configuration

Key environment variables:

```bash
# Application
APP_ENV=development
DEBUG=true

# Dataset
DATASET_PATH=../dataset
RAW_DATA_FILE=trips-500-for-different_users.csv

# Confidence thresholds
MAX_REALISTIC_SPEED_MS=100.0
MAX_GPS_ACCURACY_METERS=100.0

# Database (Phase 2+)
DATABASE_URL=mysql+aiomysql://root:password@localhost:3306/serendipity
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_confidence.py -v
```

## Docker

```bash
# Build image
docker build -t serendipity-api -f docker/Dockerfile .

# Run with compose
docker-compose -f docker/docker-compose.yml up
```

## Phase Roadmap

### Phase 1 (Current) - Tracking Refinement ✓
- [x] Data ingestion pipeline
- [x] Cleaning & normalization
- [x] Feature engineering
- [x] Confidence scoring
- [x] Baseline learning (observe-only)
- [x] Intelligence APIs

### Phase 2 - Anomaly Detection
- [ ] Database integration
- [ ] Anomaly detection models
- [ ] Location pattern learning
- [ ] NVIDIA GPU acceleration

### Phase 3 - Predictive Intelligence
- [ ] Route prediction
- [ ] Delay forecasting
- [ ] Risk assessment
- [ ] Triton Inference Server

### Phase 4 - Production Scale
- [ ] Multi-region deployment
- [ ] Real-time streaming
- [ ] Alert system
- [ ] Mobile app integration

## Technology Stack

- **Python 3.10+** - Core language
- **FastAPI** - Web framework
- **Pydantic** - Data validation
- **Polars/Pandas** - Data processing
- **NumPy/SciPy** - Scientific computing
- **MySQL** - Database (Phase 2+)
- **PyTorch** - ML models (Phase 2+)
- **NVIDIA TensorRT** - Model optimization (Phase 2+)

## License

Proprietary - Serendipity Inc.
