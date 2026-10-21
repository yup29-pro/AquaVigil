# AquaVigil Backend

FastAPI service exposing automated fish disease detection endpoints.

## Structure
```
backend/
├── app/
│   ├── main.py              # Application entrypoint & routes
│   ├── schemas.py           # Pydantic input/output schemas
│   └── services/
│       └── detection.py     # YOLOv8 inference service
└── README.md
```

## Running the Development Server

1. Ensure the Python virtual environment is activated and dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the server using Uvicorn:
   ```bash
   uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. Open interactive API docs:
   - Swagger UI: `http://localhost:8000/docs`
   - Redoc: `http://localhost:8000/redoc`

## Current Status (Phase 1)
The backend service starts in standby mode. Endpoints return status indicators stating that model training (`best.pt`) is scheduled for Phase 2.
