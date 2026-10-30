import sys
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Ensure root workspace is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.app.models import PredictRequest, PredictResponse, HealthResponse
from backend.app.pipeline import TranslationPipeline

pipeline = None
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../frontend"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Replaces deprecated @app.on_event('startup')."""
    global pipeline
    print("Initializing SignBridge Translation Pipeline...")
    pipeline = TranslationPipeline()
    yield
    # Shutdown cleanup (none needed currently)


app = FastAPI(
    title="SignBridge ISL Translator API",
    description="Real-Time Grammar-Aware Indian Sign Language Translator API",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    recognizer    = getattr(pipeline, "recognizer", None)
    encoder_loaded = recognizer is not None and getattr(recognizer, "encoder", None) is not None
    knn_loaded = (
        recognizer is not None
        and getattr(recognizer, "_knn_embs", None) is not None
        and getattr(recognizer, "_knn_labels", None) is not None
    )
    class_count = len(getattr(recognizer, "id_to_gloss", {})) if recognizer else 0
    is_neural   = encoder_loaded

    if is_neural and knn_loaded:
        rtype = f"OpenHands ST-GCN + KNN ({class_count} INCLUDE classes)"
    elif is_neural:
        rtype = f"OpenHands ST-GCN Decoder ({class_count} INCLUDE classes)"
    else:
        rtype = "Fallback Motion-Hash"

    return HealthResponse(
        status="ok",
        models_loaded=pipeline is not None,
        recognizer_type=rtype,
        neural_path_active=is_neural,
        encoder_loaded=encoder_loaded,
        knn_loaded=knn_loaded,
        class_count=class_count,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    result = pipeline.process(request.keypoints, mode=request.mode)
    return PredictResponse(**result)


# Serve static frontend files (CSS, JS, index.html) at root
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
