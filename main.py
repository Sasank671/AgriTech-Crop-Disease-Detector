"""
AgriTech Crop Disease Detector - FastAPI Backend
POST /predict       → upload leaf image → disease + treatment (JWT required)
GET  /history       → user's past scans  (JWT required)
GET  /health        → health check
GET  /classes       → list all 38 disease classes
POST /register      → create account
POST /login         → sign in
POST /language      → set language preference (JWT required)
"""

import json, os, time, logging
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt
import numpy as np
from PIL import Image
import onnxruntime as ort
import io

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import OAuth2PasswordBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, model_validator, field_validator

from sqlalchemy import (
    create_engine, Column, Integer, String, Boolean, Float,
    DateTime, Text, ForeignKey, JSON
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH       = Path("output/model.onnx")
CLASS_NAMES_PATH = Path("output/class_names.json")
TREATMENTS_PATH  = Path("treatments.json")

IMAGE_SIZE    = 300
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SECRET_KEY   = os.getenv("SECRET_KEY", "change-me-before-deploying")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/krishiai")

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, index=True)
    username      = Column(String(64), unique=True, nullable=False, index=True)
    email         = Column(String(256), unique=True, nullable=False)
    hashed_password = Column(String(256), nullable=False)
    language      = Column(String(8), default="en")
    is_onboarded  = Column(Boolean, default=False)
    created_at    = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    scans = relationship("ScanHistory", back_populates="user", cascade="all, delete-orphan")


class ScanHistory(Base):
    __tablename__ = "scan_history"

    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(Integer, ForeignKey("users.id"), nullable=False)
    predicted_class = Column(String(128))
    confidence      = Column(Float)
    is_healthy      = Column(Boolean)
    severity_level  = Column(String(16))
    treatment       = Column(JSON, nullable=True)
    top3            = Column(JSON, nullable=True)
    inference_ms    = Column(Float)
    scanned_at      = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="scans")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ─────────────────────────────────────────────
# PASSWORD HELPERS
# ─────────────────────────────────────────────
def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())

# ─────────────────────────────────────────────
# SEVERITY SCORING
# ─────────────────────────────────────────────
def get_severity(confidence: float) -> dict:
    if confidence >= 0.85:
        return {"level": "High",   "label": "🔴 Severe",      "advice": "Immediate treatment required. Disease is well-established."}
    elif confidence >= 0.60:
        return {"level": "Medium", "label": "🟡 Moderate",    "advice": "Treat within 3-5 days. Disease is progressing."}
    else:
        return {"level": "Low",    "label": "🟢 Early Stage", "advice": "Monitor closely. Treat as a precaution."}

# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────
class LoginSchema(BaseModel):
    username: str
    password: str

class OnBoardingSchema(BaseModel):
    language: str

class RegisterSchema(BaseModel):
    username: str
    email: EmailStr
    password: str
    confirm_password: str
    language: Optional[str] = None
    is_onboarded: Optional[bool] = False

    @field_validator("password")
    @classmethod
    def strong_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must have at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must have at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must have at least one number")
        if not any(c in "!@#$%^&*" for c in v):
            raise ValueError("Password must have at least one special character")
        return v

    @model_validator(mode="after")
    def passwords_match(self):
        if self.password != self.confirm_password:
            raise ValueError("Passwords do not match")
        return self

# ─────────────────────────────────────────────
# RESPONSE MODELS
# ─────────────────────────────────────────────
class TreatmentInfo(BaseModel):
    disease_common_name: str
    pathogen: Optional[str]
    crop: str
    pesticide_common_name: Optional[str]
    trade_name: Optional[str]
    dosage: Optional[str]
    application_method: Optional[str]
    frequency: Optional[str]
    additional_note: Optional[str]
    source: Optional[str]
    india_approved: Optional[bool]
    # Curated Hindi for the prose fields (pesticide/trade/dosage stay English).
    # Optional so entries without a Hindi translation still validate.
    application_method_hi: Optional[str] = None
    frequency_hi: Optional[str] = None
    additional_note_hi: Optional[str] = None

class SeverityInfo(BaseModel):
    level: str
    label: str
    advice: str

class PredictionResponse(BaseModel):
    success: bool
    predicted_class: str
    confidence: float
    confidence_pct: str
    severity: SeverityInfo
    is_healthy: bool
    treatment: Optional[TreatmentInfo]
    top3: list
    inference_time_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    num_classes: int

# ─────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────
class AppState:
    ort_session: ort.InferenceSession = None
    class_names: list = []
    treatments: dict = {}

state = AppState()

# ─────────────────────────────────────────────
# LIFESPAN
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ready.")

    logger.info("Loading ONNX model...")
    if not MODEL_PATH.exists():
        logger.warning(f"Model not found at {MODEL_PATH}. Run train.py first.")
    else:
        state.ort_session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
        logger.info("ONNX model loaded.")

    if CLASS_NAMES_PATH.exists():
        with open(CLASS_NAMES_PATH) as f:
            state.class_names = json.load(f)
        logger.info(f"Loaded {len(state.class_names)} class names.")

    if TREATMENTS_PATH.exists():
        with open(TREATMENTS_PATH) as f:
            state.treatments = json.load(f)
        logger.info(f"Loaded {len(state.treatments)} treatment entries.")

    yield
    logger.info("Shutting down.")

# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────
app = FastAPI(
    title="AgriTech Crop Disease Detector API",
    description="Upload a crop leaf image to detect disease and get treatment recommendations.",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="krishiai"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("krishiai/index.html")

# ─────────────────────────────────────────────
# JWT HELPERS
# ─────────────────────────────────────────────
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

def create_token(username: str) -> str:
    payload = {
        "username": username,
        "exp": datetime.now(timezone.utc) + timedelta(hours=24)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

async def verify_token(token: str = Depends(oauth2_scheme)) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(
    token_data: dict = Depends(verify_token),
    db: Session = Depends(get_db),
) -> User:
    """Decode the JWT and load the matching user in one step.

    Every protected route needs the authenticated User row, so doing the
    lookup here means each request hits the users table exactly once and the
    route bodies stay free of repeated token/404 boilerplate.
    """
    user = db.query(User).filter(User.username == token_data["username"]).first()
    if not user:
        raise HTTPException(status_code=401, detail="User no longer exists")
    return user

# ─────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)
    return np.expand_dims(arr, axis=0)

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()

# ─────────────────────────────────────────────
# ROUTES — public
# ─────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok", "model_loaded": state.ort_session is not None, "num_classes": len(state.class_names)}

@app.get("/classes")
def get_classes():
    return {"num_classes": len(state.class_names), "classes": state.class_names}

@app.post("/register")
async def register_user(user: RegisterSchema, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == user.username).first():
        raise HTTPException(status_code=409, detail="Username already taken")
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=409, detail="Email already registered")

    new_user = User(
        username=user.username,
        email=user.email,
        hashed_password=hash_password(user.password),
        language=user.language or "en",
        is_onboarded=False,
    )
    db.add(new_user)
    db.commit()

    return {"access_token": create_token(user.username), "is_onboarded": False}

@app.post("/login")
async def user_login(user: LoginSchema, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {
        "access_token": create_token(user.username),
        "is_onboarded": db_user.is_onboarded,
        "language": db_user.language,
    }

# ─────────────────────────────────────────────
# ROUTES — protected
# ─────────────────────────────────────────────
@app.post("/language")
async def user_language_pref(
    language: OnBoardingSchema,
    db_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    db_user.language    = language.language
    db_user.is_onboarded = True
    db.commit()

    return {"message": "Preferred language saved"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    db_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if state.ort_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        input_tensor = preprocess_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image: {e}")

    input_name = state.ort_session.get_inputs()[0].name
    t0 = time.time()
    outputs = state.ort_session.run(None, {input_name: input_tensor})
    inference_ms = (time.time() - t0) * 1000

    logits = outputs[0][0]
    probs  = softmax(logits)

    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [
        {
            "class": state.class_names[i] if i < len(state.class_names) else f"class_{i}",
            "confidence": float(round(probs[i], 4)),
            "confidence_pct": f"{probs[i]*100:.1f}%",
        }
        for i in top3_idx
    ]

    pred_idx   = int(np.argmax(probs))
    pred_class = state.class_names[pred_idx] if pred_idx < len(state.class_names) else f"class_{pred_idx}"
    confidence = float(round(probs[pred_idx], 4))
    is_healthy = "healthy" in pred_class.lower()
    severity   = get_severity(confidence)

    treatment = state.treatments.get(pred_class)
    if treatment is None:
        for key in state.treatments:
            if key.lower().replace(" ", "_") == pred_class.lower().replace(" ", "_"):
                treatment = state.treatments[key]
                break

    # persist scan to history (db_user comes straight from the auth dependency)
    scan = ScanHistory(
        user_id=db_user.id,
        predicted_class=pred_class,
        confidence=confidence,
        is_healthy=is_healthy,
        severity_level=severity["level"],
        treatment=treatment,
        top3=top3,
        inference_ms=round(inference_ms, 2),
    )
    db.add(scan)
    db.commit()

    return PredictionResponse(
        success=True,
        predicted_class=pred_class,
        confidence=confidence,
        confidence_pct=f"{confidence*100:.1f}%",
        severity=SeverityInfo(**severity),
        is_healthy=is_healthy,
        treatment=TreatmentInfo(**treatment) if treatment else None,
        top3=top3,
        inference_time_ms=round(inference_ms, 2),
    )


@app.get("/history")
async def get_history(
    db_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 20,
):
    scans = (
        db.query(ScanHistory)
        .filter(ScanHistory.user_id == db_user.id)
        .order_by(ScanHistory.scanned_at.desc())
        .limit(limit)
        .all()
    )

    return [
        {
            "id": s.id,
            "predicted_class": s.predicted_class,
            "confidence": s.confidence,
            "confidence_pct": f"{s.confidence*100:.1f}%",
            "is_healthy": s.is_healthy,
            "severity_level": s.severity_level,
            "treatment": s.treatment,
            "top3": s.top3,
            "scanned_at": s.scanned_at.isoformat(),
        }
        for s in scans
    ]
