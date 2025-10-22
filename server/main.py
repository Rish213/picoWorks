from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from core.sdk import PicoSDK  # Import from the local sdk.py file

# Pydantic Model for validating command requests
class Command(BaseModel):
    command: str

# --- App and SDK Initialization ---
sdk = PicoSDK()
app = FastAPI()

# --- Keep the user's CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---
@app.post("/connect")
def connect():
    print("[SERVER] Drone connected.")
    return sdk.connect()

@app.get("/telemetry")
def get_telemetry():
    return sdk.get_telemetry()

@app.post("/command")
def send_command(command: Command): # Upgraded to use Pydantic
    return sdk.send_command(command.command)

@app.post("/disconnect")
def disconnect():
    print("[SERVER] Drone disconnected.")
    return sdk.disconnect()

@app.get("/")
def root():
    return {"message": "picoWorks backend running."}
