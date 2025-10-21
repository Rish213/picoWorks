from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from core.sdk import PicoSDK

sdk = PicoSDK()
app = FastAPI()

# Allow local connections from UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/connect")
async def connect():
    return sdk.connect()

@app.get("/telemetry")
async def get_telemetry():
    return sdk.get_telemetry()

@app.post("/command")
async def send_command(request: Request):
    cmd = (await request.json())["command"]
    return sdk.send_command(cmd)

@app.post("/disconnect")
async def disconnect():
    return sdk.disconnect()

@app.get("/")
def root():
    return {"message": "picoWorks backend running."}
