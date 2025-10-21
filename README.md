# picoWorks
The core SDK, simulator, and ground control interface powering the picoUAV platform - built by picoWorks

# Overview
picoWorks is designed to be the developer framework for PicoZap — handling communication, telemetry logging, object detection, and control.
It’s currently in Phase 1 (Simulation & SDK Testing) and evolving toward full hardware integration (ESP32 + MPU6050 + custom PCB).

# Project Structure
picoWorks/
│
├── core/          # SDK, control logic, and computer vision modules  
├── simulator/     # Hardware-independent simulation & data emulation  
├── ui/            # Streamlit-based testing dashboard  
├── server/        # FastAPI backend bridging SDK <-> UI  
├── tests/         # Integration tests  
├── logs/          # Auto-generated telemetry logs  
└── requirements.txt  
