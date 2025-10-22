import streamlit as st
import requests
import time

API_BASE = "http://127.0.0.1:8000"
st.set_page_config(page_title="picoLink Ground Control", layout="wide")

st.title("🧠 picoLink Ground Control")

# --- Session State ---
if "connected" not in st.session_state:
    st.session_state.connected = False
if "last_telemetry" not in st.session_state:
    st.session_state.last_telemetry = {}

# --- Sidebar (Telemetry Log) ---
st.sidebar.title("Telemetry Log")
telemetry_log = st.sidebar.empty()

# --- Connection Section ---
status_placeholder = st.empty()

if not st.session_state.connected:
    connect_btn = st.button("🔌 Connect")
    if connect_btn:
        try:
            res = requests.post(f"{API_BASE}/connect")
            if res.status_code == 200:
                st.session_state.connected = True
                st.success("✅ Connected to PicoSDK")
            else:
                st.error("❌ Connection failed.")
        except Exception as e:
            st.error(f"Connection error: {e}")
else:
    disconnect_btn = st.button("❌ Disconnect")
    if disconnect_btn:
        try:
            res = requests.post(f"{API_BASE}/disconnect")
            if res.status_code == 200:
                st.session_state.connected = False
                st.warning("🔌 Disconnected.")
            else:
                st.error("❌ Disconnection failed.")
        except Exception as e:
            st.error(f"Disconnection error: {e}")

# --- Command Buttons ---
st.subheader("Commands")
col1, col2, col3 = st.columns(3)
if st.session_state.connected:
    with col1:
        if st.button("ARM"):
            requests.post(f"{API_BASE}/command", json={"command": "ARM"})
    with col2:
        if st.button("CALIBRATE_IMU"):
            requests.post(f"{API_BASE}/command", json={"command": "CALIBRATE_IMU"})
    with col3:
        if st.button("RESET"):
            requests.post(f"{API_BASE}/command", json={"command": "RESET"})

# --- Telemetry Display ---
st.subheader("Telemetry")
cols = st.columns(4)
col_labels = ["Altitude (m)", "Battery (V)", "Temperature (°C)", "Motor RPM"]

# --- Live Telemetry Update ---
if st.session_state.connected:
    placeholder_vals = [c.empty() for c in cols]

    while st.session_state.connected:
        try:
            telemetry = requests.get(f"{API_BASE}/telemetry").json()
            st.session_state.last_telemetry = telemetry
            telemetry_log.write(telemetry)

            # Display telemetry numerically
            placeholder_vals[0].metric(label="Altitude (m)", value=f"{telemetry['altitude']:.2f}")
            placeholder_vals[1].metric(label="Battery (V)", value=f"{telemetry['battery']:.2f}")
            placeholder_vals[2].metric(label="Temperature (°C)", value=f"{telemetry['temperature']:.1f}")
            placeholder_vals[3].metric(label="Motor RPM", value=f"{sum(telemetry['motor_rpm'])//4}")

            time.sleep(0.5)
        except Exception as e:
            st.error(f"Telemetry error: {e}")
            break
else:
    st.info("Connect to start telemetry stream.")
