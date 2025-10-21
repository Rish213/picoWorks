import streamlit as st
import requests
import time
import random

st.set_page_config(page_title="picoLink GCS", layout="wide")
st.title("🛰️ picoLink Ground Control")

if 'connected' not in st.session_state:
    st.session_state['connected'] = False

API_URL = "http://127.0.0.1:8000"

if not st.session_state['connected']:
    st.info("System Disconnected. Press 'Connect' to start.")
    if st.button("Connect"):
        try:
            response = requests.post(f"{API_URL}/connect")
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "success":
                st.session_state['connected'] = True
                st.rerun()
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect: {e}")

else:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.success("✅ Connected. Streaming telemetry...")
    with col2:
        if st.button("Disconnect", use_container_width=True):
            try:
                requests.post(f"{API_URL}/disconnect")
                st.session_state['connected'] = False
                st.rerun()
            except requests.exceptions.RequestException as e:
                st.error(f"Failed: {e}")

    st.markdown("---")

    telemetry_placeholder = st.empty()

    st.subheader("🎮 Actions")
    c1, c2, c3 = st.columns(3)

    def send_command(cmd):
        try:
            requests.post(f"{API_URL}/command", json={"command": cmd})
            st.toast(f"Command `{cmd}` sent!", icon="✅")
        except Exception as e:
            st.error(e)

    if c1.button("Arm", use_container_width=True):
        send_command("ARM")
    if c2.button("Calibrate IMU", use_container_width=True):
        send_command("CALIBRATE_IMU")
    if c3.button("Reset", use_container_width=True):
        send_command("RESET")

    while True:
        try:
            response = requests.get(f"{API_URL}/telemetry")
            response.raise_for_status()
            data = response.json()

            with telemetry_placeholder.container():
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Roll", f"{data['roll']:.2f}°", f"{random.uniform(-0.3, 0.3):+.2f}")
                col2.metric("Pitch", f"{data['pitch']:.2f}°", f"{random.uniform(-0.3, 0.3):+.2f}")
                col3.metric("Yaw", f"{data['yaw']:.2f}°")
                col4.metric("Battery", f"{data['batt']:.2f}V")

            time.sleep(0.5)
        except requests.exceptions.RequestException:
            st.error("Lost backend connection.")
            st.session_state['connected'] = False
            st.rerun()
