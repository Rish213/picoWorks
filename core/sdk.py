import random
import pandas as pd
from datetime import datetime
import os

class PicoSDK:
    def __init__(self):
        self.is_connected = False
        self.log_buffer = []
        print("SDK Initialized. Ready to connect.")

    def connect(self, port="COM3"):
        print(f"Attempting to connect to hardware on {port}...")
        self.is_connected = True
        self.log_buffer = []
        print("Connection Successful!")
        return {"status": "success", "port": port}

    def get_telemetry(self):
        if not self.is_connected:
            return {"error": "Not connected"}

        telemetry_data = {
            "timestamp": datetime.now().isoformat(),
            "roll": random.uniform(-5.0, 5.0),
            "pitch": random.uniform(-3.0, 3.0),
            "yaw": random.uniform(0, 360),
            "batt": random.uniform(3.7, 4.2)
        }
        self.log_buffer.append(telemetry_data)
        return telemetry_data

    def send_command(self, cmd):
        if not self.is_connected:
            return {"error": "Not connected"}
        print(f"Sending command to hardware: {cmd}")
        return {"status": "success", "command_sent": cmd}

    def disconnect(self):
        print("Disconnecting from hardware...")
        self.is_connected = False
        self.save_log_file()
        print("Disconnected.")
        return {"status": "success"}

    def save_log_file(self):
        """Saves telemetry data to a timestamped CSV file."""
        if not self.log_buffer:
            print("Log buffer empty. No log file saved.")
            return

        df = pd.DataFrame(self.log_buffer)
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(log_dir, f"session_{timestamp}.csv")
        df.to_csv(filepath, index=False)
        print(f"✅ Log file saved to: {filepath}")