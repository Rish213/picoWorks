import random
import time
from datetime import datetime
import os

# Note: This SDK will require 'pandas' for logging.
# Make sure it's installed: pip install pandas

class PicoSDK:
    def __init__(self):
        self.connected = False
        self.log_buffer = []
        self.last_message_time = time.time()
        print("[SDK] Initialized.")

    def connect(self):
        self.connected = True
        self.log_buffer = [] # Clear log buffer on new connection
        print("[SDK] Connection established.")
        return {"status": "connected"}

    def disconnect(self):
        self.connected = False
        self.save_log_file()
        print("[SDK] Connection terminated.")
        return {"status": "disconnected"}

    def send_command(self, command: str):
        if not self.connected:
            return {"error": "not connected"}
        print(f"[SDK] Received command: {command}")
        return {"response": f"executed: {command}"}

    def get_telemetry(self):
        """
        Upgraded to provide a rich telemetry packet for the new UI.
        """
        if not self.connected:
            return {"error": "not connected"}
            
        # Generate periodic messages
        messages = []
        if time.time() - self.last_message_time > random.uniform(3, 8):
            msg_type = random.choice(["INFO", "WARN", "ERROR"])
            msg_content = random.choice([
                "Calibrating Gyro", "GPS Lock Acquired (8 Sat)", 
                "Low Signal (RTL suggested)", "System OK", "High Wind Alert"
            ])
            messages.append(f"{datetime.now().strftime('%H:%M:%S')} [{msg_type}] {msg_content}")
            self.last_message_time = time.time()

        # Simulate data
        data = {
            "timestamp": datetime.now().isoformat(),
            "roll": random.uniform(-25.0, 25.0),
            "pitch": random.uniform(-15.0, 15.0),
            "yaw": random.uniform(0, 360),
            "battery": max(0, 99.0 - (time.time() - self.last_message_time) / 100), # Mock battery drain
            "signal": random.randint(70, 100),
            "cpu_temp": random.uniform(40.0, 55.0),
            "altitude": random.uniform(0.0, 50.0),
            "messages": messages,
            "video_frame_url": random.choice([
                "https://i.imgur.com/L4i1Jm2.jpeg",
                "https://i.imgur.com/bZOK2XI.jpeg",
                "https://i.imgur.com/Xk5A2Z0.jpeg"
            ])
        }
        
        # Add to log buffer
        self.log_buffer.append(data)
        return data

    def save_log_file(self):
        try:
            import pandas as pd
            if not self.log_buffer:
                print("[SDK] Log buffer empty. No file saved.")
                return

            df = pd.DataFrame(self.log_buffer)
            log_dir = "logs"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(log_dir, f"session_{timestamp}.csv")
            
            df.to_csv(filepath, index=False)
            print(f"[SDK] Log file saved to: {filepath}")
        
        except ImportError:
            print("[SDK] 'pandas' library not found. Log file not saved.")
        except Exception as e:
            print(f"[SDK] Error saving log file: {e}")
