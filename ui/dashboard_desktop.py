import sys
import io
import traceback
from functools import partial

import requests
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QPixmap, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QGroupBox, QTextEdit, QCheckBox, QSizePolicy, QMessageBox
)

# Matplotlib imports for Qt embedding
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import math
import time
import random

# ---------------------------
# HUD Widget (Matplotlib canvas)
# ---------------------------
class HUDWidget(QWidget):
    def __init__(self, parent=None, width=4, height=3, dpi=100):
        super().__init__(parent)
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # initial draw
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self._draw_horizon(self.roll, self.pitch)

    def _draw_horizon(self, roll, pitch):
        """
        Simple artificial horizon:
        - draws a circle bezel
        - draws a horizon line rotated by roll
        - pitch shifts the line up/down
        """
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-1.2, 1.2)
        self.ax.set_ylim(-1.2, 1.2)
        self.ax.axis('off')

        # bezel circle
        theta = np.linspace(0, 2 * np.pi, 200)
        x_c = np.cos(theta)
        y_c = np.sin(theta)
        self.ax.plot(x_c, y_c, linewidth=2, color='white', alpha=0.9)

        # compute horizon line (pitch moves up/down; roll rotates)
        pitch_norm = max(min(pitch / 45.0, 1.0), -1.0)
        base_y = -pitch_norm
        x_line = np.linspace(-2, 2, 10)
        y_line = np.ones_like(x_line) * base_y
        angle_rad = math.radians(roll)
        x_rot = x_line * math.cos(angle_rad) - y_line * math.sin(angle_rad)
        y_rot = x_line * math.sin(angle_rad) + y_line * math.cos(angle_rad)

        # sky and ground fill
        upper_poly_x = np.concatenate([x_rot, x_rot[::-1]])
        upper_poly_y = np.concatenate([y_rot, np.ones_like(y_rot) * 2])
        lower_poly_x = np.concatenate([x_rot, x_rot[::-1]])
        lower_poly_y = np.concatenate([y_rot, np.ones_like(y_rot) * -2])

        self.ax.fill(upper_poly_x, upper_poly_y, color='#3b82f6', alpha=0.45)
        self.ax.fill(lower_poly_x, lower_poly_y, color='#b45309', alpha=0.45)
        self.ax.plot(x_rot, y_rot, color='white', linewidth=2)

        self.ax.text(0, -0.95, f"Roll: {roll:.1f}°  Pitch: {pitch:.1f}°  Yaw: {self.yaw:.1f}°",
                     ha='center', va='center', color='white', fontsize=9)

        self.canvas.draw_idle()

    def update_hud(self, roll, pitch, yaw):
        """Public method to update the HUD graphics."""
        try:
            self.roll = roll
            self.pitch = pitch
            self.yaw = yaw
            self._draw_horizon(roll, pitch)
        except Exception:
            traceback.print_exc()


# ---------------------------
# Network Worker Thread (now fetches image bytes)
# ---------------------------
class TelemetryWorker(QThread):
    telemetry_received = Signal(dict)
    error = Signal(str)

    def __init__(self, telemetry_url="http://127.0.0.1:8000/telemetry", poll_interval=0.3, parent=None):
        super().__init__(parent)
        self.telemetry_url = telemetry_url
        self.poll_interval = poll_interval
        self._running = False

    def run(self):
        self._running = True
        while self._running:
            try:
                resp = requests.get(self.telemetry_url, timeout=1.0)
                if resp.status_code == 200:
                    data = resp.json()
                    if not isinstance(data, dict):
                        data = {"error": "bad telemetry format"}

                    # If telemetry contains a video URL, fetch the image bytes here (in worker thread)
                    video_url = data.get("video_frame_url")
                    if video_url:
                        try:
                            r_img = requests.get(video_url, timeout=1.0)
                            if r_img.status_code == 200:
                                data["video_frame_bytes"] = r_img.content
                            else:
                                # Attach None or skip; emit an info error
                                data["video_frame_bytes"] = None
                                self.error.emit(f"image http {r_img.status_code}")
                        except Exception as e_img:
                            data["video_frame_bytes"] = None
                            # Emit an error but keep telemetry flowing
                            self.error.emit(f"image fetch error: {e_img}")

                    self.telemetry_received.emit(data)
                else:
                    self.error.emit(f"telemetry http {resp.status_code}")
            except Exception as e:
                # emit an error string for UI to show in messages
                self.error.emit(str(e))

            # poll interval wait loop (non-blocking check)
            t0 = time.time()
            while self._running and (time.time() - t0) < self.poll_interval:
                time.sleep(0.01)

    def stop(self):
        self._running = False
        self.quit()
        self.wait(timeout=2000)


# ---------------------------
# Main Window
# ---------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("picoWorks - GCS (Desktop)")
        self.resize(1200, 720)
        self._connected = False
        self.worker = None

        # Apply dark stylesheet
        self.apply_stylesheet()

        # Central widget
        central = QWidget()
        main_v = QVBoxLayout(central)
        main_v.setContentsMargins(12, 12, 12, 12)
        main_v.setSpacing(12)

        # Top navigation
        nav_widget = QWidget()
        nav_layout = QHBoxLayout(nav_widget)
        nav_layout.setContentsMargins(6, 6, 6, 6)
        nav_layout.setSpacing(8)

        # Left nav buttons
        self.home_btn = QPushButton("HOME")
        self.help_btn = QPushButton("HELP")
        for b in (self.home_btn, self.help_btn):
            b.setCursor(Qt.PointingHandCursor)
            b.setFixedHeight(36)
            b.setStyleSheet("font-weight:600;")

        nav_layout.addWidget(self.home_btn)
        nav_layout.addWidget(self.help_btn)

        nav_layout.addStretch(1)

        # Right connect button
        self.connect_btn = QPushButton("CONNECT")
        self.connect_btn.setFixedHeight(36)
        self.update_connect_button_style()
        self.connect_btn.clicked.connect(self.toggle_connection)
        nav_layout.addWidget(self.connect_btn)

        main_v.addWidget(nav_widget)

        # Main split area
        split_widget = QWidget()
        split_layout = QHBoxLayout(split_widget)
        split_layout.setContentsMargins(0, 0, 0, 0)
        split_layout.setSpacing(12)

        # ---------- Left Column ----------
        left_col = QVBoxLayout()
        left_col.setSpacing(12)

        # Camera Group
        cam_group = QGroupBox("Camera")
        cam_group.setStyleSheet("QGroupBox { font-weight:700; }")
        cam_layout = QVBoxLayout()
        cam_layout.setContentsMargins(8, 8, 8, 8)
        cam_layout.setSpacing(6)
        self.detect_checkbox = QCheckBox("Detection")
        self.detect_checkbox.setCursor(Qt.PointingHandCursor)
        self.video_label = QLabel()
        # Removed fixed size to make it responsive
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet("background: #0f0f0f; border: 1px solid #333;")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("No Video")
        cam_layout.addWidget(self.detect_checkbox)
        cam_layout.addWidget(self.video_label, alignment=Qt.AlignCenter)
        cam_group.setLayout(cam_layout)
        left_col.addWidget(cam_group)

        # Messages Group
        msg_group = QGroupBox("Messages")
        msg_group.setStyleSheet("QGroupBox { font-weight:700; }")
        msg_layout = QVBoxLayout()
        msg_layout.setContentsMargins(8, 8, 8, 8)
        self.messages_edit = QTextEdit()
        self.messages_edit.setReadOnly(True)
        self.messages_edit.setFixedHeight(200)
        self.messages_edit.setStyleSheet("background: #121212;")
        msg_layout.addWidget(self.messages_edit)
        msg_group.setLayout(msg_layout)
        left_col.addWidget(msg_group, stretch=1)

        split_layout.addLayout(left_col, stretch=3)

        # ---------- Right Column ----------
        right_col = QVBoxLayout()
        right_col.setSpacing(12)

        # HUD Group
        hud_group = QGroupBox("HUD")
        hud_layout = QVBoxLayout()
        hud_layout.setContentsMargins(8, 8, 8, 8)
        self.hud_widget = HUDWidget(width=4, height=3, dpi=110)
        # Telemetry numeric label
        self.telemetry_label = QLabel("#Pitch: 0.0, Roll: 0.0, Yaw: 0.0, Battery: 0%")
        self.telemetry_label.setFont(QFont("Monospace", 10))
        self.telemetry_label.setStyleSheet("color: white;")
        hud_layout.addWidget(self.hud_widget)
        hud_layout.addWidget(self.telemetry_label, alignment=Qt.AlignCenter)
        hud_group.setLayout(hud_layout)
        right_col.addWidget(hud_group, stretch=2)

        # Controls Group (grid 2x2)
        ctrl_group = QGroupBox("Controls")
        ctrl_layout = QGridLayout()
        ctrl_layout.setSpacing(8)
        self.calibrate_btn = QPushButton("CALIBRATE")
        self.arm_btn = QPushButton("ARM")
        self.reset_btn = QPushButton("RESET")
        self.disarm_btn = QPushButton("DISARM")
        for btn in (self.calibrate_btn, self.arm_btn, self.reset_btn, self.disarm_btn):
            btn.setFixedHeight(48)
            btn.setCursor(Qt.PointingHandCursor)
        ctrl_layout.addWidget(self.calibrate_btn, 0, 0)
        ctrl_layout.addWidget(self.arm_btn, 0, 1)
        ctrl_layout.addWidget(self.reset_btn, 1, 0)
        ctrl_layout.addWidget(self.disarm_btn, 1, 1)
        ctrl_group.setLayout(ctrl_layout)
        right_col.addWidget(ctrl_group)

        split_layout.addLayout(right_col, stretch=2)

        main_v.addWidget(split_widget)

        self.setCentralWidget(central)

        # Connect control buttons to send_command
        self.calibrate_btn.clicked.connect(partial(self.send_command, "CALIBRATE"))
        self.arm_btn.clicked.connect(partial(self.send_command, "ARM"))
        self.reset_btn.clicked.connect(partial(self.send_command, "RESET"))
        self.disarm_btn.clicked.connect(partial(self.send_command, "DISARM"))
        self.home_btn.clicked.connect(self.on_home_clicked)
        self.help_btn.clicked.connect(self.on_help_clicked)

    def apply_stylesheet(self):
        style = """
        QMainWindow { background: #1a1a1a; color: white; }
        QGroupBox { background: #2b2b2b; color: white; border-radius: 6px; padding: 6px; }
        QPushButton { background: #2b2b2b; color: white; border: 1px solid #3a3a3a; border-radius: 6px; padding: 6px; }
        QPushButton:hover { border-color: #5a5a5a; }
        QTextEdit { color: white; }
        QLabel { color: white; }
        QCheckBox { color: white; }
        """
        self.setStyleSheet(style)

    def update_connect_button_style(self):
        if self._connected:
            self.connect_btn.setText("DISCONNECT")
            self.connect_btn.setStyleSheet("background: #8b1a1a; color: white; font-weight:700;")
        else:
            self.connect_btn.setText("CONNECT")
            self.connect_btn.setStyleSheet("background: #1a8b3a; color: white; font-weight:700;")

    def on_home_clicked(self):
        QMessageBox.information(self, "HOME", "Home pressed (TODO).")

    def on_help_clicked(self):
        QMessageBox.information(self, "HELP", "Help pressed (TODO).")

    def toggle_connection(self):
        if not self._connected:
            # call backend connect endpoint
            try:
                resp = requests.post("http://127.0.0.1:8000/connect", timeout=2.0)
                if resp.status_code == 200:
                    self._connected = True
                    self.start_worker()
                else:
                    QMessageBox.warning(self, "Connect Failed", f"Server returned {resp.status_code}")
                    return
            except Exception as e:
                QMessageBox.critical(self, "Connect Error", str(e))
                return
        else:
            # disconnect
            try:
                requests.post("http://127.0.0.1:8000/disconnect", timeout=2.0)
            except Exception:
                pass
            self._connected = False
            self.stop_worker()

        self.update_connect_button_style()

    def start_worker(self):
        if self.worker and self.worker.isRunning():
            return
        self.worker = TelemetryWorker(telemetry_url="http://127.0.0.1:8000/telemetry", poll_interval=0.33)
        self.worker.telemetry_received.connect(self.update_telemetry_ui)
        self.worker.error.connect(self.on_worker_error)
        self.worker.start()

    def stop_worker(self):
        if self.worker:
            try:
                self.worker.stop()
            except Exception:
                pass
            self.worker = None

    @Slot(dict)
    def update_telemetry_ui(self, data: dict):
        """
        Receives telemetry dict from worker and updates HUD, labels, messages and camera.
        Now expects optional 'video_frame_bytes' in the dict (provided by worker).
        """
        # If error packet
        if "error" in data:
            self.prepend_message(f"{time.strftime('%H:%M:%S')} [ERROR] {data.get('error')}")
            return

        # Parse numeric telemetry safely
        roll = float(data.get("roll", 0.0))
        pitch = float(data.get("pitch", 0.0))
        yaw = float(data.get("yaw", 0.0))
        battery = data.get("battery", None)
        altitude = data.get("altitude", None)
        signal = data.get("signal", None)
        cpu_temp = data.get("cpu_temp", None)
        timestamp = data.get("timestamp", "")
        # update HUD
        try:
            self.hud_widget.update_hud(roll, pitch, yaw)
        except Exception:
            traceback.print_exc()

        # update telemetry label
        batt_text = f"{battery:.1f}%" if isinstance(battery, (float, int)) else str(battery)
        self.telemetry_label.setText(f"#Pitch: {pitch:.1f}°, Roll: {roll:.1f}°, Yaw: {yaw:.1f}°  |  Battery: {batt_text}  | Alt: {altitude}m")

        # handle messages: list of strings
        msgs = data.get("messages", [])
        if isinstance(msgs, list) and msgs:
            for m in msgs:
                self.prepend_message(m)

        # handle video frame bytes (non-blocking UI)
        video_bytes = data.get("video_frame_bytes", None)
        if video_bytes:
            try:
                pix = QPixmap()
                pix.loadFromData(video_bytes)
                # scale to the current label size but allow expansion
                scaled = pix.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.video_label.setPixmap(scaled)
            except Exception:
                # keep previous pixmap or "No Video"
                traceback.print_exc()

    @Slot(str)
    def on_worker_error(self, message: str):
        # Show once per error (don't spam)
        self.prepend_message(f"{time.strftime('%H:%M:%S')} [NET-ERR] {message}")

    def prepend_message(self, msg: str):
        """
        Prepend message to the top of the messages box.
        """
        existing = self.messages_edit.toPlainText()
        if existing:
            new_text = msg + "\n" + existing
        else:
            new_text = msg
        self.messages_edit.setPlainText(new_text)

    def send_command(self, command: str):
        """
        Sends a POST to /command with { "command": "<COMMAND>" }
        """
        try:
            resp = requests.post("http://127.0.0.1:8000/command", json={"command": command}, timeout=2.0)
            if resp.status_code != 200:
                self.prepend_message(f"{time.strftime('%H:%M:%S')} [CMD-ERR] {command} -> HTTP {resp.status_code}")
            else:
                try:
                    data = resp.json()
                    self.prepend_message(f"{time.strftime('%H:%M:%S')} [CMD] {command} -> {data}")
                except Exception:
                    self.prepend_message(f"{time.strftime('%H:%M:%S')} [CMD] {command} -> OK")
        except Exception as e:
            self.prepend_message(f"{time.strftime('%H:%M:%S')} [CMD-ERR] {command} -> {e}")

    def closeEvent(self, event):
        # ensure worker stopped
        try:
            self.stop_worker()
        except Exception:
            pass
        event.accept()


# ---------------------------
# App Entrypoint
# ---------------------------
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
