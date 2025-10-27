import sys
import numpy as np
import random
from datetime import datetime
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QGroupBox, QTextEdit, QFrame, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer, Slot
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.transforms as transforms
import matplotlib.pyplot as plt

# ---------------------------------
# CLASS 1: UPGRADED HUD WIDGET
# This is a high-performance HUD that is much faster and looks better.
# ---------------------------------
class HUDWidget(QWidget):
    def __init__(self):
        super().__init__()
        
        # Remove figsize and set transparent facecolor for the figure
        self.figure = Figure(facecolor='#1a1a1a') # Changed to '#1a1a1a'
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        self.init_hud()

    def init_hud(self):
        """Create all Matplotlib artists once."""
        # Removed self.ax.set_aspect('equal')
        # Removed self.ax.set_xlim(-30, 30)
        # Removed self.ax.set_ylim(-30, 30)
        self.ax.axis('off')
        # Set axes facecolor to transparent
        self.ax.set_facecolor('#1a1a1a') # Changed to '#1a1a1a'

        # Create sky and ground patches
        # Use large values for coordinates to always fill the view
        # The key is that these shapes are much larger than any expected view
        self.sky = plt.Rectangle((-1000, 0), 2000, 1000, color='#3079a8', zorder=0)
        self.ground = plt.Rectangle((-1000, -1000), 2000, 1000, color='#6b8c5a', zorder=0)
        self.ax.add_patch(self.sky)
        self.ax.add_patch(self.ground)
        
        # Create horizon line
        self.horizon_line = plt.Line2D([-1000, 1000], [0, 0], color='white', lw=2, zorder=1)
        self.ax.add_line(self.horizon_line)

        # Create static aircraft symbol
        # The coordinates here are relative to the *center of the view*, not the world.
        # So these should remain small and centered.
        self.aircraft_symbol_lines = [ # Renamed for clarity
            plt.Line2D([-5, 0, 5], [-5, 0, -5], color='red', lw=2, zorder=10), # The 'V'
            plt.Line2D([-15, -7], [0, 0], color='red', lw=2, zorder=10),    # Left wing
            plt.Line2D([7, 15], [0, 0], color='red', lw=2, zorder=10)      # Right wing
        ]
        for line in self.aircraft_symbol_lines: # Use new name
            self.ax.add_line(line)

    def update_hud(self, roll, pitch, yaw):
        """Update the transforms of existing artists (fast)."""
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        
        # We need to set the view limits dynamically or based on desired pitch range
        # Let's say we want to show +/- 30 degrees vertically for pitch display
        # and proportional horizontal view.
        view_height = 60 # total vertical view range
        view_width = view_height * (self.width() / self.height()) # Adjust width based on actual widget aspect ratio

        self.ax.set_xlim(-view_width/2, view_width/2)
        self.ax.set_ylim(-view_height/2, view_height/2)
        
        # Normalize pitch for translation. We'll show +/- 30 degrees max.
        pitch_display_range = 30.0 # How many degrees of pitch to map to the full viewport
        pitch_norm = max(min(pitch, pitch_display_range), -pitch_display_range)

        # Create the transformation
        # We rotate by -roll and translate by -pitch (so world moves)
        transform = transforms.Affine2D().rotate_deg(-roll).translate(0, -pitch_norm) + self.ax.transData
        
        # Apply transform to dynamic elements
        self.sky.set_transform(transform)
        self.ground.set_transform(transform)
        self.horizon_line.set_transform(transform)

        self.canvas.draw_idle()


# ---------------------------------
# CLASS 2: MESSAGES WIDGET
# (No changes needed, it's well-designed)
# ---------------------------------
class MessagesWidget(QGroupBox):
    def __init__(self):
        super().__init__("Messages")
        layout = QVBoxLayout()
        self.text_box = QTextEdit()
        self.text_box.setReadOnly(True)
        layout.addWidget(self.text_box)
        self.setLayout(layout)

    def add_message(self, msg):
        self.text_box.append(msg)
        self.text_box.verticalScrollBar().setValue(self.text_box.verticalScrollBar().maximum())


# ---------------------------------
# CLASS 3: COMMANDS WIDGET
# (No changes needed, it's well-designed)
# ---------------------------------
class CommandsWidget(QGroupBox):
    def __init__(self):
        super().__init__("Controls")
        layout = QGridLayout()
        self.buttons = {}
        names = ["CALIBRATE", "ARM", "RESET", "DISARM"]
        for i, name in enumerate(names):
            btn = QPushButton(name)
            # Removed setFixedHeight for responsiveness
            btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            btn.setMaximumHeight(35)
            btn.setMaximumWidth(150)
            layout.addWidget(btn, i // 2, i % 2)
            self.buttons[name] = btn
        self.setLayout(layout)

# ---------------------------------
# NEW CLASS: TELEMETRY WIDGET
# A dedicated, clean widget for numeric data
# ---------------------------------
class TelemetryWidget(QGroupBox):
    def __init__(self):
        super().__init__("Telemetry")
        self.value_labels = {}
        layout = QGridLayout()

        # Create labels
        metrics = ["Roll", "Pitch", "Yaw", "Battery", "Altitude", "Signal"]
        for i, name in enumerate(metrics):
            name_label = QLabel(name)
            value_label = QLabel("0.0")
            layout.addWidget(name_label, i // 2, i % 2, Qt.AlignLeft)
            layout.addWidget(value_label, i // 2, i % 2, Qt.AlignRight)
            self.value_labels[name] = value_label
        
        self.setLayout(layout)

    def update_data(self, data):
        """Updates all telemetry labels from a data dictionary."""
        self.value_labels["Roll"].setText(f"{data.get('roll', 0.0):.1f}°")
        self.value_labels["Pitch"].setText(f"{data.get('pitch', 0.0):.1f}°")
        self.value_labels["Yaw"].setText(f"{data.get('yaw', 0.0):.1f}°")
        self.value_labels["Battery"].setText(f"{data.get('battery', 0.0):.1f}%")
        self.value_labels["Altitude"].setText(f"{data.get('altitude', 0.0):.1f} m")
        self.value_labels["Signal"].setText(f"{data.get('signal', 0)}%")


# ---------------------------------
# CLASS 4: MAIN WINDOW (Refactored)
# ---------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("picoWorks - GCS (Desktop)")
        self.resize(1400, 800) # Use resize for initial, not fixed
        
        self.is_connected = False
        
        # --- Create Fake Telemetry Timer ---
        self.dev_timer = QTimer(self)
        self.dev_timer.setInterval(100) # 100ms = 10Hz update rate
        self.dev_timer.timeout.connect(self._update_fake_telemetry)

        self.init_ui()
        self.apply_stylesheet()

    def init_ui(self):
        """Create and assemble all UI components."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # ----- Top bar -----
        top_bar = QHBoxLayout()
        btn_home = QPushButton("HOME")
        btn_help = QPushButton("HELP")
        self.btn_connect = QPushButton("CONNECT")
        
        top_bar.addWidget(btn_home)
        top_bar.addWidget(btn_help)
        top_bar.addStretch()
        top_bar.addWidget(self.btn_connect)
        
        # Connect signals
        self.btn_connect.clicked.connect(self.toggle_connection)

        main_layout.addLayout(top_bar)

        # ----- Center layout (camera + right column) -----
        center_layout = QHBoxLayout()

        # Left side (Camera + Messages)
        left_col = QVBoxLayout()
        camera_box = QGroupBox("Camera")
        camera_layout = QVBoxLayout()
        self.camera_placeholder = QLabel("Camera Feed Placeholder")
        self.camera_placeholder.setAlignment(Qt.AlignCenter)
        self.camera_placeholder.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) # Makes it responsive
        camera_layout.addWidget(self.camera_placeholder)
        camera_box.setLayout(camera_layout)
        left_col.addWidget(camera_box, 3) # 3 stretch factor

        self.messages = MessagesWidget()
        left_col.addWidget(self.messages, 1) # 1 stretch factor

        # Right side (HUD + Telemetry + Controls)
        right_col = QVBoxLayout()
        self.hud = HUDWidget()
        self.hud.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding) # Makes it responsive
        hud_box = QGroupBox("HUD")
        hud_layout = QVBoxLayout()
        hud_layout.addWidget(self.hud)
        hud_box.setLayout(hud_layout)

        self.telemetry = TelemetryWidget() # Use new telemetry widget
        self.commands = CommandsWidget()

        right_col.addWidget(hud_box, 2) # 2 stretch factor
        right_col.addWidget(self.telemetry, 1) # 1 stretch factor
        right_col.addWidget(self.commands, 1) # 1 stretch factor

        center_layout.addLayout(left_col, 2) # Left col is 2/3 of width
        center_layout.addLayout(right_col, 1) # Right col is 1/3 of width

        main_layout.addLayout(center_layout)

    def apply_stylesheet(self):
        """A single, clean stylesheet for the whole app."""
        self.setStyleSheet("""
            QMainWindow, QWidget { 
                background-color: #1a1a1a; 
            }
            QGroupBox {
                background-color: #2b2b2b;
                color: white;
                border: 1px solid #3a3a3a;
                border-radius: 8px;
                margin-top: 10px;
                font-size: 14px;
                font-weight: bold;
                padding: 5px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: -5px 10px 0 10px;
            }
            QLabel { 
                color: #e0e0e0; 
                font-size: 13px;
            }
            QTextEdit {
                background-color: #1a1a1a;
                color: #00ff00; /* Green console text */
                border: 1px solid #3a3a3a;
                border-radius: 5px;
                font-family: 'Courier New', monospace;
            }
            QPushButton {
                background-color: #444;
                color: white;
                border: 1px solid #555;
                border-radius: 6px;
                padding: 8px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #555;
            }
            QPushButton:pressed {
                background-color: #333;
            }
            /* Special style for CONNECT button */
            #ConnectButton {
                background-color: #2ECC71; /* Green */
            }
            #ConnectButton:hover {
                background-color: #27ae60;
            }
            /* Special style for DISCONNECT button */
            #DisconnectButton {
                background-color: #E74C3C; /* Red */
            }
            #DisconnectButton:hover {
                background-color: #c0392b;
            }
        """)
        # Apply object names for special styling
        self.btn_connect.setObjectName("ConnectButton")
        
    @Slot()
    def toggle_connection(self):
        """Starts or stops the fake telemetry timer."""
        if not self.is_connected:
            self.dev_timer.start()
            self.is_connected = True
            self.btn_connect.setText("DISCONNECT")
            self.btn_connect.setObjectName("DisconnectButton")
            self.messages.add_message(f"[{datetime.now().strftime('%H:%M:%S')}] [INFO] Fake telemetry stream started.")
        else:
            self.dev_timer.stop()
            self.is_connected = False
            self.btn_connect.setText("CONNECT")
            self.btn_connect.setObjectName("ConnectButton")
            self.messages.add_message(f"[{datetime.now().strftime('%H:%M:%S')}] [INFO] Fake telemetry stream stopped.")
        
        # Re-apply stylesheet to update button color
        self.apply_stylesheet()
    
    @Slot()
    def _update_fake_telemetry(self):
        """Generates fake data and updates all UI widgets."""
        # 1. Generate Fake Data
        data = {
            "roll": random.uniform(-25.0, 25.0),
            "pitch": random.uniform(-15.0, 15.0),
            "yaw": random.uniform(0, 360),
            "battery": max(0, 99.0 - (random.random() * 5)),
            "signal": random.randint(70, 100),
            "altitude": random.uniform(0.0, 50.0)
        }
        
        # 2. Update HUD
        self.hud.update_hud(data['roll'], data['pitch'], data['yaw'])
        
        # 3. Update Telemetry Panel
        self.telemetry.update_data(data)

    def closeEvent(self, event):
        """Ensure the timer stops when the app closes."""
        self.dev_timer.stop()
        event.accept()

# ---------------------------------
# MAIN EXECUTION
# ---------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())