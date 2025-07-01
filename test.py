import cv2
import numpy as np
import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QLineEdit, QPushButton, QLabel)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PyQt5.QtGui import QFont  # Make sure this import is present
from PySide6.QtGui import QFont


# --- Global Variables ---
current_filter_command = None
video_running = True

# --- Filter Functions ---
def apply_grayscale_filter(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

def apply_invert_filter(image):
    return cv2.bitwise_not(image)

def apply_sepia_filter(image):
    kernel = np.array([[0.272, 0.534, 0.189],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia_image = cv2.transform(image, kernel)
    return np.clip(sepia_image, 0, 255).astype(np.uint8)

# --- Video Processing Worker Thread ---
class VideoWorker(QThread):
    # Signal to emit the processed frame to the GUI
    change_pixmap_signal = Signal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.active_filter_func = None # Stores the actual filter function

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame. Exiting thread...")
                break

            # --- Process Command from GUI ---
            # Access global current_filter_command and update active_filter_func
            global current_filter_command
            if current_filter_command:
                if "grayscale" in current_filter_command.lower():
                    self.active_filter_func = apply_grayscale_filter
                elif "invert" in current_filter_command.lower():
                    self.active_filter_func = apply_invert_filter
                elif "sepia" in current_filter_command.lower():
                    self.active_filter_func = apply_sepia_filter
                elif "original" in current_filter_command.lower() or "normal" in current_filter_command.lower():
                    self.active_filter_func = None # Reset to no filter
                
                current_filter_command = None # Reset command after processing

            # --- Apply Active Filter ---
            display_frame_right = frame.copy()
            if self.active_filter_func:
                display_frame_right = self.active_filter_func(frame.copy())

            # --- Prepare Frame for GUI ---
            # Convert frame to RGB (OpenCV uses BGR by default)
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_image_right = cv2.cvtColor(display_frame_right, cv2.COLOR_BGR2RGB)

            # Resize if necessary for display consistency (optional, but good practice)
            # For simplicity, we'll assume original frame is the reference size
            h_left, w_left, _ = rgb_image.shape
            h_right, w_right, _ = rgb_image_right.shape

            if h_left != h_right:
                scale_percent = h_left / float(h_right)
                new_w_right = int(w_right * scale_percent)
                rgb_image_right = cv2.resize(rgb_image_right, (new_w_right, h_left), interpolation=cv2.INTER_AREA)
            
            # Concatenate horizontally
            combined_frame_rgb = np.hstack((rgb_image, rgb_image_right))

            # Emit the processed frame
            self.change_pixmap_signal.emit(combined_frame_rgb)

        cap.release()
        print("Video thread finished.")

    def stop(self):
        self._run_flag = False
        self.wait() # Wait for the thread to finish

# --- PySide6 GUI Window ---
class VideoFilterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Video Filters")
        self.setGeometry(100, 100, 1200, 600) # x, y, width, height

        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout(central_widget)

        # Label to display the video feed
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.video_label)

        # Chat Input Area
        self.chat_input_layout = QVBoxLayout()
        self.layout.addLayout(self.chat_input_layout)

        self.instruction_label = QLabel("Enter filter command (grayscale, invert, sepia, original):")
        self.instruction_label.setFont(QFont("Arial", 12))

        self.chat_input_layout.addWidget(self.instruction_label)

        self.chat_entry = QLineEdit()
        self.chat_entry.setFont(QFont("Arial", 11))
        self.chat_entry.setPlaceholderText("Type filter command here...")
        self.chat_entry.returnPressed.connect(self.process_chat_input) # Connect Enter key
        self.chat_input_layout.addWidget(self.chat_entry)

        self.send_button = QPushButton("Send Filter")
        self.send_button.setFont(QFont("Arial", 11))
        self.send_button.clicked.connect(self.process_chat_input)
        self.chat_input_layout.addWidget(self.send_button)

        # Start the video worker thread
        self.thread = VideoWorker()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    def process_chat_input(self):
        """Processes the text from the chat entry."""
        global current_filter_command
        command = self.chat_entry.text().strip().lower()
        if command:
            current_filter_command = command
            self.chat_entry.clear() # Clear the input field

    def update_image(self, cv_img):
        """ Updates the QLabel with a new processed frame. """
        qt_image = QImage(cv_img.data, cv_img.shape[1], cv_img.shape[0], cv_img.shape[1] * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap)

    def closeEvent(self, event):
        """Handles the window closing event."""
        global video_running
        video_running = False # Signal the video thread to stop
        self.thread.stop()    # Explicitly stop the thread
        event.accept()        # Accept the close event

# --- Main Execution ---
if __name__ == "__main__":
    app = QApplication(sys.argv) # Initialize the PySide6 application

    main_window = VideoFilterApp()
    main_window.show()

    # We don't need to manually start the thread here if it's in the class __init__
    # The PySide6 event loop will manage thread execution.

    # The video_thread.start() is in the VideoFilterApp constructor.

    sys.exit(app.exec()) # Start the PySide6 event loop