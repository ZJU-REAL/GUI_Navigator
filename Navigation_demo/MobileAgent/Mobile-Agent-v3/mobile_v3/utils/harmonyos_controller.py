import os
import time
import subprocess
from .controller import Controller

class HarmonyOSController(Controller):
    def __init__(self, hdc_path):
        self.hdc_path = hdc_path
        self._recording_process = None
        self._recording_output = None

    def get_screenshot(self, save_path):
        command = self.hdc_path + " shell rm /data/local/tmp/screenshot.png"
        subprocess.run(command, capture_output=True, text=True, shell=True)
        time.sleep(0.5)
        command = self.hdc_path + " shell uitest screenCap -p /data/local/tmp/screenshot.png"
        subprocess.run(command, capture_output=True, text=True, shell=True)
        time.sleep(0.5)
        command = self.hdc_path + " file recv /data/local/tmp/screenshot.png " + save_path
        subprocess.run(command, capture_output=True, text=True, shell=True)
        time.sleep(0.5)

        if not os.path.exists(save_path):
            return False
        else:
            return True

    def tap(self, x, y):
        command = self.hdc_path + f" shell uitest uiInput click {x} {y}"
        subprocess.run(command, capture_output=True, text=True, shell=True)

    def type(self, text):
        text = text.replace("\\n", "_").replace("\n", "_")
        for char in text:
            if char == ' ':
                command = self.adb_path + f" shell uitest uiInput keyEvent 2050"
                subprocess.run(command, capture_output=True, text=True, shell=True)
            elif char == '_':
                command = self.hdc_path + f" shell uitest uiInput keyEvent 2054"
                subprocess.run(command, capture_output=True, text=True, shell=True)
            elif 'a' <= char <= 'z' or 'A' <= char <= 'Z' or char.isdigit():
                command = self.hdc_path + f" shell uitest uiInput inputText 1 1 {char}"
                subprocess.run(command, capture_output=True, text=True, shell=True)
            elif char in '-.,!?@\'°/:;()':
                command = self.hdc_path + f" shell uitest uiInput inputText 1 1 \"{char}\""
                subprocess.run(command, capture_output=True, text=True, shell=True)
            else:
                command = self.hdc_path + f" shell uitest uiInput inputText 1 1 {char}"
                subprocess.run(command, capture_output=True, text=True, shell=True)

    def slide(self, x1, y1, x2, y2):
        command = self.hdc_path + f" shell uitest uiInput swipe {x1} {y1} {x2} {y2} 500"
        subprocess.run(command, capture_output=True, text=True, shell=True)

    def back(self):
        command = self.hdc_path + " shell uitest uiInput keyEvent Back"
        subprocess.run(command, capture_output=True, text=True, shell=True)

    def home(self):
        command = self.hdc_path + " shell uitest uiInput keyEvent Home"
        subprocess.run(command, capture_output=True, text=True, shell=True)

    def start_recording(self, output_path):
        if not output_path:
            raise ValueError("output_path must be provided for recording.")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        command = self.hdc_path + " shell rm /data/local/tmp/screenrecord.mp4"
        subprocess.run(command, capture_output=True, text=True, shell=True)
        command = self.hdc_path + " shell screenrecord /data/local/tmp/screenrecord.mp4"
        self._recording_output = output_path
        self._recording_process = subprocess.Popen(command, shell=True)

    def stop_recording(self):
        if not self._recording_process:
            return
        stop_command = self.hdc_path + " shell pkill -SIGINT screenrecord"
        subprocess.run(stop_command, capture_output=True, text=True, shell=True)
        try:
            self._recording_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._recording_process.kill()
        pull_command = f"{self.hdc_path} file recv /data/local/tmp/screenrecord.mp4 {self._recording_output}"
        subprocess.run(pull_command, capture_output=True, text=True, shell=True)
        self._recording_process = None
        self._recording_output = None
