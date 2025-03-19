import numpy as np
from collections import deque
import time

class PupillometryAnalyzer:
    def __init__(self):
        self.recording = False
        self.start_time = None
        self.baseline_duration = 3.0  # seconds for baseline collection
        
        # Storage for measurements
        self.left_pupil_data = []
        self.right_pupil_data = []
        self.timestamps = []
        
        # Baseline values
        self.baseline_left = None
        self.baseline_right = None
        
        # Moving windows for real-time analysis
        self.window_size = 90  # 3 seconds at 30fps
        self.left_diameter_window = deque(maxlen=self.window_size)
        self.right_diameter_window = deque(maxlen=self.window_size)
        self.velocity_window = deque(maxlen=self.window_size)
        
    def start_recording(self):
        """Start pupillometry recording session"""
        self.recording = True
        self.start_time = time.time()
        self.clear_data()
        print("Pupillometry recording started")
        
    def stop_recording(self):
        """Stop pupillometry recording session"""
        self.recording = False
        print("Pupillometry recording stopped")
        
    def clear_data(self):
        """Clear all stored data"""
        self.left_pupil_data = []
        self.right_pupil_data = []
        self.timestamps = []
        self.baseline_left = None
        self.baseline_right = None
        self.left_diameter_window.clear()
        self.right_diameter_window.clear()
        self.velocity_window.clear()
        
    def update(self, left_diameter, right_diameter):
        """Update with new pupil measurements"""
        if not self.recording:
            return None
            
        current_time = time.time()
        relative_time = current_time - self.start_time
        
        # Store raw data
        self.left_pupil_data.append(left_diameter)
        self.right_pupil_data.append(right_diameter)
        self.timestamps.append(relative_time)
        
        # Update moving windows
        self.left_diameter_window.append(left_diameter)
        self.right_diameter_window.append(right_diameter)
        
        # Calculate baseline if needed
        if relative_time <= self.baseline_duration:
            return {
                'status': 'baseline_collection',
                'progress': (relative_time / self.baseline_duration) * 100,
                'time': relative_time
            }
        elif self.baseline_left is None:
            self.baseline_left = np.mean(self.left_pupil_data)
            self.baseline_right = np.mean(self.right_pupil_data)
        
        # Calculate metrics
        metrics = self._calculate_metrics(left_diameter, right_diameter, relative_time)
        metrics['status'] = 'recording'
        metrics['time'] = relative_time
        
        return metrics
    
    def _calculate_metrics(self, left_diameter, right_diameter, current_time):
        """Calculate pupillometry metrics"""
        # Basic metrics
        left_change = ((left_diameter - self.baseline_left) / self.baseline_left) * 100
        right_change = ((right_diameter - self.baseline_right) / self.baseline_right) * 100
        
        # Variability
        left_variability = np.std(list(self.left_diameter_window))
        right_variability = np.std(list(self.right_diameter_window))
        
        # Calculate velocity (last 100ms)
        if len(self.left_pupil_data) > 3:
            left_velocity = (left_diameter - self.left_pupil_data[-2]) / 0.033  # Assuming 30fps
            right_velocity = (right_diameter - self.right_pupil_data[-2]) / 0.033
        else:
            left_velocity = right_velocity = 0
            
        self.velocity_window.append((left_velocity + right_velocity) / 2)
        
        # Power spectrum (if enough data)
        if len(self.left_diameter_window) >= self.window_size:
            left_power = np.abs(np.fft.fft(list(self.left_diameter_window)))[1:5].mean()
            right_power = np.abs(np.fft.fft(list(self.right_diameter_window)))[1:5].mean()
        else:
            left_power = right_power = 0
            
        return {
            'left_diameter': left_diameter,
            'right_diameter': right_diameter,
            'left_change': left_change,
            'right_change': right_change,
            'left_variability': left_variability,
            'right_variability': right_variability,
            'left_velocity': left_velocity,
            'right_velocity': right_velocity,
            'left_power': left_power,
            'right_power': right_power,
            'cognitive_load_index': (abs(left_change) + abs(right_change)) / 2
        } 