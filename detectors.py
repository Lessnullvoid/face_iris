"""
This module contains detector classes for eye tracking and analysis.
Each detector is responsible for a specific aspect of eye tracking.
"""

from collections import deque
import time
import numpy as np

class BlinkDetector:
    """Detects and tracks eye blinks."""
    
    def __init__(self, ear_threshold=0.2, blink_time_threshold=0.5):
        """
        Initialize the blink detector.
        
        Args:
            ear_threshold (float): Threshold for eye aspect ratio to detect blink
            blink_time_threshold (float): Maximum time for a blink in seconds
        """
        self.ear_threshold = ear_threshold
        self.blink_time_threshold = blink_time_threshold
        self.ear_history = deque(maxlen=3)
        self.blink_start = None
        self.blink_counter = 0
        self.current_blink_frame_count = 0
        self.is_eye_closed = False
        
    def detect(self, ear):
        """
        Detect blinks based on eye aspect ratio.
        
        Args:
            ear (float): Current eye aspect ratio
            
        Returns:
            dict: Dictionary containing blink detection results
        """
        self.ear_history.append(ear)
        
        # Detect eye closure
        if ear < self.ear_threshold and not self.is_eye_closed:
            self.is_eye_closed = True
            self.blink_start = time.time()
            self.current_blink_frame_count = 0
        
        # Detect eye opening
        elif ear >= self.ear_threshold and self.is_eye_closed:
            self.is_eye_closed = False
            if self.blink_start is not None:
                blink_duration = time.time() - self.blink_start
                if blink_duration < self.blink_time_threshold:
                    self.blink_counter += 1
                self.blink_start = None
        
        # Count frames while eye is closed
        if self.is_eye_closed:
            self.current_blink_frame_count += 1
        
        return {
            'is_blinking': self.is_eye_closed,
            'blink_count': self.blink_counter,
            'current_blink_frames': self.current_blink_frame_count,
            'ear_history': list(self.ear_history)
        }

class PupilTracker:
    """Tracks pupil size changes and detects significant variations."""
    
    def __init__(self, history_size=30, change_threshold=0.15):
        """
        Initialize the pupil tracker.
        
        Args:
            history_size (int): Number of frames to keep in history
            change_threshold (float): Threshold for significant pupil size change
        """
        self.diameter_history = deque(maxlen=history_size)
        self.baseline = None
        self.change_threshold = change_threshold
        
    def detect_change(self, current_diameter):
        """
        Detect changes in pupil diameter.
        
        Args:
            current_diameter (float): Current pupil diameter
            
        Returns:
            dict: Dictionary containing pupil change detection results
        """
        self.diameter_history.append(current_diameter)
        
        # Initialize baseline with the average of first 10 measurements
        if len(self.diameter_history) >= 10 and self.baseline is None:
            self.baseline = sum(list(self.diameter_history)[:10]) / 10
            
        if self.baseline is None:
            return {
                'is_changing': False,
                'change_percentage': 0,
                'current_diameter': current_diameter,
                'baseline_diameter': None
            }
            
        # Calculate current change
        change_percentage = (current_diameter - self.baseline) / self.baseline
        is_changing = abs(change_percentage) > self.change_threshold
        
        # Update baseline slowly if change is small
        if abs(change_percentage) < self.change_threshold / 2:
            self.baseline = 0.95 * self.baseline + 0.05 * current_diameter
            
        return {
            'is_changing': is_changing,
            'change_percentage': change_percentage * 100,
            'current_diameter': current_diameter,
            'baseline_diameter': self.baseline
        }

class EmotionDetector:
    """Detects emotions based on eye metrics."""
    
    def __init__(self):
        """Initialize the emotion detector with default parameters."""
        self.blink_rate_history = deque(maxlen=90)  # 3 seconds at 30fps
        self.pupil_size_history = deque(maxlen=30)
        self.eye_openness_history = deque(maxlen=30)
        self.last_blink_time = time.time()
        self.emotion_scores = {
            'surprise': 0.0,
            'focus': 0.0,
            'tired': 0.0,
            'relaxed': 0.0,
            'stressed': 0.0
        }
        
    def update(self, left_eye, right_eye, left_pupil, right_pupil, left_blink, right_blink):
        """
        Update emotion detection based on eye metrics.
        
        Args:
            left_eye (dict): Left eye parameters
            right_eye (dict): Right eye parameters
            left_pupil (dict): Left pupil parameters
            right_pupil (dict): Right pupil parameters
            left_blink (dict): Left eye blink data
            right_blink (dict): Right eye blink data
            
        Returns:
            dict: Dictionary containing emotion detection results
        """
        current_time = time.time()
        
        # Update histories
        avg_pupil_size = (left_eye['iris_diameter'] + right_eye['iris_diameter']) / 2
        avg_eye_openness = (left_eye['eye_height'] + right_eye['eye_height']) / 2
        self.pupil_size_history.append(avg_pupil_size)
        self.eye_openness_history.append(avg_eye_openness)
        
        # Update blink rate
        if left_blink['is_blinking'] or right_blink['is_blinking']:
            if current_time - self.last_blink_time > 0.1:  # Prevent double counting
                self.blink_rate_history.append(1)
                self.last_blink_time = current_time
        
        # Calculate metrics
        blink_rate = len(self.blink_rate_history) * (60 / 3)  # Convert to blinks per minute
        pupil_change = 0
        if len(self.pupil_size_history) > 1:
            pupil_change = (self.pupil_size_history[-1] - self.pupil_size_history[0]) / self.pupil_size_history[0]
        
        eye_openness_ratio = avg_eye_openness / np.mean(list(self.eye_openness_history)) if self.eye_openness_history else 1.0
        
        # Update emotion scores
        self._update_emotion_scores(eye_openness_ratio, pupil_change, blink_rate)
        
        # Get dominant emotion
        dominant_emotion = max(self.emotion_scores.items(), key=lambda x: x[1])
        
        return {
            'dominant_emotion': dominant_emotion[0],
            'emotion_scores': self.emotion_scores,
            'blink_rate': blink_rate,
            'pupil_change': pupil_change * 100,
            'eye_openness_ratio': eye_openness_ratio
        }
    
    def _update_emotion_scores(self, eye_openness_ratio, pupil_change, blink_rate):
        """
        Update emotion scores based on eye metrics.
        
        Args:
            eye_openness_ratio (float): Ratio of current eye openness to average
            pupil_change (float): Change in pupil size
            blink_rate (float): Current blink rate
        """
        self.emotion_scores['surprise'] = self._calculate_surprise_score(eye_openness_ratio, pupil_change)
        self.emotion_scores['focus'] = self._calculate_focus_score(pupil_change, blink_rate)
        self.emotion_scores['tired'] = self._calculate_tired_score(blink_rate, eye_openness_ratio)
        self.emotion_scores['relaxed'] = self._calculate_relaxed_score(pupil_change, blink_rate)
        self.emotion_scores['stressed'] = self._calculate_stressed_score(pupil_change, blink_rate)
    
    def _calculate_surprise_score(self, eye_openness_ratio, pupil_change):
        score = 0.0
        if eye_openness_ratio > 1.2:  # Eyes wide open
            score += 0.5
        if pupil_change > 0.15:  # Pupils dilated
            score += 0.5
        return min(1.0, score)
    
    def _calculate_focus_score(self, pupil_change, blink_rate):
        score = 0.0
        if abs(pupil_change) < 0.1:  # Stable pupil size
            score += 0.3
        if 10 <= blink_rate <= 15:  # Normal blink rate
            score += 0.4
        if blink_rate < 10:  # Reduced blink rate (intense focus)
            score += 0.3
        return min(1.0, score)
    
    def _calculate_tired_score(self, blink_rate, eye_openness_ratio):
        score = 0.0
        if blink_rate > 20:  # Increased blink rate
            score += 0.4
        if eye_openness_ratio < 0.8:  # Eyes not fully open
            score += 0.6
        return min(1.0, score)
    
    def _calculate_relaxed_score(self, pupil_change, blink_rate):
        score = 0.0
        if abs(pupil_change) < 0.05:  # Very stable pupil size
            score += 0.5
        if 15 <= blink_rate <= 20:  # Slightly elevated blink rate
            score += 0.5
        return min(1.0, score)
    
    def _calculate_stressed_score(self, pupil_change, blink_rate):
        score = 0.0
        if abs(pupil_change) > 0.1:  # Fluctuating pupil size
            score += 0.3
        if blink_rate > 20:  # High blink rate
            score += 0.3
        if blink_rate < 5:  # Very low blink rate (frozen stare)
            score += 0.4
        return min(1.0, score) 