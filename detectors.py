"""
Eye tracking detection classes for blink, pupil, and emotion detection.
"""

import numpy as np
from collections import deque
import time
import logging

class BlinkDetector:
    """Detects blinks using Eye Aspect Ratio (EAR) analysis."""
    
    def __init__(self):
        """Initialize blink detector with default parameters."""
        self.ear_history = deque(maxlen=3)
        self.blink_start = None
        self.blink_counter = 0
        self.current_blink_frame_count = 0
        self.is_eye_closed = False
        
    def detect(self, ear):
        """
        Detect blink state based on EAR value.
        
        Args:
            ear (float): Eye Aspect Ratio value
            
        Returns:
            dict: Blink detection results including state and metrics
        """
        self.ear_history.append(ear)
        
        # Detect eye closure
        if ear < 0.2 and not self.is_eye_closed:
            self.is_eye_closed = True
            self.blink_start = time.time()
            self.current_blink_frame_count = 0
        
        # Detect eye opening
        elif ear >= 0.2 and self.is_eye_closed:
            self.is_eye_closed = False
            if self.blink_start is not None:
                blink_duration = time.time() - self.blink_start
                if blink_duration < 0.5:
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
    """Tracks pupil size and changes over time."""
    
    def __init__(self):
        """Initialize pupil tracker with default parameters."""
        self.diameter_history = deque(maxlen=30)
        self.baseline = None
        self.change_threshold = 0.15
        
    def detect_change(self, current_diameter):
        """
        Detect changes in pupil size relative to baseline.
        
        Args:
            current_diameter (float): Current pupil diameter
            
        Returns:
            dict: Pupil change detection results
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
    """Detects emotional states based on eye behavior patterns."""
    
    def __init__(self):
        """Initialize emotion detector with default parameters."""
        self.blink_rate_history = deque(maxlen=90)
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
        Update emotion detection based on current eye metrics.
        
        Args:
            left_eye (dict): Left eye parameters
            right_eye (dict): Right eye parameters
            left_pupil (dict): Left pupil metrics
            right_pupil (dict): Right pupil metrics
            left_blink (dict): Left blink state
            right_blink (dict): Right blink state
            
        Returns:
            dict: Emotion detection results
        """
        current_time = time.time()
        
        # Update histories
        avg_pupil_size = (left_eye['iris_diameter'] + right_eye['iris_diameter']) / 2
        avg_eye_openness = (left_eye['eye_height'] + right_eye['eye_height']) / 2
        self.pupil_size_history.append(avg_pupil_size)
        self.eye_openness_history.append(avg_eye_openness)
        
        # Calculate blink rate (blinks per minute)
        if left_blink['is_blinking'] or right_blink['is_blinking']:
            if current_time - self.last_blink_time > 0.1:
                self.blink_rate_history.append(1)
                self.last_blink_time = current_time
        
        # Calculate metrics
        blink_rate = len(self.blink_rate_history) * (60 / 3)
        pupil_change = 0
        if len(self.pupil_size_history) > 1:
            pupil_change = (self.pupil_size_history[-1] - self.pupil_size_history[0]) / self.pupil_size_history[0]
        
        eye_openness_ratio = avg_eye_openness / np.mean(list(self.eye_openness_history)) if self.eye_openness_history else 1.0
        
        # Update emotion scores
        self.emotion_scores['surprise'] = self._calculate_surprise_score(eye_openness_ratio, pupil_change)
        self.emotion_scores['focus'] = self._calculate_focus_score(pupil_change, blink_rate)
        self.emotion_scores['tired'] = self._calculate_tired_score(blink_rate, eye_openness_ratio)
        self.emotion_scores['relaxed'] = self._calculate_relaxed_score(pupil_change, blink_rate)
        self.emotion_scores['stressed'] = self._calculate_stressed_score(pupil_change, blink_rate)
        
        # Get dominant emotion
        dominant_emotion = max(self.emotion_scores.items(), key=lambda x: x[1])
        
        return {
            'dominant_emotion': dominant_emotion[0],
            'emotion_scores': self.emotion_scores,
            'blink_rate': blink_rate,
            'pupil_change': pupil_change * 100,
            'eye_openness_ratio': eye_openness_ratio
        }
    
    def _calculate_surprise_score(self, eye_openness_ratio, pupil_change):
        """Calculate surprise score based on eye openness and pupil dilation."""
        score = 0.0
        if eye_openness_ratio > 1.2:
            score += 0.5
        if pupil_change > 0.15:
            score += 0.5
        return min(1.0, score)
    
    def _calculate_focus_score(self, pupil_change, blink_rate):
        """Calculate focus score based on pupil stability and blink rate."""
        score = 0.0
        if abs(pupil_change) < 0.1:
            score += 0.3
        if 10 <= blink_rate <= 15:
            score += 0.4
        if blink_rate < 10:
            score += 0.3
        return min(1.0, score)
    
    def _calculate_tired_score(self, blink_rate, eye_openness_ratio):
        """Calculate tired score based on blink rate and eye openness."""
        score = 0.0
        if blink_rate > 20:
            score += 0.4
        if eye_openness_ratio < 0.8:
            score += 0.6
        return min(1.0, score)
    
    def _calculate_relaxed_score(self, pupil_change, blink_rate):
        """Calculate relaxed score based on pupil stability and blink rate."""
        score = 0.0
        if abs(pupil_change) < 0.05:
            score += 0.5
        if 15 <= blink_rate <= 20:
            score += 0.5
        return min(1.0, score)
    
    def _calculate_stressed_score(self, pupil_change, blink_rate):
        """Calculate stressed score based on pupil variability and blink patterns."""
        score = 0.0
        if abs(pupil_change) > 0.1:
            score += 0.3
        if blink_rate > 20:
            score += 0.3
        if blink_rate < 5:
            score += 0.4
        return min(1.0, score) 