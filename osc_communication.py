"""
This module handles OSC (Open Sound Control) communication for the eye tracking system.
"""

from pythonosc import udp_client
import logging

logger = logging.getLogger(__name__)

class OSCHandler:
    """Handles OSC communication for sending eye tracking data."""
    
    def __init__(self, ip="127.0.0.1", port=12345):
        """
        Initialize OSC handler.
        
        Args:
            ip (str): IP address for OSC communication
            port (int): Port number for OSC communication
        """
        try:
            self.client = udp_client.SimpleUDPClient(ip, port)
            self.connected = True
            logger.info(f"OSC client initialized at {ip}:{port}")
        except Exception as e:
            logger.error(f"Failed to initialize OSC client: {e}")
            self.connected = False
            self.client = None
    
    def send_eye_data(self, left_eye, right_eye, left_blink, right_blink, left_pupil, right_pupil, emotion_data):
        """
        Send eye tracking data via OSC.
        
        Args:
            left_eye, right_eye: Eye parameter dictionaries
            left_blink, right_blink: Blink detection dictionaries
            left_pupil, right_pupil: Pupil tracking dictionaries
            emotion_data: Emotion detection dictionary
        """
        if not self.connected or not self.client:
            return
            
        try:
            # Basic iris parameters
            self.client.send_message("/iris/left/position", list(left_eye['iris_center']))
            self.client.send_message("/iris/right/position", list(right_eye['iris_center']))
            self.client.send_message("/iris/left/diameter", float(left_eye['iris_diameter']))
            self.client.send_message("/iris/right/diameter", float(right_eye['iris_diameter']))
            
            # Eye openness and aspect ratio
            self.client.send_message("/eye/left/height", float(left_eye['eye_height']))
            self.client.send_message("/eye/right/height", float(right_eye['eye_height']))
            self.client.send_message("/eye/left/ear", float(left_eye['ear']))
            self.client.send_message("/eye/right/ear", float(right_eye['ear']))
            
            # Blink detection
            self.client.send_message("/eye/left/blink/state", int(left_blink['is_blinking']))
            self.client.send_message("/eye/right/blink/state", int(right_blink['is_blinking']))
            self.client.send_message("/eye/left/blink/count", int(left_blink['blink_count']))
            self.client.send_message("/eye/right/blink/count", int(right_blink['blink_count']))
            self.client.send_message("/eye/left/blink/duration", int(left_blink['current_blink_frames']))
            self.client.send_message("/eye/right/blink/duration", int(right_blink['current_blink_frames']))
            
            # Full eye contour points
            self.client.send_message("/eye/left/contour", [coord for point in left_eye['eye_coords'] for coord in point])
            self.client.send_message("/eye/right/contour", [coord for point in right_eye['eye_coords'] for coord in point])
            
            # Full iris contour points
            self.client.send_message("/iris/left/contour", [coord for point in left_eye['iris_coords'] for coord in point])
            self.client.send_message("/iris/right/contour", [coord for point in right_eye['iris_coords'] for coord in point])
            
            # Pupil change messages
            self.client.send_message("/pupil/left/changing", int(left_pupil['is_changing']))
            self.client.send_message("/pupil/right/changing", int(right_pupil['is_changing']))
            self.client.send_message("/pupil/left/change_percentage", float(left_pupil['change_percentage']))
            self.client.send_message("/pupil/right/change_percentage", float(right_pupil['change_percentage']))
            self.client.send_message("/pupil/left/baseline", float(left_pupil['baseline_diameter']) if left_pupil['baseline_diameter'] else 0.0)
            self.client.send_message("/pupil/right/baseline", float(right_pupil['baseline_diameter']) if right_pupil['baseline_diameter'] else 0.0)
            
            # Emotion messages
            self.client.send_message("/emotion/dominant", emotion_data['dominant_emotion'])
            for emotion, score in emotion_data['emotion_scores'].items():
                self.client.send_message(f"/emotion/score/{emotion}", float(score))
            self.client.send_message("/emotion/blink_rate", float(emotion_data['blink_rate']))
            self.client.send_message("/emotion/eye_openness", float(emotion_data['eye_openness_ratio']))
            
            # Aggregated metrics
            avg_ear = (left_eye['ear'] + right_eye['ear']) / 2
            avg_diameter = (left_eye['iris_diameter'] + right_eye['iris_diameter']) / 2
            total_blinks = (left_blink['blink_count'] + right_blink['blink_count']) // 2
            
            self.client.send_message("/eye/average/ear", float(avg_ear))
            self.client.send_message("/iris/average/diameter", float(avg_diameter))
            self.client.send_message("/eye/total/blinks", int(total_blinks))
            
        except Exception as e:
            logger.error(f"Error sending OSC messages: {e}")
            self.connected = False 

    def send_pupillometry_data(self, metrics):
        """Send pupillometry analysis data via OSC."""
        if not self.connected or not metrics:
            return
            
        try:
            # Send analysis status
            self.client.send_message("/pupillometry/status", metrics['status'])
            
            # Only send analysis data if we're not in baseline collection
            if metrics['status'] != 'baseline_collection':
                # Time and basic metrics
                self.client.send_message("/pupillometry/time", float(metrics['time']))
                self.client.send_message("/pupillometry/left/change", float(metrics['left_change']))
                self.client.send_message("/pupillometry/right/change", float(metrics['right_change']))
                
                # Variability metrics
                self.client.send_message("/pupillometry/left/variability", float(metrics['left_variability']))
                self.client.send_message("/pupillometry/right/variability", float(metrics['right_variability']))
                
                # Velocity metrics
                self.client.send_message("/pupillometry/left/velocity", float(metrics['left_velocity']))
                self.client.send_message("/pupillometry/right/velocity", float(metrics['right_velocity']))
                
                # Average metrics
                avg_change = (metrics['left_change'] + metrics['right_change']) / 2
                avg_variability = (metrics['left_variability'] + metrics['right_variability']) / 2
                avg_velocity = (metrics['left_velocity'] + metrics['right_velocity']) / 2
                
                self.client.send_message("/pupillometry/average/change", float(avg_change))
                self.client.send_message("/pupillometry/average/variability", float(avg_variability))
                self.client.send_message("/pupillometry/average/velocity", float(avg_velocity))
            else:
                # Send baseline collection progress
                self.client.send_message("/pupillometry/baseline/progress", float(metrics['progress']))
            
        except Exception as e:
            logger.error(f"Error sending pupillometry OSC messages: {e}") 