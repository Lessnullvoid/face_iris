"""
Main module for the eye tracking system.
Coordinates the different components and runs the main processing loop.
"""

import cv2
import mediapipe as mp
import logging
import warnings
from pupillometry import PupillometryAnalyzer
from detectors import BlinkDetector, PupilTracker, EmotionDetector
from visualization import (draw_facial_features, draw_face_mesh_tesselation,
                         draw_face_mesh_contours, organize_display_layout, zoom_image,
                         GUIControls)
from eye_tracking import calculate_eye_parameters
from osc_communication import OSCHandler

# Filter warnings and logging
warnings.filterwarnings('ignore', category=UserWarning)
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EyeTrackingSystem:
    """Main class that coordinates the eye tracking system components."""
    
    def __init__(self):
        """Initialize the eye tracking system."""
        try:
            # Initialize MediaPipe Face Mesh
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                static_image_mode=False
            )
            
            # Initialize detectors
            self.left_blink_detector = BlinkDetector()
            self.right_blink_detector = BlinkDetector()
            self.left_pupil_tracker = PupilTracker()
            self.right_pupil_tracker = PupilTracker()
            self.emotion_detector = EmotionDetector()
            self.pupillometry_analyzer = PupillometryAnalyzer()
            
            # Initialize OSC communication
            self.osc_handler = OSCHandler()
            
            # Initialize video capture
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Error: Could not open video capture device")
            
            # Create window and GUI controls
            cv2.namedWindow('Iris Tracking')
            self.gui = GUIControls()
            
            self.prev_time = cv2.getTickCount()
            self.running = True
            
        except Exception as e:
            logger.error(f"Error initializing EyeTrackingSystem: {e}")
            raise
    
    def _handle_keyboard_input(self, key):
        """
        Handle keyboard input events.
        
        Args:
            key: Key code from cv2.waitKey()
            
        Returns:
            bool: True to continue running, False to quit
        """
        try:
            if key == -1:  # No key pressed
                return True
                
            key = key & 0xFF  # Convert to ASCII
            
            if key == 27 or key == ord('q'):  # ESC or 'q' to quit
                return False
                
            # Map keys to control names
            key_map = {
                ord('m'): 'mesh',
                ord('f'): 'features',
                ord('c'): 'contours',
                ord('p'): 'pupil',
                ord('z'): 'zoom'
            }
            
            if key in key_map:
                self.gui.toggle_state(key_map[key])
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling keyboard input: {e}")
            return True
    
    def _process_frame(self, frame):
        """
        Process a single frame.
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Processed frame with visualizations
        """
        try:
            if frame is None:
                raise ValueError("Input frame is None")
            
            # Apply zoom
            frame = zoom_image(frame, self.gui.get_zoom_factor())
            
            # Create a copy for face mesh visualization
            mesh_image = frame.copy()
            
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.face_mesh.process(frame_rgb)
            
            if not results.multi_face_landmarks:
                # Draw controls even if no face is detected
                mesh_image = self.gui.draw_controls(mesh_image)
                return mesh_image
                
            # Process face landmarks
            for face_landmarks in results.multi_face_landmarks:
                # Draw visualizations based on current state
                if self.gui.show_mesh:
                    draw_face_mesh_tesselation(mesh_image, face_landmarks, self.mp_face_mesh)
                if self.gui.show_contours:
                    draw_face_mesh_contours(mesh_image, face_landmarks, self.mp_face_mesh)
                if self.gui.show_features:
                    draw_facial_features(mesh_image, face_landmarks)
                
                # Calculate eye parameters
                left_eye = calculate_eye_parameters(face_landmarks, frame.shape, "left")
                right_eye = calculate_eye_parameters(face_landmarks, frame.shape, "right")
                
                if left_eye is None or right_eye is None:
                    continue
                
                # Process eye data
                left_blink = self.left_blink_detector.detect(left_eye['ear'])
                right_blink = self.right_blink_detector.detect(right_eye['ear'])
                left_pupil = self.left_pupil_tracker.detect_change(left_eye['iris_diameter'])
                right_pupil = self.right_pupil_tracker.detect_change(right_eye['iris_diameter'])
                emotion_data = self.emotion_detector.update(
                    left_eye, right_eye,
                    left_pupil, right_pupil,
                    left_blink, right_blink
                )
                
                # Handle pupillometry
                pupil_metrics = None
                if self.gui.pupillometry_active:
                    if not self.pupillometry_analyzer.recording:
                        self.pupillometry_analyzer.start_recording()
                    pupil_metrics = self.pupillometry_analyzer.update(
                        left_eye['iris_diameter'],
                        right_eye['iris_diameter']
                    )
                    # Send pupillometry data via OSC
                    if self.osc_handler.connected:
                        self.osc_handler.send_pupillometry_data(pupil_metrics)
                elif self.pupillometry_analyzer.recording:
                    self.pupillometry_analyzer.stop_recording()
                
                # Calculate FPS
                curr_time = cv2.getTickCount()
                fps = cv2.getTickFrequency() / (curr_time - self.prev_time)
                self.prev_time = curr_time
                
                # Update display
                mesh_image = organize_display_layout(
                    mesh_image, fps, emotion_data,
                    left_blink, right_blink,
                    left_pupil, right_pupil,
                    left_eye, right_eye,
                    pupillometry_metrics=pupil_metrics
                )
                
                # Send OSC data
                if self.osc_handler.connected:
                    self.osc_handler.send_eye_data(
                        left_eye, right_eye,
                        left_blink, right_blink,
                        left_pupil, right_pupil,
                        emotion_data
                    )
            
            # Draw GUI controls
            mesh_image = self.gui.draw_controls(mesh_image)
            return mesh_image
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame if frame is not None else None
    
    def run(self):
        """Run the main processing loop."""
        try:
            while self.running:
                # Read frame
                success, frame = self.cap.read()
                if not success:
                    logger.error("Failed to read frame")
                    break
                
                # Process frame
                processed_frame = self._process_frame(frame)
                if processed_frame is None:
                    logger.error("Failed to process frame")
                    continue
                
                # Display result
                cv2.imshow('Iris Tracking', processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1)
                self.running = self._handle_keyboard_input(key)
                    
        except KeyboardInterrupt:
            logger.info("\nGracefully shutting down...")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if hasattr(self, 'cap'):
                self.cap.release()
            cv2.destroyAllWindows()
            if hasattr(self, 'face_mesh'):
                self.face_mesh.close()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    try:
        system = EyeTrackingSystem()
        system.run()
    except Exception as e:
        logger.error(f"Application error: {e}")
        cv2.destroyAllWindows() 