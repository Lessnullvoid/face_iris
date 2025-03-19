"""
This module contains visualization utilities for the eye tracking system.
"""

import cv2
import mediapipe as mp
import numpy as np
import logging
from collections import deque

# MediaPipe Drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configure logging
logger = logging.getLogger(__name__)

# Constants for oscilloscope visualization
HISTORY_LENGTH = 200  # Number of frames to show in history
GRAPH_HEIGHT = 100    # Height of each graph line
GRAPH_PADDING = 20    # Padding between graph lines

# Face mesh specific parameters
FACE_FEATURES = {
    'silhouette': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
    'left_eyebrow': [336, 296, 334, 293, 300, 276, 283, 282, 295, 285],
    'right_eyebrow': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    'nose_bridge': [168, 6, 197, 195, 5, 4, 1, 19, 94],
    'nose_tip': [19, 94, 2, 164, 0, 11, 12, 13, 14, 15, 16, 17, 18, 200, 199, 175],
    'left_eye_contour': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
    'right_eye_contour': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    'lips_outer': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 80, 191],
    'lips_inner': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
}

# Feature colors (BGR format)
FEATURE_COLORS = {
    'silhouette': (255, 0, 0),        # Blue
    'left_eyebrow': (255, 0, 0),      # Blue
    'right_eyebrow': (255, 0, 0),     # Blue
    'nose_bridge': (255, 0, 0),       # Blue
    'nose_tip': (255, 0, 0),          # Blue
    'left_eye_contour': (0, 255, 0),  # Green
    'right_eye_contour': (0, 255, 0), # Green
    'lips_outer': (255, 0, 0),        # Blue
    'lips_inner': (255, 0, 0)         # Blue
}

class GUIControls:
    """Manages GUI controls and their states."""
    
    def __init__(self):
        """Initialize GUI controls and their states."""
        try:
            # Control states
            self.show_mesh = False
            self.show_features = False
            self.show_contours = False
            self.pupillometry_active = False
            self.zoom_level = 0  # 0-4 for zoom levels 1x-3x
            
            # Control labels and their associated states
            self.controls = {
                'mesh': {
                    'label': '[M] Mesh',
                    'state': lambda: self.show_mesh
                },
                'features': {
                    'label': '[F] Features',
                    'state': lambda: self.show_features
                },
                'contours': {
                    'label': '[C] Contours',
                    'state': lambda: self.show_contours
                },
                'pupil': {
                    'label': '[P] Pupillometry',
                    'state': lambda: self.pupillometry_active
                },
                'zoom': {
                    'label': '[Z] Zoom',
                    'state': lambda: bool(self.zoom_level)
                }
            }
            
        except Exception as e:
            logger.error(f"Error initializing GUI controls: {e}")
            raise
    
    def _toggle_state(self, state_name):
        """
        Safely toggle a boolean state.
        
        Args:
            state_name: Name of the state to toggle
        """
        try:
            if hasattr(self, state_name):
                setattr(self, state_name, not getattr(self, state_name))
                logger.debug(f"Toggled {state_name} to {getattr(self, state_name)}")
            else:
                logger.warning(f"Attempted to toggle unknown state: {state_name}")
        except Exception as e:
            logger.error(f"Error toggling state {state_name}: {e}")
    
    def _cycle_zoom(self):
        """Safely cycle through zoom levels."""
        try:
            self.zoom_level = (self.zoom_level + 1) % 5
            logger.debug(f"Cycled zoom level to {self.zoom_level}")
        except Exception as e:
            logger.error(f"Error cycling zoom level: {e}")
    
    def get_zoom_factor(self):
        """
        Get the current zoom factor.
        
        Returns:
            float: Zoom factor between 1.0 and 3.0
        """
        try:
            return 1.0 + (self.zoom_level * 0.5)
        except Exception as e:
            logger.error(f"Error calculating zoom factor: {e}")
            return 1.0
    
    def toggle_state(self, control_name):
        """
        Toggle the state of a control by name.
        
        Args:
            control_name: Name of the control to toggle
        """
        try:
            if control_name == 'zoom':
                self._cycle_zoom()
            elif control_name in ['mesh', 'features', 'contours', 'pupil']:
                state_map = {
                    'mesh': 'show_mesh',
                    'features': 'show_features',
                    'contours': 'show_contours',
                    'pupil': 'pupillometry_active'
                }
                self._toggle_state(state_map[control_name])
            else:
                logger.warning(f"Attempted to toggle unknown control: {control_name}")
        except Exception as e:
            logger.error(f"Error toggling control {control_name}: {e}")
    
    def draw_controls(self, frame):
        """
        Draw control labels and states on the frame.
        
        Args:
            frame: Input frame to draw on
            
        Returns:
            Frame with controls drawn
        """
        try:
            if frame is None:
                raise ValueError("Input frame is None")
            
            # Draw control labels with their states
            y_offset = frame.shape[0] - 30  # Position at bottom of frame
            x_offset = 10
            
            # Create status text
            status_text = []
            for control_info in self.controls.values():
                is_active = control_info['state']()
                label = control_info['label']
                if 'Zoom' in label:
                    status = f"{self.get_zoom_factor()}x"
                else:
                    status = "ON" if is_active else "OFF"
                status_text.append(f"{label}: {status}")
            
            # Join all status text with separators
            full_status = " | ".join(status_text)
            
            # Draw the combined status text with background
            draw_text_with_background(frame, full_status, (x_offset, y_offset),
                                   color=(200, 200, 200), font_scale=0.6)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error drawing controls: {e}")
            return frame if frame is not None else None

class PupillometryVisualizer:
    """Manages the oscilloscope-style visualization of pupillometry data."""
    
    def __init__(self, history_length=HISTORY_LENGTH):
        """Initialize the visualizer with empty histories."""
        self.left_diameter_history = deque(maxlen=history_length)
        self.right_diameter_history = deque(maxlen=history_length)
        self.left_change_history = deque(maxlen=history_length)
        self.right_change_history = deque(maxlen=history_length)
        self.baseline_left = None
        self.baseline_right = None
    
    def update(self, left_eye, right_eye, left_pupil, right_pupil):
        """Update the visualization data with new measurements."""
        # Store raw diameters
        self.left_diameter_history.append(left_eye['iris_diameter'])
        self.right_diameter_history.append(right_eye['iris_diameter'])
        
        # Store percentage changes
        self.left_change_history.append(left_pupil['change_percentage'])
        self.right_change_history.append(right_pupil['change_percentage'])
        
        # Update baselines
        self.baseline_left = left_pupil['baseline_diameter']
        self.baseline_right = right_pupil['baseline_diameter']

def draw_text_with_background(img, text, position, font_scale=0.6, color=(0, 255, 0), thickness=1):
    """
    Draw text with a black background for better visibility.
    
    Args:
        img: Image to draw on
        text: Text to display
        position: (x, y) coordinates for text
        font_scale: Scale of the font
        color: Color of the text (BGR)
        thickness: Thickness of the font
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Calculate background rectangle dimensions
    padding = 4
    bg_rect_start = (position[0] - padding, position[1] - text_height - padding)
    bg_rect_end = (position[0] + text_width + padding, position[1] + padding)
    
    # Draw black background rectangle
    cv2.rectangle(img, bg_rect_start, bg_rect_end, (0, 0, 0), -1)
    
    # Draw text
    cv2.putText(img, text, position, font, font_scale, color, thickness)

def draw_facial_features(image, landmarks, features=FACE_FEATURES, colors=FEATURE_COLORS):
    """
    Draw facial features using custom colors and connections.
    
    Args:
        image: Image to draw on
        landmarks: MediaPipe face landmarks
        features: Dictionary of feature indices
        colors: Dictionary of colors for each feature
    
    Returns:
        Image with drawn facial features
    """
    h, w = image.shape[:2]
    for feature_name, feature_indices in features.items():
        color = colors[feature_name]
        for idx in feature_indices:
            point = landmarks.landmark[idx]
            x, y = int(point.x * w), int(point.y * h)
            cv2.circle(image, (x, y), 2, color, -1)
    
    return image

def draw_face_mesh_tesselation(image, landmarks, mp_face_mesh):
    """
    Draw the full face mesh tesselation.
    
    Args:
        image: Image to draw on
        landmarks: MediaPipe face landmarks
        mp_face_mesh: MediaPipe face mesh solution
    """
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=1)
    )

def draw_face_mesh_contours(image, landmarks, mp_face_mesh):
    """
    Draw face mesh contours.
    
    Args:
        image: Image to draw on
        landmarks: MediaPipe face landmarks
        mp_face_mesh: MediaPipe face mesh solution
    """
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=landmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=1)
    )

def draw_oscilloscope(image, visualizer, rect_y, rect_height):
    """
    Draw oscilloscope-style visualization of pupillometry data.
    
    Args:
        image: Image to draw on
        visualizer: PupillometryVisualizer instance
        rect_y: Y position of the rectangle
        rect_height: Height of the rectangle
    """
    try:
        h, w = image.shape[:2]
        
        # Define graph regions
        graph_width = w - 40  # Leave some padding on sides
        x_start = 20
        
        # Calculate y positions for each graph
        y_positions = {
            'diameter': rect_y + GRAPH_PADDING,
            'change': rect_y + rect_height//2 + GRAPH_PADDING
        }
        
        # Draw graph labels
        draw_text_with_background(image, "Iris Diameter (px)", 
                                (x_start, y_positions['diameter'] - 5),
                                color=(255, 255, 255), font_scale=0.5)
        draw_text_with_background(image, "Pupil Change (%)", 
                                (x_start, y_positions['change'] - 5),
                                color=(255, 255, 255), font_scale=0.5)
        
        # Draw diameter graph
        if visualizer.left_diameter_history and visualizer.right_diameter_history:
            # Calculate scaling factors for diameter
            all_diameters = list(visualizer.left_diameter_history) + list(visualizer.right_diameter_history)
            min_diameter = min(all_diameters)
            max_diameter = max(all_diameters)
            diameter_range = max_diameter - min_diameter
            if diameter_range == 0:
                diameter_range = 1  # Prevent division by zero
            
            diameter_scale = (GRAPH_HEIGHT - 20) / diameter_range
            
            # Draw baseline references if available
            if visualizer.baseline_left and visualizer.baseline_right:
                baseline_y_left = int(y_positions['diameter'] + GRAPH_HEIGHT - 
                                    (visualizer.baseline_left - min_diameter) * diameter_scale)
                baseline_y_right = int(y_positions['diameter'] + GRAPH_HEIGHT - 
                                     (visualizer.baseline_right - min_diameter) * diameter_scale)
                cv2.line(image, (x_start, baseline_y_left), (x_start + graph_width, baseline_y_left),
                        (0, 128, 0), 1, cv2.LINE_AA)
                cv2.line(image, (x_start, baseline_y_right), (x_start + graph_width, baseline_y_right),
                        (0, 0, 128), 1, cv2.LINE_AA)
            
            # Draw diameter history
            points_left = []
            points_right = []
            for i, (left_d, right_d) in enumerate(zip(visualizer.left_diameter_history,
                                                     visualizer.right_diameter_history)):
                x = int(x_start + (i * graph_width / HISTORY_LENGTH))
                y_left = int(y_positions['diameter'] + GRAPH_HEIGHT - 
                            (left_d - min_diameter) * diameter_scale)
                y_right = int(y_positions['diameter'] + GRAPH_HEIGHT - 
                             (right_d - min_diameter) * diameter_scale)
                points_left.append((x, y_left))
                points_right.append((x, y_right))
            
            # Draw the lines
            if len(points_left) > 1:
                cv2.polylines(image, [np.array(points_left)], False, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.polylines(image, [np.array(points_right)], False, (255, 0, 0), 1, cv2.LINE_AA)
        
        # Draw change percentage graph
        if visualizer.left_change_history and visualizer.right_change_history:
            # Fixed scale for percentage change (-50% to +50%)
            change_scale = GRAPH_HEIGHT / 100.0  # -50 to +50 = 100 range
            
            points_left = []
            points_right = []
            for i, (left_c, right_c) in enumerate(zip(visualizer.left_change_history,
                                                     visualizer.right_change_history)):
                x = int(x_start + (i * graph_width / HISTORY_LENGTH))
                # Center the graph vertically and scale the changes
                y_left = int(y_positions['change'] + GRAPH_HEIGHT/2 - (left_c * change_scale))
                y_right = int(y_positions['change'] + GRAPH_HEIGHT/2 - (right_c * change_scale))
                points_left.append((x, y_left))
                points_right.append((x, y_right))
            
            # Draw zero line
            zero_y = int(y_positions['change'] + GRAPH_HEIGHT/2)
            cv2.line(image, (x_start, zero_y), (x_start + graph_width, zero_y),
                    (128, 128, 128), 1, cv2.LINE_AA)
            
            # Draw the lines
            if len(points_left) > 1:
                cv2.polylines(image, [np.array(points_left)], False, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.polylines(image, [np.array(points_right)], False, (255, 0, 0), 1, cv2.LINE_AA)
        
        # Draw legend
        legend_x = w - 150
        legend_y = rect_y + 20
        draw_text_with_background(image, "Left Eye", (legend_x, legend_y),
                                color=(0, 255, 0), font_scale=0.5)
        draw_text_with_background(image, "Right Eye", (legend_x, legend_y + 20),
                                color=(255, 0, 0), font_scale=0.5)
        
    except Exception as e:
        logger.error(f"Error drawing oscilloscope: {e}")

def organize_display_layout(image, fps, emotion_data, left_blink, right_blink, left_pupil, right_pupil, left_eye, right_eye, pupillometry_metrics=None):
    """
    Organize and draw the display layout for metrics and information.
    
    Args:
        image: Image to draw on
        fps: Current FPS
        emotion_data: Dictionary containing emotion detection results
        left_blink, right_blink: Blink detection results
        left_pupil, right_pupil: Pupil tracking results
        left_eye, right_eye: Eye parameter results
        pupillometry_metrics: Optional pupillometry analysis results
    
    Returns:
        Image with organized display layout
    """
    h, w = image.shape[:2]
    
    # Get or create pupillometry visualizer
    if not hasattr(organize_display_layout, 'visualizer'):
        organize_display_layout.visualizer = PupillometryVisualizer()
    
    # Update visualizer data
    organize_display_layout.visualizer.update(left_eye, right_eye, left_pupil, right_pupil)
    
    # Draw oscilloscope in the green rectangle area
    rect_y = h // 3
    rect_height = h // 3
    draw_oscilloscope(image, organize_display_layout.visualizer, rect_y, rect_height)
    
    # Define display regions
    left_margin = 10
    right_margin = w - 300
    
    # Colors for different sections
    TITLE_COLOR = (255, 255, 255)    # White
    METRIC_COLOR = (0, 255, 0)       # Green
    ALERT_COLOR = (0, 165, 255)      # Orange
    WARNING_COLOR = (0, 0, 255)      # Red
    INFO_COLOR = (200, 200, 200)     # Light gray
    
    # Draw left eye parameters
    y_offset = 30
    left_params = [
        ("Left Eye Parameters:", TITLE_COLOR),
        (f"Iris Center: ({left_eye['iris_center'][0]}, {left_eye['iris_center'][1]})", METRIC_COLOR),
        (f"Iris Diameter: {left_eye['iris_diameter']:.1f}px", METRIC_COLOR),
        (f"Pupil Change: {left_pupil['change_percentage']:.1f}%", METRIC_COLOR),
        (f"Eye Height: {left_eye['eye_height']:.1f}px", METRIC_COLOR),
        (f"EAR: {left_eye['ear']:.3f}", METRIC_COLOR),
        (f"Blinks: {left_blink['blink_count']}", METRIC_COLOR)
    ]
    
    for text, color in left_params:
        draw_text_with_background(image, text, (left_margin, y_offset), color=color)
        y_offset += 25
    
    # Draw right eye parameters
    y_offset = 30
    right_params = [
        ("Right Eye Parameters:", TITLE_COLOR),
        (f"Iris Center: ({right_eye['iris_center'][0]}, {right_eye['iris_center'][1]})", METRIC_COLOR),
        (f"Iris Diameter: {right_eye['iris_diameter']:.1f}px", METRIC_COLOR),
        (f"Pupil Change: {right_pupil['change_percentage']:.1f}%", METRIC_COLOR),
        (f"Eye Height: {right_eye['eye_height']:.1f}px", METRIC_COLOR),
        (f"EAR: {right_eye['ear']:.3f}", METRIC_COLOR),
        (f"Blinks: {right_blink['blink_count']}", METRIC_COLOR)
    ]
    
    for text, color in right_params:
        draw_text_with_background(image, text, (right_margin, y_offset), color=color)
        y_offset += 25
    
    # Draw emotion feedback in lower right, but higher up
    emotion_x = w - 300
    emotion_y = h - 280  # Moved up by 100 pixels from previous position
    
    # Draw emotion section title with black background
    draw_text_with_background(image, "Emotion Analysis:", 
                            (emotion_x, emotion_y), 
                            color=TITLE_COLOR, font_scale=0.6)
    emotion_y += 25
    
    # Draw dominant emotion with larger font
    if 'dominant_emotion' in emotion_data:
        dominant_text = f"State: {emotion_data['dominant_emotion'].upper()}"
        draw_text_with_background(image, dominant_text,
                                (emotion_x, emotion_y),
                                color=ALERT_COLOR, font_scale=0.7)
        emotion_y += 30
    
    # Draw emotion scores with bar visualization
    if 'emotion_scores' in emotion_data:
        bar_width = 100
        bar_height = 15
        padding = 5
        
        for emotion, score in emotion_data['emotion_scores'].items():
            # Draw emotion label
            draw_text_with_background(image, f"{emotion}:",
                                   (emotion_x, emotion_y),
                                   color=METRIC_COLOR, font_scale=0.5)
            
            # Draw score bar background
            bar_x = emotion_x + 80
            bar_y = emotion_y - bar_height + 5
            cv2.rectangle(image,
                        (bar_x, bar_y),
                        (bar_x + bar_width, bar_y + bar_height),
                        (50, 50, 50), -1)
            
            # Draw score bar fill
            fill_width = int(bar_width * score)
            if fill_width > 0:
                cv2.rectangle(image,
                            (bar_x, bar_y),
                            (bar_x + fill_width, bar_y + bar_height),
                            METRIC_COLOR, -1)
            
            # Draw score percentage
            score_text = f"{score*100:.0f}%"
            draw_text_with_background(image, score_text,
                                   (bar_x + bar_width + 5, emotion_y),
                                   color=METRIC_COLOR, font_scale=0.5)
            
            emotion_y += 20
    
    # Draw additional metrics
    if 'blink_rate' in emotion_data:
        metrics_text = [
            f"Blink Rate: {emotion_data['blink_rate']:.1f}/min",
            f"Eye Openness: {emotion_data['eye_openness_ratio']:.2f}"
        ]
        
        emotion_y += 5  # Add small space
        for text in metrics_text:
            draw_text_with_background(image, text,
                                   (emotion_x, emotion_y),
                                   color=INFO_COLOR, font_scale=0.5)
            emotion_y += 20
    
    # Draw controls and status at the very bottom
    bottom_margin = h - 30  # Moved up slightly
    controls_text = [
        ("Controls: [M]esh | [F]eatures | [C]ontours | [P]upillometry", INFO_COLOR),
        (f"FPS: {fps:.1f}", INFO_COLOR)
    ]
    
    y_offset = bottom_margin
    for text, color in controls_text:
        draw_text_with_background(image, text, (left_margin, y_offset), color=color)
        y_offset += 20  # Reduced spacing between control lines
    
    # Draw alerts
    if left_blink['is_blinking'] or right_blink['is_blinking']:
        draw_text_with_background(image, "BLINK DETECTED!", 
                                (w//2 - 100, bottom_margin - 30), 
                                color=WARNING_COLOR, font_scale=0.7)
    
    if left_pupil['is_changing'] or right_pupil['is_changing']:
        avg_change = (left_pupil['change_percentage'] + right_pupil['change_percentage']) / 2
        draw_text_with_background(image, f"PUPIL CHANGE: {avg_change:.1f}%", 
                                (w//2 - 100, bottom_margin), 
                                color=ALERT_COLOR, font_scale=0.7)
    
    # Draw pupillometry analysis if active
    if pupillometry_metrics and pupillometry_metrics['status'] != 'baseline_collection':
        draw_text_with_background(image, "Pupillometry Analysis:", 
                                (w//2 - 100, 30), color=TITLE_COLOR)
        metrics_text = [
            f"Time: {pupillometry_metrics['time']:.1f}s",
            f"L/R Change: {pupillometry_metrics['left_change']:.1f}%/{pupillometry_metrics['right_change']:.1f}%",
            f"Variability: {pupillometry_metrics['left_variability']:.2f}/{pupillometry_metrics['right_variability']:.2f}",
            f"Velocity: {pupillometry_metrics['left_velocity']:.1f}/{pupillometry_metrics['right_velocity']:.1f}"
        ]
        y_offset = 60
        for text in metrics_text:
            draw_text_with_background(image, text, (w//2 - 100, y_offset), color=METRIC_COLOR)
            y_offset += 25
    
    return image

def zoom_image(image, zoom_factor):
    """
    Zoom into the center of the image.
    
    Args:
        image: Image to zoom
        zoom_factor: Zoom factor (1.0 means no zoom)
    
    Returns:
        Zoomed image
    """
    if zoom_factor == 1:
        return image
    
    height, width = image.shape[:2]
    
    # Calculate dimensions for cropping
    new_height = int(height / zoom_factor)
    new_width = int(width / zoom_factor)
    
    # Calculate crop starting points to keep center
    start_y = (height - new_height) // 2
    start_x = (width - new_width) // 2
    
    # Crop image
    cropped = image[start_y:start_y + new_height, start_x:start_x + new_width]
    
    # Resize back to original size
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR) 