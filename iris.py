# Import necessary libraries
import cv2
import mediapipe as mp
from pythonosc import udp_client
import numpy as np
import logging
import warnings
from collections import deque
import time
from pupillometry import PupillometryAnalyzer

# Filter warnings
warnings.filterwarnings('ignore', category=UserWarning)
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# MediaPipe Drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Face mesh specific parameters
FACE_MESH_TESSELATION_COLOR = (0, 0, 0)  # Black color for mesh
FACE_MESH_CONTOURS_COLOR = (0, 0, 0)     # Black color for contours
FACE_MESH_THICKNESS = 1

# Custom drawing styles for face mesh
CUSTOM_FACE_MESH_STYLE = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)  # White markers
CUSTOM_FACE_MESH_CONNECTIONS_STYLE = mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=1)  # Black connections

# Define facial feature indices
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

# Feature colors (BGR format) - All blue except for eyes and iris
FEATURE_COLORS = {
    'silhouette': (255, 0, 0),        # Blue
    'left_eyebrow': (255, 0, 0),      # Blue
    'right_eyebrow': (255, 0, 0),     # Blue
    'nose_bridge': (255, 0, 0),       # Blue
    'nose_tip': (255, 0, 0),          # Blue
    'left_eye_contour': (0, 255, 0),  # Keep green for visibility
    'right_eye_contour': (0, 255, 0), # Keep green for visibility
    'lips_outer': (255, 0, 0),        # Blue
    'lips_inner': (255, 0, 0)         # Blue
}

def draw_facial_features(image, landmarks, features=FACE_FEATURES, colors=FEATURE_COLORS):
    """Draw facial features using custom colors and connections"""
    h, w = image.shape[:2]
    for feature_name, feature_indices in features.items():
        color = colors[feature_name]
        for idx in feature_indices:
            point = landmarks.landmark[idx]
            x, y = int(point.x * w), int(point.y * h)
            # Draw points for all features
            cv2.circle(image, (x, y), 2, color, -1)
    
    return image

def draw_face_mesh_tesselation(image, landmarks):
    """Draw the full face mesh tesselation"""
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=CUSTOM_FACE_MESH_STYLE,
        connection_drawing_spec=CUSTOM_FACE_MESH_CONNECTIONS_STYLE
    )

def draw_face_mesh_contours(image, landmarks):
    """Draw face mesh contours"""
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=landmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=CUSTOM_FACE_MESH_STYLE,
        connection_drawing_spec=CUSTOM_FACE_MESH_CONNECTIONS_STYLE
    )

# Initialize MediaPipe Iris with more specific parameters
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    static_image_mode=False
)

# Initialize OSC client
try:
    osc_client = udp_client.SimpleUDPClient("127.0.0.1", 12345)
except Exception as e:
    print(f"Failed to initialize OSC client: {e}")
    osc_client = None

# Constants for blink detection
EAR_THRESHOLD = 0.2  # Threshold for eye closure
BLINK_CONSEC_FRAMES = 2  # Number of consecutive frames for a blink
BLINK_TIME_THRESHOLD = 0.5  # Maximum time for a blink in seconds

# Add this class before the calculate_eye_parameters function
class BlinkDetector:
    def __init__(self):
        self.ear_history = deque(maxlen=3)
        self.blink_start = None
        self.blink_counter = 0
        self.current_blink_frame_count = 0
        self.is_eye_closed = False
        
    def detect(self, ear):
        self.ear_history.append(ear)
        
        # Detect eye closure
        if ear < EAR_THRESHOLD and not self.is_eye_closed:
            self.is_eye_closed = True
            self.blink_start = time.time()
            self.current_blink_frame_count = 0
        
        # Detect eye opening
        elif ear >= EAR_THRESHOLD and self.is_eye_closed:
            self.is_eye_closed = False
            if self.blink_start is not None:
                blink_duration = time.time() - self.blink_start
                if blink_duration < BLINK_TIME_THRESHOLD:
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
    def __init__(self):
        self.diameter_history = deque(maxlen=30)  # Track last 30 frames
        self.baseline = None
        self.change_threshold = 0.15  # 15% change threshold
        
    def detect_change(self, current_diameter):
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
            'change_percentage': change_percentage * 100,  # Convert to percentage
            'current_diameter': current_diameter,
            'baseline_diameter': self.baseline
        }

class EmotionDetector:
    def __init__(self):
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
        current_time = time.time()
        
        # Update histories
        avg_pupil_size = (left_eye['iris_diameter'] + right_eye['iris_diameter']) / 2
        avg_eye_openness = (left_eye['eye_height'] + right_eye['eye_height']) / 2
        self.pupil_size_history.append(avg_pupil_size)
        self.eye_openness_history.append(avg_eye_openness)
        
        # Calculate blink rate (blinks per minute)
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
            'pupil_change': pupil_change * 100,  # Convert to percentage
            'eye_openness_ratio': eye_openness_ratio
        }
    
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

def calculate_eye_parameters(landmarks, image_shape, side="left"):
    """Calculate various eye parameters including iris size and eye openness"""
    try:
        if side == "left":
            iris_points = [468, 469, 470, 471, 472]
            eye_points = [362, 385, 387, 263, 373, 380]  # Left eye contour
        else:
            iris_points = [473, 474, 475, 476, 477]
            eye_points = [33, 160, 158, 133, 153, 144]  # Right eye contour

        # Get iris points
        iris_landmarks = [landmarks.landmark[i] for i in iris_points]
        iris_coords = [(int(point.x * image_shape[1]), int(point.y * image_shape[0])) 
                       for point in iris_landmarks]
        
        # Calculate iris size (diameter)
        iris_center = iris_coords[0]
        iris_edge = iris_coords[1]
        iris_diameter = np.sqrt(
            (iris_center[0] - iris_edge[0])**2 + 
            (iris_center[1] - iris_edge[1])**2
        ) * 2

        # Get eye contour points
        eye_landmarks = [landmarks.landmark[i] for i in eye_points]
        eye_coords = [(int(point.x * image_shape[1]), int(point.y * image_shape[0])) 
                      for point in eye_landmarks]
        
        # Calculate eye openness (vertical distance between top and bottom eye points)
        eye_height = abs(eye_coords[1][1] - eye_coords[5][1])
        
        # Calculate eye aspect ratio (EAR)
        ear = (eye_height) / (abs(eye_coords[0][0] - eye_coords[3][0]) + 1e-6)
        
        return {
            'iris_center': iris_center,
            'iris_coords': iris_coords,
            'iris_diameter': iris_diameter,
            'eye_coords': eye_coords,
            'eye_height': eye_height,
            'ear': ear
        }
    except Exception as e:
        print(f"Error in calculate_eye_parameters: {e}")
        return None

def zoom_image(image, zoom_factor):
    """Zoom into the center of the image"""
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

def on_zoom_change(value):
    """Callback for zoom slider"""
    # Convert slider value to zoom factor (1.0 to 4.0)
    zoom_factor = 1.0 + (value / 100.0 * 3.0)
    on_zoom_change.zoom_factor = zoom_factor

# Initialize zoom factor
on_zoom_change.zoom_factor = 1.0

# Add after other global variables
PUPILLOMETRY_MODE = False
pupillometry_analyzer = PupillometryAnalyzer()

# Add this function before organize_display_layout
def draw_text_with_background(img, text, position, font_scale=0.6, color=(0, 255, 0), thickness=1):
    """Draw text with a black background for better visibility"""
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

# Then the organize_display_layout function can use it
def organize_display_layout(image, fps, emotion_data, left_blink, right_blink, left_pupil, right_pupil, left_eye, right_eye, pupillometry_metrics=None):
    """Organize the display layout for all metrics and information"""
    h, w = image.shape[:2]
    
    # Define display regions
    left_margin = 10
    right_margin = w - 300  # Adjust based on your needs
    
    # Colors for different sections
    TITLE_COLOR = (255, 255, 255)    # White
    METRIC_COLOR = (0, 255, 0)       # Green
    ALERT_COLOR = (0, 165, 255)      # Orange
    WARNING_COLOR = (0, 0, 255)      # Red
    INFO_COLOR = (200, 200, 200)     # Light gray
    
    # Left column - Eye Parameters
    y_offset = 30
    # Left Eye Parameters
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
    
    # Right Eye Parameters
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
    
    # Bottom section - Controls and Status
    bottom_margin = h - 100
    controls_text = [
        ("Controls: [M]esh | [F]eatures | [C]ontours | [P]upillometry", INFO_COLOR),
        (f"Zoom: {on_zoom_change.zoom_factor:.1f}x", INFO_COLOR),
        (f"FPS: {fps:.1f}", INFO_COLOR)
    ]
    
    y_offset = bottom_margin
    for text, color in controls_text:
        draw_text_with_background(image, text, (left_margin, y_offset), color=color)
        y_offset += 30
    
    # Alert messages (centered)
    if left_blink['is_blinking'] or right_blink['is_blinking']:
        draw_text_with_background(image, "BLINK DETECTED!", 
                                (w//2 - 100, bottom_margin - 30), 
                                color=WARNING_COLOR, font_scale=0.7)
    
    if left_pupil['is_changing'] or right_pupil['is_changing']:
        avg_change = (left_pupil['change_percentage'] + right_pupil['change_percentage']) / 2
        draw_text_with_background(image, f"PUPIL CHANGE: {avg_change:.1f}%", 
                                (w//2 - 100, bottom_margin), 
                                color=ALERT_COLOR, font_scale=0.7)
    
    # Pupillometry section (if active)
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

def main():
    # Start video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture device")
        return

    # Initialize detectors
    left_blink_detector = BlinkDetector()
    right_blink_detector = BlinkDetector()
    left_pupil_tracker = PupilTracker()
    right_pupil_tracker = PupilTracker()
    emotion_detector = EmotionDetector()

    # Create window and add controls
    cv2.namedWindow('Iris Tracking')
    cv2.createTrackbar('Zoom', 'Iris Tracking', 0, 100, on_zoom_change)
    
    # Add visualization controls
    cv2.createTrackbar('Show Mesh', 'Iris Tracking', 0, 1, lambda x: None)
    cv2.createTrackbar('Show Features', 'Iris Tracking', 1, 1, lambda x: None)
    cv2.createTrackbar('Show Contours', 'Iris Tracking', 0, 1, lambda x: None)

    # Add pupillometry control
    cv2.createTrackbar('Pupillometry', 'Iris Tracking', 0, 1, lambda x: None)

    try:
        while True:
            success, image = cap.read()
            if not success:
                print("Error: Failed to read frame")
                break

            # Get visualization settings
            show_mesh = cv2.getTrackbarPos('Show Mesh', 'Iris Tracking')
            show_features = cv2.getTrackbarPos('Show Features', 'Iris Tracking')
            show_contours = cv2.getTrackbarPos('Show Contours', 'Iris Tracking')

            # Get pupillometry mode state
            pupillometry_active = cv2.getTrackbarPos('Pupillometry', 'Iris Tracking')

            # Apply zoom
            image = zoom_image(image, on_zoom_change.zoom_factor)

            # Create a copy for face mesh visualization
            mesh_image = image.copy()

            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the image and find the face landmarks
            results = face_mesh.process(image_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Draw face mesh visualizations based on settings
                    if show_mesh:
                        draw_face_mesh_tesselation(mesh_image, face_landmarks)
                    if show_contours:
                        draw_face_mesh_contours(mesh_image, face_landmarks)
                    if show_features:
                        draw_facial_features(mesh_image, face_landmarks)

                    # Calculate parameters for both eyes
                    left_eye = calculate_eye_parameters(face_landmarks, image.shape, "left")
                    right_eye = calculate_eye_parameters(face_landmarks, image.shape, "right")

                    if left_eye is None or right_eye is None:
                        continue

                    # Detect blinks, pupil changes, and emotions
                    left_blink = left_blink_detector.detect(left_eye['ear'])
                    right_blink = right_blink_detector.detect(right_eye['ear'])
                    left_pupil = left_pupil_tracker.detect_change(left_eye['iris_diameter'])
                    right_pupil = right_pupil_tracker.detect_change(right_eye['iris_diameter'])
                    emotion_data = emotion_detector.update(
                        left_eye, right_eye,
                        left_pupil, right_pupil,
                        left_blink, right_blink
                    )

                    # Get pupillometry metrics if active
                    pupil_metrics = None
                    if pupillometry_active:
                        if not pupillometry_analyzer.recording:
                            pupillometry_analyzer.start_recording()
                        pupil_metrics = pupillometry_analyzer.update(
                            left_eye['iris_diameter'],
                            right_eye['iris_diameter']
                        )
                        
                        if pupil_metrics and pupil_metrics['status'] == 'baseline_collection':
                            draw_text_with_background(
                                mesh_image,
                                f"Collecting baseline: {pupil_metrics['progress']:.1f}%",
                                (10, mesh_image.shape[0] - 200),
                                color=(0, 255, 255)
                            )
                    
                    elif pupillometry_analyzer.recording:
                        pupillometry_analyzer.stop_recording()

                    # Calculate FPS
                    if not hasattr(calculate_eye_parameters, 'prev_time'):
                        calculate_eye_parameters.prev_time = cv2.getTickCount()
                    
                    curr_time = cv2.getTickCount()
                    time_diff = (curr_time - calculate_eye_parameters.prev_time) / cv2.getTickFrequency()
                    fps = 1.0 / time_diff
                    calculate_eye_parameters.prev_time = curr_time

                    # Organize and draw all display elements
                    mesh_image = organize_display_layout(
                        mesh_image,
                        fps,
                        emotion_data,
                        left_blink,
                        right_blink,
                        left_pupil,
                        right_pupil,
                        left_eye,
                        right_eye,
                        pupillometry_metrics=pupil_metrics
                    )

                    # Send OSC messages
                    if osc_client:
                        try:
                            # Basic iris parameters
                            osc_client.send_message("/iris/left/position", list(left_eye['iris_center']))
                            osc_client.send_message("/iris/right/position", list(right_eye['iris_center']))
                            osc_client.send_message("/iris/left/diameter", float(left_eye['iris_diameter']))
                            osc_client.send_message("/iris/right/diameter", float(right_eye['iris_diameter']))
                            
                            # Eye openness and aspect ratio
                            osc_client.send_message("/eye/left/height", float(left_eye['eye_height']))
                            osc_client.send_message("/eye/right/height", float(right_eye['eye_height']))
                            osc_client.send_message("/eye/left/ear", float(left_eye['ear']))
                            osc_client.send_message("/eye/right/ear", float(right_eye['ear']))
                            
                            # Blink detection
                            osc_client.send_message("/eye/left/blink/state", int(left_blink['is_blinking']))
                            osc_client.send_message("/eye/right/blink/state", int(right_blink['is_blinking']))
                            osc_client.send_message("/eye/left/blink/count", int(left_blink['blink_count']))
                            osc_client.send_message("/eye/right/blink/count", int(right_blink['blink_count']))
                            osc_client.send_message("/eye/left/blink/duration", int(left_blink['current_blink_frames']))
                            osc_client.send_message("/eye/right/blink/duration", int(right_blink['current_blink_frames']))
                            
                            # Full eye contour points
                            osc_client.send_message("/eye/left/contour", [coord for point in left_eye['eye_coords'] for coord in point])
                            osc_client.send_message("/eye/right/contour", [coord for point in right_eye['eye_coords'] for coord in point])
                            
                            # Full iris contour points
                            osc_client.send_message("/iris/left/contour", [coord for point in left_eye['iris_coords'] for coord in point])
                            osc_client.send_message("/iris/right/contour", [coord for point in right_eye['iris_coords'] for coord in point])
                            
                            # Add pupil change messages
                            osc_client.send_message("/pupil/left/changing", int(left_pupil['is_changing']))
                            osc_client.send_message("/pupil/right/changing", int(right_pupil['is_changing']))
                            osc_client.send_message("/pupil/left/change_percentage", float(left_pupil['change_percentage']))
                            osc_client.send_message("/pupil/right/change_percentage", float(right_pupil['change_percentage']))
                            osc_client.send_message("/pupil/left/baseline", float(left_pupil['baseline_diameter']) if left_pupil['baseline_diameter'] else 0.0)
                            osc_client.send_message("/pupil/right/baseline", float(right_pupil['baseline_diameter']) if right_pupil['baseline_diameter'] else 0.0)
                            
                            # Add emotion messages
                            osc_client.send_message("/emotion/dominant", emotion_data['dominant_emotion'])
                            for emotion, score in emotion_data['emotion_scores'].items():
                                osc_client.send_message(f"/emotion/score/{emotion}", float(score))
                            osc_client.send_message("/emotion/blink_rate", float(emotion_data['blink_rate']))
                            osc_client.send_message("/emotion/eye_openness", float(emotion_data['eye_openness_ratio']))
                            
                            # Aggregated metrics
                            avg_ear = (left_eye['ear'] + right_eye['ear']) / 2
                            avg_diameter = (left_eye['iris_diameter'] + right_eye['iris_diameter']) / 2
                            total_blinks = (left_blink['blink_count'] + right_blink['blink_count']) // 2
                            
                            osc_client.send_message("/eye/average/ear", float(avg_ear))
                            osc_client.send_message("/iris/average/diameter", float(avg_diameter))
                            osc_client.send_message("/eye/total/blinks", int(total_blinks))
                            
                        except Exception as e:
                            print(f"OSC sending error: {e}")

                    # Visualization
                    # Use mesh_image instead of image for all drawing operations
                    for eye in [left_eye, right_eye]:
                        # Draw iris points in green
                        for point in eye['iris_coords']:
                            cv2.circle(mesh_image, point, 2, (0, 255, 0), -1)
                        
                        # Draw eye contour points in blue
                        for point in eye['eye_coords']:
                            cv2.circle(mesh_image, point, 2, (255, 0, 0), -1)

            # Display the image
            cv2.imshow('Iris Tracking', mesh_image)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or 'q' to quit
                break
            elif key == ord('m'):  # Toggle mesh
                cv2.setTrackbarPos('Show Mesh', 'Iris Tracking', 
                                 1 - cv2.getTrackbarPos('Show Mesh', 'Iris Tracking'))
            elif key == ord('f'):  # Toggle features
                cv2.setTrackbarPos('Show Features', 'Iris Tracking',
                                 1 - cv2.getTrackbarPos('Show Features', 'Iris Tracking'))
            elif key == ord('c'):  # Toggle contours
                cv2.setTrackbarPos('Show Contours', 'Iris Tracking',
                                 1 - cv2.getTrackbarPos('Show Contours', 'Iris Tracking'))
            elif key == ord('p'):  # Toggle pupillometry
                new_state = 1 - cv2.getTrackbarPos('Pupillometry', 'Iris Tracking')
                cv2.setTrackbarPos('Pupillometry', 'Iris Tracking', new_state)

    except KeyboardInterrupt:
        print("\nGracefully shutting down...")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()

if __name__ == "__main__":
    main()