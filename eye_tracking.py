"""
This module contains utilities for eye tracking and parameter calculation.
"""

def calculate_eye_parameters(landmarks, image_shape, side="left"):
    """
    Calculate various eye parameters including iris size and eye openness.
    
    Args:
        landmarks: MediaPipe face landmarks
        image_shape: Shape of the input image (height, width)
        side: Which eye to process ('left' or 'right')
    
    Returns:
        Dictionary containing eye parameters or None if calculation fails
    """
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
        iris_diameter = ((iris_center[0] - iris_edge[0])**2 + 
                        (iris_center[1] - iris_edge[1])**2) ** 0.5 * 2

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