# Eye and Iris Tracking System

A real-time eye and iris tracking system using computer vision and machine learning. The system tracks eye movements, blinks, pupil dilation, and can detect emotions based on eye behavior.

## Features

- Real-time eye and iris tracking
- Blink detection
- Pupil dilation tracking
- Emotion detection based on eye metrics
- Pupillometry analysis
- OSC (Open Sound Control) data streaming
- Interactive GUI controls

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- Python-OSC
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/eye-tracking-system.git
cd eye-tracking-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python main.py
```

### Controls

The system features an intuitive GUI with clickable buttons for all controls:

- Click "Mesh [M]" or press 'M' to toggle face mesh visualization
- Click "Features [F]" or press 'F' to toggle facial feature points
- Click "Contours [C]" or press 'C' to toggle face contours
- Click "Pupillometry [P]" or press 'P' to toggle pupillometry analysis
- Click "Zoom" or press 'Z' to cycle through zoom levels (1x to 3x)
- Press 'Q' or 'ESC' to quit

All buttons are color-coded:
- Green: Feature is active
- Gray: Feature is inactive

## Project Structure

- `main.py`: Main application entry point
- `detectors.py`: Contains detector classes for blinks, pupils, and emotions
- `visualization.py`: Visualization utilities and GUI controls
- `eye_tracking.py`: Eye tracking utilities
- `osc_communication.py`: OSC communication handler
- `pupillometry.py`: Pupillometry analysis (not included in this repository)

## OSC Communication

The system sends eye tracking data via OSC on port 12345. Available OSC messages include:

- `/iris/left/position`
- `/iris/right/position`
- `/eye/left/blink/state`
- `/eye/right/blink/state`
- `/emotion/dominant`
- And many more...

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 