# Eye Tracking System

Real-time eye tracking and analysis system using MediaPipe Face Mesh, with pupillometry, emotion detection, and OSC communication.

## Features

- Real-time face and eye tracking
- Pupil size measurement and analysis
- Blink detection
- Emotion state analysis
- Interactive GUI controls
- OSC data streaming
- Oscilloscope-style visualization

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main application:
```bash
python main.py
```

### Controls

- [M] - Toggle face mesh visualization
- [F] - Toggle facial feature points
- [C] - Toggle face contours
- [P] - Toggle pupillometry analysis
- [Z] - Cycle zoom levels (1x-3x)
- ESC/Q - Quit application

## Detection Methods

### 1. Eye Tracking
- Uses MediaPipe Face Mesh for precise facial landmark detection
- Tracks 468 facial landmarks including detailed eye regions
- Calculates eye parameters:
  - Iris diameter and center position
  - Eye aspect ratio (EAR)
  - Eye height and contour points
  - Pupil position and movement

### 2. Blink Detection
- Monitors Eye Aspect Ratio (EAR) for each eye
- Parameters:
  - EAR Threshold: 0.2
  - Minimum consecutive frames: 2
  - Maximum blink duration: 0.5 seconds
- Tracks:
  - Blink state (open/closed)
  - Blink count
  - Blink duration
  - Blink rate (blinks per minute)

### 3. Pupillometry Analysis
- Real-time pupil size monitoring
- Baseline calibration period
- Measures:
  - Absolute pupil diameter
  - Relative size changes
  - Change velocity
  - Size variability
  - Baseline deviation

### 4. Emotion Detection
Based on combined analysis of:
- Blink patterns
- Pupil dynamics
- Eye openness
- Temporal patterns

Detected States:
- Surprise: Characterized by wide eyes and pupil dilation
- Focus: Stable pupil size and reduced blink rate
- Tired: Increased blink rate and reduced eye openness
- Relaxed: Stable measurements and normal blink rate
- Stressed: Fluctuating pupil size and irregular blink patterns

## OSC Communication

The system streams real-time data via OSC protocol (default: localhost:12345)

### OSC Parameters

#### Basic Eye Parameters
```
/iris/left/position          [x, y]     Iris center coordinates
/iris/right/position         [x, y]     Iris center coordinates
/iris/left/diameter         float      Iris diameter in pixels
/iris/right/diameter        float      Iris diameter in pixels
/iris/average/diameter      float      Average iris diameter
```

#### Eye Metrics
```
/eye/left/height           float      Eye opening height
/eye/right/height          float      Eye opening height
/eye/left/ear             float      Eye aspect ratio
/eye/right/ear            float      Eye aspect ratio
/eye/average/ear          float      Average eye aspect ratio
```

#### Blink Detection
```
/eye/left/blink/state      int        0: open, 1: closed
/eye/right/blink/state     int        0: open, 1: closed
/eye/left/blink/count      int        Cumulative blink count
/eye/right/blink/count     int        Cumulative blink count
/eye/left/blink/duration   int        Current blink duration (frames)
/eye/right/blink/duration  int        Current blink duration (frames)
/eye/total/blinks         int        Total blink count
```

#### Pupil Analysis
```
/pupil/left/changing       int        0: stable, 1: changing
/pupil/right/changing      int        0: stable, 1: changing
/pupil/left/change_percentage   float    % change from baseline
/pupil/right/change_percentage  float    % change from baseline
/pupil/left/baseline      float      Baseline pupil size
/pupil/right/baseline     float      Baseline pupil size
```

#### Pupillometry Analysis
```
/pupillometry/status       string     Current analysis state
/pupillometry/time        float      Analysis duration (seconds)
/pupillometry/left/change  float      Left pupil % change
/pupillometry/right/change float      Right pupil % change
/pupillometry/left/variability   float    Left pupil variability
/pupillometry/right/variability  float    Right pupil variability
/pupillometry/left/velocity      float    Left change velocity
/pupillometry/right/velocity     float    Right change velocity
/pupillometry/average/change     float    Average pupil change
/pupillometry/average/variability float    Average variability
/pupillometry/average/velocity   float    Average velocity
/pupillometry/baseline/progress  float    Baseline collection progress
```

#### Emotion Analysis
```
/emotion/dominant         string     Current dominant emotion
/emotion/score/surprise   float      Surprise score (0-1)
/emotion/score/focus      float      Focus score (0-1)
/emotion/score/tired      float      Tired score (0-1)
/emotion/score/relaxed    float      Relaxed score (0-1)
/emotion/score/stressed   float      Stressed score (0-1)
/emotion/blink_rate      float      Blinks per minute
/emotion/eye_openness    float      Relative eye openness
```

#### Contour Data
```
/eye/left/contour        [x,y,...]   Eye contour points
/eye/right/contour       [x,y,...]   Eye contour points
/iris/left/contour       [x,y,...]   Iris contour points
/iris/right/contour      [x,y,...]   Iris contour points
```

## Project Structure

```
├── main.py              # Main application
├── detectors.py         # Detection classes
├── visualization.py     # Visualization utilities
├── eye_tracking.py      # Eye tracking functions
├── pupillometry.py      # Pupillometry analysis
├── osc_communication.py # OSC handling
└── requirements.txt     # Dependencies
```


