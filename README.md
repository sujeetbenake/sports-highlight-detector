# Real-Time Sports Highlights Detector · 运动帧影

A Python application that automatically detects and extracts highlights from sports videos using OpenCV, YOLOv8 object detection, and sport-specific action analysis. Supports **badminton**, basketball, and soccer highlights.

## Features

- 🎥 **Video Processing**: Process pre-recorded videos to extract highlights
- 🏸 **Badminton Detection**: 
  - Smash detection (arm speed + shuttle speed burst)
  - Rally counting (shuttle back-and-forth)
  - Scoring detection (shuttle landing + player stop)
  - Dive save detection (rapid body descent)
- 🏀 **Basketball Detection**: Fast breaks, scoring opportunities, player movements
- ⚽ **Soccer Detection**: Ball tracking, shots on goal, counter attacks, player clustering
- 🎯 **Advanced Object Detection**: YOLOv8-based detection of players, shuttlecocks, rackets
- ⚡ **Optimized Performance**: 
  - Frame skipping for faster processing
  - Lower resolution detection for efficiency
- ⏱️ **Intelligent Highlight Timing**: 
  - Pre-roll capture (5.0s before action starts)
  - Post-action recording (8.0s after action ends)
  - Smart highlight merging for continuous action
- 🎯 **Configurable Parameters**: Adjust sport-specific thresholds and timing
- 🔍 **Debug Mode**: Visualize sport-specific detection in real-time
- 📹 **Multiple Codec Support**: Automatically tries different video codecs for compatibility
- 🔄 **Highlight Preview**: Instantly replay the last detected highlight

## Installation

1. Clone the repository:
```bash
git clone https://github.com/1leonleizi3/sports-highlight-detector.git
cd sports-highlight-detector
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: The first run will automatically download the YOLOv8 model weights.

## Usage

### Processing a Video File
```bash
# For badminton
python detector.py --mode video --file path/to/your/video.mp4 --sport badminton

# For basketball
python detector.py --mode video --file path/to/your/video.mp4 --sport basketball

# For soccer
python detector.py --mode video --file path/to/your/video.mp4 --sport soccer
```

### Standalone Badminton Detection
```bash
python sport_detection.py --video path/to/your/video.mp4 --mode video
```

### Debug Mode (with sport-specific visualization)
```bash
python detector.py --mode video --file path/to/your/video.mp4 --sport badminton --debug
```

### Parameters
The following parameters can be adjusted in `detector.py`:

```python
# Timing Parameters
HIGHLIGHT_COOLDOWN = 3.0      # seconds between highlights
MIN_HIGHLIGHT_DURATION = 15.0 # minimum duration of a highlight
MAX_HIGHLIGHT_DURATION = 45.0 # maximum duration of a highlight
PRE_ROLL_SECONDS = 5.0        # seconds to keep before action starts
POST_ACTION_SECONDS = 8.0     # seconds to keep recording after action ends
MERGE_GAP_THRESHOLD = 2.0     # seconds - merge highlights if gap is smaller than this

# Badminton-Specific Parameters (in badminton_detection.py)
confidence_threshold = 0.3    # detection confidence
action_threshold = 0.3        # action trigger threshold
min_players = 2               # minimum players for rally
```

### Controls
- Press `q` to quit the application
- Press `r` to replay the last highlight
- In debug mode:
  - Sport-specific detection visualization
  - Action type display
  - Real-time statistics

## Project Structure

```
Sports-Highlight-Detector/
├── detector.py                # Main detection runner with highlight pipeline
├── badminton_detection.py     # Badminton-specific detection engine
├── sport_detection.py         # Integrated sport detection with motion analysis
├── motion_detection.py        # Motion-based frame analysis
├── object_detection.py        # Object detection utilities
├── requirements.txt           # Python dependencies
└── output/                    # Generated highlight clips
```

## How It Works

1. **Frame Processing**: The application processes video frames at a configurable sample rate
2. **Sport-Specific Detection**: 
   - **Badminton**: Detects smashes, rallies, diving saves, and scoring moments
   - Basketball: Analyzes player movements, court regions, and scoring opportunities
   - Soccer: Tracks ball movement, player interactions, goal opportunities
3. **Action Analysis**: Identifies significant plays based on sport-specific criteria
4. **Highlight Detection**: 
   - Starts recording when significant action is detected
   - Includes pre-roll frames for context
   - Continues recording during active play
   - Merges nearby highlights with small gaps
   - Adds post-action frames to complete the play
5. **Clip Saving**: Automatically saves highlights when action ends or max duration is reached

## Output

Highlights are saved in the `output/clips` directory with timestamps in the filename:
```
output/
  └── clips/
      ├── highlight_1234567890.mp4
      ├── highlight_1234567891.mp4
      └── ...
```

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- PyTorch
- Ultralytics (YOLOv8)
- See `requirements.txt` for complete list

## License

This project is licensed under the MIT License - see the LICENSE file for details.
