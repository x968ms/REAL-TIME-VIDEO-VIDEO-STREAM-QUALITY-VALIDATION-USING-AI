# Real-Time Quality Validation for Streaming Data Using AI

This project performs real-time quality validation for video streams using AI, generating analytics and visualization for quality metrics.

## Features

- Real-time analysis of video streams
- Metrics for frame rate, brightness, blur, and overall quality
- Interactive visualization dashboard
- Quality alerts for problematic streams
- HTML report generation

## File Structure

- `quality_validator.py` - Main application script
- `requirements.txt` - Required Python dependencies

## Installation

1. Install the required Python packages:

```bash
pip install -r requirements.txt
```

Note: TensorFlow installation might take some time depending on your connection speed.

## Usage

```bash
# For webcam:
python quality_validator.py

# For video stream:
python quality_validator.py --source "rtsp://your_stream_url"

# For video file:
python quality_validator.py --source "path/to/video.mp4"
```

## Command Line Arguments

- `--source`: Stream source URL, camera index, or video file (default: 0 for webcam)
- `--buffer`: Number of frames to keep in history buffer (default: 100)
- `--report`: Filename for saving the quality report (default: quality_report.html)
- `--min-fps`: Minimum acceptable frame rate (default: 15.0)

## Requirements

- Python 3.6 or higher
- OpenCV
- NumPy
- Matplotlib
- TensorFlow

## How It Works

The application captures frames from the specified source and analyzes them for various quality metrics:

1. Frame Rate: Measures frames per second
2. Brightness: Analyzes overall brightness level
3. Blur Detection: Uses Laplacian variance to detect blurry frames
4. AI Quality Assessment: Uses a MobileNetV2-based model to evaluate overall quality

The results are displayed in real-time on an interactive dashboard with six panels showing different metrics. When you close the application, it generates an HTML report with statistics and recommendations.
