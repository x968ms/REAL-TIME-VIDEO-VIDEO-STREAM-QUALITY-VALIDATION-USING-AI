"""
REAL-TIME QUALITY VALIDATION FOR STREAMING DATA USING AI
--------------------------------------------------------
This project performs real-time quality validation for video streams using AI,
generating analytics and visualization for quality metrics.

Main components:
1. Stream capture and processing
2. AI-based quality assessment
3. Real-time visualization of quality metrics
4. Quality alert system
"""

import cv2
import numpy as np
import time
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import os
import datetime
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

class StreamQualityValidator:
    def __init__(self, source, buffer_size=100, resolution_threshold=(480, 360),
                 frame_rate_threshold=15, brightness_range=(40, 240), blur_threshold=100):
        """
        Initialize the Stream Quality Validator.
        
        Args:
            source: Video source (URL or camera index)
            buffer_size: Number of frames to keep history for
            resolution_threshold: Minimum acceptable resolution
            frame_rate_threshold: Minimum acceptable frame rate
            brightness_range: Acceptable brightness range
            blur_threshold: Threshold for blur detection (lower means more sensitive)
        """
        self.source = source
        self.buffer_size = buffer_size
        self.resolution_threshold = resolution_threshold
        self.frame_rate_threshold = frame_rate_threshold
        self.brightness_range = brightness_range
        self.blur_threshold = blur_threshold
        
        # Metrics storage
        self.frame_rates = deque(maxlen=buffer_size)
        self.brightness_values = deque(maxlen=buffer_size)
        self.blur_scores = deque(maxlen=buffer_size)
        self.quality_scores = deque(maxlen=buffer_size)
        self.timestamps = deque(maxlen=buffer_size)
        self.resolution_history = deque(maxlen=buffer_size)
        
        # Status flags
        self.is_running = False
        self.current_frame = None
        self.frame_count = 0
        
        # Initialize AI model for quality assessment
        self.quality_model = self._build_quality_model()
        
        # Visualization
        self.fig = plt.figure(figsize=(15, 10))
        self.fig.suptitle("REAL-TIME QUALITY VALIDATION", fontsize=16)
        self.fig.canvas.manager.set_window_title('Stream Quality Analytics')
        self.setup_plots()
        
    def _build_quality_model(self):
        """Build a simple model for quality assessment based on MobileNetV2"""
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(128, activation='relu')(x)
        output = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=output)
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
            
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
        
    def setup_plots(self):
        """Set up the visualization plots"""
        # Create subplots
        self.ax1 = self.fig.add_subplot(2, 3, 1)  # Frame rate
        self.ax2 = self.fig.add_subplot(2, 3, 2)  # Brightness
        self.ax3 = self.fig.add_subplot(2, 3, 3)  # Blur detection
        self.ax4 = self.fig.add_subplot(2, 3, 4)  # Overall quality
        self.ax5 = self.fig.add_subplot(2, 3, 5)  # Live frame
        self.ax6 = self.fig.add_subplot(2, 3, 6)  # Resolution history
        
        # Set titles
        self.ax1.set_title('Frame Rate (FPS)')
        self.ax2.set_title('Brightness Level')
        self.ax3.set_title('Blur Detection')
        self.ax4.set_title('Overall Quality Score')
        self.ax5.set_title('Current Frame')
        self.ax6.set_title('Resolution')
        
        # Set y-axis limits
        self.ax1.set_ylim(0, 60)
        self.ax2.set_ylim(0, 255)
        self.ax3.set_ylim(0, 500)
        self.ax4.set_ylim(0, 1)
        
        # Initialize line plots
        self.line1, = self.ax1.plot([], [], 'b-')
        self.line2, = self.ax2.plot([], [], 'g-')
        self.line3, = self.ax3.plot([], [], 'r-')
        self.line4, = self.ax4.plot([], [], 'c-')
        self.img_display = self.ax5.imshow(np.zeros((240, 320, 3), dtype=np.uint8))
        self.line6, = self.ax6.plot([], [], 'm-')
        
        # Add thresholds as horizontal lines
        self.ax1.axhline(y=self.frame_rate_threshold, color='r', linestyle='--')
        self.ax2.axhline(y=self.brightness_range[0], color='r', linestyle='--')
        self.ax2.axhline(y=self.brightness_range[1], color='r', linestyle='--')
        self.ax3.axhline(y=self.blur_threshold, color='g', linestyle='--')
        self.ax4.axhline(y=0.6, color='r', linestyle='--')
        
        # Add grid to all plots
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax6]:
            ax.grid(True)
        
    def start(self):
        """Start the video stream processing"""
        if self.is_running:
            print("Stream processing is already running")
            return
            
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._process_stream)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        # Start the animation
        self.ani = FuncAnimation(self.fig, self._update_plots, interval=100, cache_frame_data=False)
        plt.tight_layout()
        plt.show()
        
    def stop(self):
        """Stop the video stream processing"""
        self.is_running = False
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join(timeout=1.0)
        print("Stream processing stopped")
        
    def _process_stream(self):
        """Process the video stream and compute quality metrics"""
        # Open the video capture
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {self.source}")
            self.is_running = False
            return
            
        last_time = time.time()
        
        while self.is_running:
            # Read a frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to receive frame from stream")
                break
                
            self.current_frame = frame
            current_time = time.time()
            
            # Calculate metrics
            frame_rate = 1 / (current_time - last_time) if current_time != last_time else 0
            brightness = np.mean(frame)
            blur_score = self._calculate_blur(frame)
            quality_score = self._evaluate_quality(frame)
            resolution = (frame.shape[1], frame.shape[0])
            
            # Store metrics
            self.frame_rates.append(frame_rate)
            self.brightness_values.append(brightness)
            self.blur_scores.append(blur_score)
            self.quality_scores.append(quality_score)
            self.timestamps.append(current_time)
            self.resolution_history.append(resolution[0] * resolution[1] / 1000)  # In thousands of pixels
            
            # Update counters and timestamps
            self.frame_count += 1
            last_time = current_time
            
            # Optional: Add delay to reduce CPU usage
            time.sleep(0.01)
            
        # Release resources
        cap.release()
        
    def _calculate_blur(self, frame):
        """Calculate blur score using Laplacian variance method"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
        
    def _evaluate_quality(self, frame):
        """Evaluate overall quality using AI model"""
        # Prepare the frame for the model
        resized = cv2.resize(frame, (224, 224))
        normalized = resized / 255.0
        expanded = np.expand_dims(normalized, axis=0)
        
        # Ensure the input is a valid numpy array
        if not isinstance(expanded, np.ndarray):
            expanded = np.array(expanded)

        # Use a trained model for quality assessment
        brightness_score = min(1.0, max(0.0, 1.0 - abs((np.mean(resized) - 128) / 128)))
        blur_score = min(1.0, max(0.0, self._calculate_blur(resized) / 1000))

        
        # Combine with model prediction
        # Note: In a real scenario, the model would be properly trained on quality data
        model_score = self.quality_model.predict(expanded, verbose=0)[0][0]
        combined_score = 0.3 * brightness_score + 0.3 * blur_score + 0.4 * model_score
        
        return combined_score
        
    def _update_plots(self, frame):
        """Update the visualization plots with the latest data"""
        if not self.is_running or len(self.timestamps) < 2:
            return
            
        # Get the last buffer_size or fewer data points
        x_data = list(range(len(self.timestamps)))
        
        # Update line plots
        self.line1.set_data(x_data, self.frame_rates)
        self.line2.set_data(x_data, self.brightness_values)
        self.line3.set_data(x_data, self.blur_scores)
        self.line4.set_data(x_data, self.quality_scores)
        self.line6.set_data(x_data, self.resolution_history)
        
        # Update frame display
        if self.current_frame is not None:
            display_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            display_frame = cv2.resize(display_frame, (320, 240))
            self.img_display.set_array(display_frame)
        
        # Adjust x-axis limits to show all data
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax6]:
            ax.set_xlim(0, len(x_data))
            
        # Return the artists that were updated
        return self.line1, self.line2, self.line3, self.line4, self.img_display, self.line6
        
    def save_report(self, filename="quality_report.html"):
        """Save a quality report to a file"""
        if len(self.timestamps) == 0:
            print("No data available to generate report")
            return
            
        # Calculate average metrics
        avg_frame_rate = np.mean(self.frame_rates)
        avg_brightness = np.mean(self.brightness_values)
        avg_blur = np.mean(self.blur_scores)
        avg_quality = np.mean(self.quality_scores)
        
        # Determine overall status
        status = "GOOD" if (avg_frame_rate >= self.frame_rate_threshold and 
                           self.brightness_range[0] <= avg_brightness <= self.brightness_range[1] and
                           avg_blur >= self.blur_threshold and
                           avg_quality >= 0.6) else "NEEDS IMPROVEMENT"
        
        # Create HTML report
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Stream Quality Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #2c3e50; }}
                .metric {{ margin: 10px 0; padding: 10px; border-radius: 5px; background-color: #f8f9fa; }}
                .good {{ color: green; }}
                .bad {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <h1>Stream Quality Validation Report</h1>
            <p><strong>Date:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Source:</strong> {self.source}</p>
            <p><strong>Frames Analyzed:</strong> {self.frame_count}</p>
            <p><strong>Overall Status:</strong> <span class="{'good' if status == 'GOOD' else 'bad'}">{status}</span></p>
            
            <h2>Summary Metrics</h2>
            <div class="metric">
                <p><strong>Average Frame Rate:</strong> {avg_frame_rate:.2f} FPS 
                <span class="{'good' if avg_frame_rate >= self.frame_rate_threshold else 'bad'}">
                    ({avg_frame_rate >= self.frame_rate_threshold and 'GOOD' or 'LOW'})
                </span></p>
            </div>
            <div class="metric">
                <p><strong>Average Brightness:</strong> {avg_brightness:.2f} 
                <span class="{'good' if self.brightness_range[0] <= avg_brightness <= self.brightness_range[1] else 'bad'}">
                    ({self.brightness_range[0] <= avg_brightness <= self.brightness_range[1] and 'GOOD' or 'OUT OF RANGE'})
                </span></p>
            </div>
            <div class="metric">
                <p><strong>Average Blur Score:</strong> {avg_blur:.2f} 
                <span class="{'good' if avg_blur >= self.blur_threshold else 'bad'}">
                    ({avg_blur >= self.blur_threshold and 'GOOD' or 'BLURRY'})
                </span></p>
            </div>
            <div class="metric">
                <p><strong>Average Quality Score:</strong> {avg_quality:.2f} 
                <span class="{'good' if avg_quality >= 0.6 else 'bad'}">
                    ({avg_quality >= 0.6 and 'GOOD' or 'POOR'})
                </span></p>
            </div>
            
            <h2>Recommendations</h2>
            <ul>
        """
        
        # Add recommendations based on metrics
        if avg_frame_rate < self.frame_rate_threshold:
            html += "<li>Improve network connection or reduce stream resolution to increase frame rate.</li>"
        
        if not (self.brightness_range[0] <= avg_brightness <= self.brightness_range[1]):
            if avg_brightness < self.brightness_range[0]:
                html += "<li>Increase lighting in the scene as the stream appears too dark.</li>"
            else:
                html += "<li>Reduce lighting or adjust camera exposure as the stream appears too bright.</li>"
        
        if avg_blur < self.blur_threshold:
            html += "<li>Improve camera focus or stability as the stream appears blurry.</li>"
        
        if avg_quality < 0.6:
            html += "<li>General quality issues detected. Consider upgrading camera equipment or improving lighting conditions.</li>"
        
        html += """
            </ul>
            
            <h2>Detailed Metrics</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Minimum</th>
                    <th>Maximum</th>
                    <th>Average</th>
                    <th>Standard Deviation</th>
                </tr>
                <tr>
                    <td>Frame Rate (FPS)</td>
                    <td>{:.2f}</td>
                    <td>{:.2f}</td>
                    <td>{:.2f}</td>
                    <td>{:.2f}</td>
                </tr>
                <tr>
                    <td>Brightness</td>
                    <td>{:.2f}</td>
                    <td>{:.2f}</td>
                    <td>{:.2f}</td>
                    <td>{:.2f}</td>
                </tr>
                <tr>
                    <td>Blur Score</td>
                    <td>{:.2f}</td>
                    <td>{:.2f}</td>
                    <td>{:.2f}</td>
                    <td>{:.2f}</td>
                </tr>
                <tr>
                    <td>Quality Score</td>
                    <td>{:.2f}</td>
                    <td>{:.2f}</td>
                    <td>{:.2f}</td>
                    <td>{:.2f}</td>
                </tr>
            </table>
        </body>
        </html>
        """.format(
            min(self.frame_rates), max(self.frame_rates), avg_frame_rate, np.std(self.frame_rates),
            min(self.brightness_values), max(self.brightness_values), avg_brightness, np.std(self.brightness_values),
            min(self.blur_scores), max(self.blur_scores), avg_blur, np.std(self.blur_scores),
            min(self.quality_scores), max(self.quality_scores), avg_quality, np.std(self.quality_scores)
        )
        
        # Write the report to a file
        with open(filename, 'w') as f:
            f.write(html)
            
        print(f"Report saved to {filename}")


# Example class for handling alerts based on quality metrics
class QualityAlertSystem:
    def __init__(self, validator, check_interval=5.0):
        """
        Initialize the alert system.
        
        Args:
            validator: A StreamQualityValidator instance
            check_interval: How often to check metrics (in seconds)
        """
        self.validator = validator
        self.check_interval = check_interval
        self.is_running = False
        self.alert_history = []
        
    def start(self):
        """Start the alert monitoring system"""
        if self.is_running:
            print("Alert system is already running")
            return
            
        self.is_running = True
        self.alert_thread = threading.Thread(target=self._monitor_metrics)
        self.alert_thread.daemon = True
        self.alert_thread.start()
        print("Alert system started")
        
    def stop(self):
        """Stop the alert monitoring system"""
        self.is_running = False
        if hasattr(self, 'alert_thread'):
            self.alert_thread.join(timeout=1.0)
        print("Alert system stopped")
        
    def _monitor_metrics(self):
        """Monitor quality metrics and generate alerts when needed"""
        while self.is_running:
            if len(self.validator.frame_rates) > 0:
                # Check frame rate
                current_fps = self.validator.frame_rates[-1]
                if current_fps < self.validator.frame_rate_threshold:
                    self._create_alert("Low Frame Rate", f"Current FPS: {current_fps:.2f}")
                
                # Check brightness
                current_brightness = self.validator.brightness_values[-1]
                if not (self.validator.brightness_range[0] <= current_brightness <= self.validator.brightness_range[1]):
                    condition = "dark" if current_brightness < self.validator.brightness_range[0] else "bright"
                    self._create_alert(f"Brightness Issue", f"Stream too {condition} ({current_brightness:.2f})")
                
                # Check blur
                current_blur = self.validator.blur_scores[-1]
                if current_blur < self.validator.blur_threshold:
                    self._create_alert("Blur Detected", f"Blur score: {current_blur:.2f}")
                
                # Check overall quality
                current_quality = self.validator.quality_scores[-1]
                if current_quality < 0.6:
                    self._create_alert("Poor Overall Quality", f"Quality score: {current_quality:.2f}")
            
            # Wait before next check
            time.sleep(self.check_interval)
            
    def _create_alert(self, alert_type, message):
        """Create and log an alert"""
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        alert = {
            'timestamp': timestamp,
            'type': alert_type,
            'message': message
        }
        self.alert_history.append(alert)
        print(f"ALERT [{timestamp}] {alert_type}: {message}")


def main():
    """Main function to run the stream quality validator"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Real-time Stream Quality Validator')
    parser.add_argument('--source', type=str, default=r"C:\Users\VIRAJ RAY\Videos\Screen Recordings\Screen Recording 2024-07-09 215816.mp4",



                        help='Stream source URL or camera index (default: 0 for webcam)')
    parser.add_argument('--buffer', type=int, default=100,

                        help='Number of frames to keep in history buffer (default: 100)')
    parser.add_argument('--report', type=str, default='quality_report.html',

                        help='Filename for saving the quality report (default: quality_report.html)')
    parser.add_argument('--min-fps', type=float, default=15.0,

                        help='Minimum acceptable frame rate (default: 15.0)')
    args = parser.parse_args()
    
    try:
        # Create validator instance
        validator = StreamQualityValidator(
            source=args.source,
            buffer_size=args.buffer,
            frame_rate_threshold=args.min_fps
        )
        
        # Create alert system
        alert_system = QualityAlertSystem(validator)
        
        # Start the validator and alert system
        alert_system.start()
        validator.start()
        
        # Main loop runs until the visualization window is closed
        
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up and save report
        if 'validator' in locals():
            validator.stop()
            validator.save_report(args.report)
        if 'alert_system' in locals():
            alert_system.stop()
        
    print("Program terminated")


if __name__ == "__main__":
    main()
