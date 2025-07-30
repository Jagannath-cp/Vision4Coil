# Vision4Coil

> ⚙️ Python-based coil inspection using FFT and YOLO object detection.

A vision-based system to detect and analyze frequency characteristics in steel coil manufacturing using real-time or video feed. The system uses FFT analysis and YOLO detection to monitor coil tail presence and save relevant visual and statistical data during significant coil motion.

---

## 📦 Installation

Create a virtual environment (optional but recommended), then install dependencies:

```bash
pip install -r requirements.txt
```

### Requirements
```
opencv-python~=4.11.0.86
numpy~=2.2.6
plotly~=6.1.2
matplotlib~=3.10.3
ultralytics~=8.3.158
pandas~=2.3.0
```

---

## 📹 Using RTSP Stream

To process a live camera feed via RTSP, edit the script to include:

```python
USERNAME = "your_username"
PASSWORD = "your_password"
CAMERA_IP = "camera_ip"  # e.g., 192.168.1.100
RTSP_URL = f"rtsp://{USERNAME}:{PASSWORD}@{CAMERA_IP}:554/cam/realmonitor?channel=1&subtype=1"
```

Then uncomment:
```python
process_rtsp_stream(RTSP_URL, roi_points)
```

---

## 📁 Output Folder Structure

When a coil motion segment is detected (lasting at least 10 seconds), a timestamped folder is created. For example:

```
2025_Jul_13-14-00-12_to_14-00-22/
├── 2025_Jul_13-14-00-12_to_14-00-22.txt        # Frequency intensity over time
├── 2025_Jul_13-14-00-12_to_14-00-22.html       # Interactive Plotly graph
├── tail_detected_0.87.jpg                      # Frame with detected tail and bounding box
└── tail_detected_0.87.json                     # YOLO detection metadata (class, confidence, bbox)
```

Multiple segments will result in multiple such folders.

---

## ⚙️ Threshold Settings

FFT intensity threshold varies by coil type and must be set manually in the script for now:

```python
THRESHOLD = 4264.8  # For DB16
# THRESHOLD = 3200  # For R5.5
# THRESHOLD = 3900  # For R8.5
```

> In future versions, a configuration file will support automatic mapping between coil thickness, threshold, and acceptable ranges.

---

## ▶️ Running the Script

To run on a saved video:

```python
video_path = "long_video.mov"
process_rtsp_stream(video_path, roi_points)
```

To run and save the log:

```
python your_script.py > output_log.txt 2>&1         # if using Linux or Windows CMD
python your_script.py *>&1 | Tee-Object -FilePath output_log.txt   # if using Windows PowerShell
```

To use RTSP, update and uncomment the RTSP block as shown above.

---

## ✨ Coming Soon

- Config file for different coil types
- Web interface for visualization
- Integration with industrial dashboard
