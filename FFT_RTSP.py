import cv2
import numpy as np
import os
from datetime import timedelta
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# ----- CONFIG -----
THRESHOLD = 3200
TARGET_SIZE = (480, 270)


def compute_fft_spectrum(frame, roi_points):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray, dtype=np.uint8)
    roi_contour = np.array(roi_points, dtype=np.int32)
    cv2.fillPoly(mask, [roi_contour], 255)
    roi = cv2.bitwise_and(gray, gray, mask=mask)
    x, y, w, h = cv2.boundingRect(roi_contour)
    roi_cropped = roi[y:y + h, x:x + w]

    roi_float = np.float32(roi_cropped)
    dft = cv2.dft(roi_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft, axes=[0, 1])
    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

    return np.mean(magnitude), roi_cropped


def format_time(seconds):
    return str(timedelta(seconds=int(seconds))).replace(":", "-")


def create_segment_folder(video_num, batch_num, start_sec, end_sec):
    folder = f"Video_{video_num}_Batch_{batch_num}_{format_time(start_sec)}_to_{format_time(end_sec)}"
    os.makedirs(folder, exist_ok=True)
    return folder


def save_results_txt(time_axis, intensity_values, save_path):
    txt_file = f"{save_path}.txt"
    with open(txt_file, "w") as f:
        f.write("Time (s)\tFrequency Intensity\n")
        for t, v in zip(time_axis, intensity_values):
            f.write(f"{t:.2f}\t{v:.2f}\n")
    print(f"Saved: {txt_file}")


def save_results_html(time_axis, intensity_values, save_path):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_axis, y=intensity_values, mode='lines', name='Intensity'))
    fig.update_layout(title="Frequency Intensity Over Time",
                      xaxis_title="Time (s)",
                      yaxis_title="Intensity",
                      template="simple_white")
    html_file = f"{save_path}.html"
    fig.write_html(html_file)
    print(f"Saved: {html_file}")


def process_large_video(video_path, roi_points):
    video_num = input("Enter video number (e.g., 1): ")
    batch_num = input("Enter batch number (e.g., 2): ")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    frame_idx = 0
    in_segment = False
    segment_start = None
    segment_frames = []
    segment_time = []
    segment_intensities = []

    # --- Live graph setup ---
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], color='black')
    ax.set_title("Live Frequency Intensity")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Intensity")
    ax.set_xlim(0, 30)
    ax.set_ylim(auto=True)
    graph_time = []
    graph_intensity = []
    segment_starts = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_idx / fps
        intensity, roi_view = compute_fft_spectrum(frame, roi_points)

        # Show ROI view
        cv2.imshow("ROI View", roi_view)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Append data for live plot
        graph_time.append(current_time)
        graph_intensity.append(intensity)

        # If starting new segment, record time
        if not in_segment and intensity > THRESHOLD:
            in_segment = True
            segment_start = round(current_time, 2)
            segment_frames = []
            segment_time = []
            segment_intensities = []
            print(f"Segment START at {segment_start:.2f}s")

        elif in_segment and intensity < THRESHOLD:
            segment_end = round(current_time, 2)
            in_segment = False
            segment_duration = segment_end - segment_start
            print(f"Segment END at {segment_end:.2f}s")

            if segment_duration >= 10:
                segment_starts.append(segment_start)
                # Keep only data from last segment end onwards
                if len(segment_starts) >= 2:
                    cutoff = segment_starts[-1]  # start of current segment
                    graph_time = [t for t in graph_time if t >= cutoff]
                    graph_intensity = graph_intensity[-len(graph_time):]

                folder = create_segment_folder(video_num, batch_num, segment_start, segment_end)
                base = os.path.join(folder, os.path.basename(folder))
                save_results_txt(segment_time, segment_intensities, base)
                save_results_html(segment_time, segment_intensities, base)
            else:
                print(f"Segment duration {segment_duration:.2f}s too short. Skipped.")

            segment_frames.clear()
            segment_time.clear()
            segment_intensities.clear()

        if in_segment:
            segment_frames.append(frame)
            segment_time.append(current_time)
            segment_intensities.append(intensity)

        # Graph update every 5 frames
        if frame_idx % 5 == 0 and graph_time:
            line.set_xdata(graph_time)
            line.set_ydata(graph_intensity)
            ax.set_xlim(min(graph_time), max(graph_time) + 5)
            # Auto-scale Y-axis based on visible range
            ax.set_ylim(min(graph_intensity) - 50, max(graph_intensity) + 50)

            ax.figure.canvas.draw()
            ax.figure.canvas.flush_events()

        frame_idx += 1

    if in_segment and segment_frames:
        segment_end = round(current_time, 2)
        segment_duration = segment_end - segment_start
        print(f"Segment END at {segment_end:.2f}s (video ended)")

        if segment_duration >= 10:
            segment_starts.append(segment_start)
            if len(segment_starts) >= 2:
                cutoff = segment_starts[-2]  # keep only last and current
                indices = [i for i, t in enumerate(graph_time) if t >= cutoff]
                graph_time = [graph_time[i] for i in indices]
                graph_intensity = [graph_intensity[i] for i in indices]

            folder = create_segment_folder(video_num, batch_num, segment_start, segment_end)
            base = os.path.join(folder, os.path.basename(folder))
            save_results_txt(segment_time, segment_intensities, base)
            save_results_html(segment_time, segment_intensities, base)
        else:
            print(f"Final segment duration {segment_duration:.2f}s too short. Skipped.")

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()
    print("Processing complete.")


# --- Run ---
video_path = "P1_P3.mov"
roi_points = [(677, 1288), (1325, 1418), (1425, 1171), (893, 1051)]
process_large_video(video_path, roi_points)
