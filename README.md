# HABITRAC
HABITRAC - HABItat TRACking and Behavioral Analysis.


This repository hosts a video analysis system utilizing the YOLOv8 model to process daily video recordings stored in folders. The system traverses each folder, applies object detection on every frame, and records detections in a CSV file. Each record includes the class, the position (x, y coordinates) of each detected object, and the exact time in HH:MM:SS format derived from the video filename. The system is designed to handle multiple classes and efficiently manages the processing of extensive video data.

Originally designed to monitor animals within enclosed spaces, this system is versatile enough to be adapted for various applications. It efficiently processes large volumes of video files, organized by day, to simplify data management and enhance usability. This approach allows for easier viewing and analysis of the data collected over extended periods.


``` python


import os
import csv
import cv2
from ultralytics import YOLO
import gc

# Initialize YOLO model
model = YOLO('dir/to/model/goes/here/yolo_model.pt')
names = model.names

# Define colour mapping for each class - you can make pretty pictures later on
class_colors = {
    'class_goes_here': (0, 255, 0),
    'other_classes_also_go_here': (0, 0, 255)
    'and_so_on': (255, 0, 0)
}

# Main directory containing subdirectories with videos
main_dir_path = 'dir/goes/here'

def extract_date_and_time_from_filename(filename):
    # File name of video pattern: video_YY_MM_DD_HH_MM_SS.mp4
    ## I had minute long videos that looked like this: video_24_01_24_11_00.mp4
    parts = filename.split('_')
    if len(parts) >= 7 and parts[0] == 'video' and parts[6].endswith('.mp4'):
        year = '20' + parts[1]  # Assuming 21st century (20xx)
        month = parts[2]
        day = parts[3]
        hours = parts[4]
        minutes = parts[5]
        date = f"{year}-{month}-{day}"
        hour = f"{int(hours):02}00"
        minute = f"{int(minutes):02}"
        return date, hour, minute
    else:
        print(f"Filename pattern not recognized: {filename}")
        return "9999-99-99", "0000", "00"  # Return high dummy values

def process_video(video_path):
    video_filename = os.path.basename(video_path)
    video_date, video_hour, video_minute = extract_date_and_time_from_filename(video_filename)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_dots = []

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, verbose=False)
        if results:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            clist = results[0].boxes.cls.cpu().numpy()

            for box, cls in zip(boxes, clist):
                class_label = names[int(cls)]
                color = class_colors.get(class_label, (0, 255, 0))

                x_center = int((box[0] + box[2]) / 2)
                y_center = int((box[1] + box[3]) / 2)

                # Calculate time in seconds
                elapsed_seconds = frame_number / fps

                video_dots.append((video_filename, x_center, y_center, color, class_label, video_date, video_hour, video_minute, int(elapsed_seconds)))

        frame_number += 1

    cap.release()
    return video_dots

def save_to_csv(data, csv_filename):
    with open(csv_filename, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "x", "y", "b", "g", "r", "class", "date", "hour", "minutes", "seconds"])
        for item in data:
            writer.writerow([item[0], item[1], item[2], *item[3], item[4], item[5], item[6], item[7], item[8]])

# Process each subdirectory
for subdir in sorted(os.listdir(main_dir_path)):
    subdir_path = os.path.join(main_dir_path, subdir)
    if os.path.isdir(subdir_path):
        subdir_history = []

        for video_file in sorted(os.listdir(subdir_path)):
            video_path = os.path.join(subdir_path, video_file)
            if os.path.isfile(video_path) and video_path.endswith('.mp4'):
                print(f"Processing {video_path}...")
                video_dots = process_video(video_path)
                subdir_history.extend(video_dots)

        # Save data for the current subdirectory to a CSV file
        if subdir_history:
            csv_filename = os.path.join(main_dir_path, f"{subdir}.csv")
            save_to_csv(subdir_history, csv_filename)
            print(f"Saved data for {subdir} to {csv_filename}")

        # Clear memory after processing each subdirectory
        del subdir_history
        gc.collect()

print("Done processing all subdirectories!")



```
