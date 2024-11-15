import cv2
import torch
from ultralytics import YOLO
from google.colab.patches import cv2_imshow  # Import Colab's image display function
from IPython.display import clear_output

# Initialize the YOLOv8 model
model = YOLO('yolov8.pt')

# Define the video capture object
video_path = 'video.mp4'  # Replace with your video path
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Define the output video parameters
output_path = 'output_video.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

frame_count = 0
display_frequency = 30  # Display every 30th frame to avoid flooding the notebook

try:
    # Loop through each frame of the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break

        frame_count += 1

        # Run human detection model
        results = model(frame)

        # Check if results exist and have valid detections
        if results and len(results) > 0:
            # Process each detection result
            for result in results:
                if hasattr(result, 'boxes') and hasattr(result.boxes, 'data'):
                    boxes_data = result.boxes.data
                    if len(boxes_data) > 0:
                        # Extract bounding boxes, class ids, and confidence scores
                        for detection in boxes_data:
                            try:
                                x1, y1, x2, y2, score, class_id = detection[:6]

                                # Check if the detected class is a person (class id 0 in COCO dataset)
                                if int(class_id) == 0 and score > 0.5:  # Filter by confidence score > 0.5
                                    # Convert coordinates to integers
                                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                                    # Draw bounding box and label
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    label = f"Person: {score:.2f}"
                                    cv2.putText(frame, label, (x1, y1 - 10),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            except Exception as e:
                                print(f"Error processing detection: {e}")
                                continue

        # Write the frame to output video
        out.write(frame)

        # Display frame periodically to avoid overwhelming the notebook
        if frame_count % display_frequency == 0:
            clear_output(wait=True)  # Clear previous frame
            cv2_imshow(frame)  # Display frame using Colab's function
            print(f"Processing frame {frame_count}")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Release resources
    print("Cleaning up...")
    cap.release()
    out.release()
