from ultralytics import YOLO
import cv2
import math
import torch
import time
import os
import imutils

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                "teddy bear", "hair drier", "toothbrush"
                ]

def process_counting(camera_source, camera_name):
# Initialize the video capture
    if isinstance(camera_source, str):
        # Try to open the video file first
        if os.path.isfile(camera_source):
            cap = cv2.VideoCapture(camera_source)
        # If it's not a file, it might be a webcam index or an RTSP stream
        else:
            if camera_source.isdigit():
                cap = cv2.VideoCapture(int(camera_source))
            elif camera_source.startswith('rtsp://'):
                cap = cv2.VideoCapture(camera_source)
            else:
                raise ValueError("Invalid camera source. Please provide a valid file path, an integer for webcam, or an RTSP stream URL.")
    elif isinstance(camera_source, int):
        cap = cv2.VideoCapture(camera_source)
    else:
        raise ValueError("Invalid camera source. Please provide a valid file path, an integer for webcam, or an RTSP stream URL.")

    # Check if the video capture is opened correctly
    if not cap.isOpened():
        raise Exception("Could not open video source")

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise Exception("Could not open video device")
    
    frame_width=int(cap.get(3))
    frame_height=int(cap.get(4))

    # Load the model and move it to the GPU
    model = YOLO("yolov8n.pt", task='detect')

    # Initialize variables for FPS calculation
    start_time = time.time()
    frame_counter = 0
    last_save_time = time.time()

    while True:
        # Check if the session is still active
        # if check_session_status(reference):
        #     print("Session is stopped. Stopping video processing.")
        #     break

        success, frame = cap.read()
        frame = imutils.resize(frame, width = 500)
        results=model(frame,stream=True)
        person_count = 0
        for r in results:
            boxes=r.boxes
            for box in boxes:
                x1,y1,x2,y2=box.xyxy[0]
                x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
                conf=math.ceil((box.conf[0]*100))/100
                cls=int(box.cls[0])
                class_name=classNames[cls]
                if class_name == "person":
                    person_count += 1
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,255),3)
                    label=f'{class_name}{conf}'
                    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                    c2 = x1 + t_size[0], y1 - t_size[1] - 3
                    cv2.rectangle(frame, (x1,y1), c2, [255,0,255], -1, cv2.LINE_AA)  # filled
                    cv2.putText(frame, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)
        # Draw a black rectangle at the top of the image
        cv2.rectangle(frame, (8, 7), (270, 30), (0, 0, 0), -1)

        # Write the text on the black rectangleS
        cv2.putText(frame, f'Number of people: {person_count}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Calculate and display FPS
        frame_counter += 1
        elapsed_time = time.time() - start_time
        fps = frame_counter / elapsed_time
        cv2.putText(frame, f'FPS: {fps:.0f}', (frame_width - 100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Check if 5 minutes have passed since the last save
        # current_time = time.time()
        # if current_time - last_save_time >= 1 * 60:  # 5 minutes in seconds
        #     # Save the number of people detected to the database
        #     save_to_challenges_log(reference, person_count, camera_name)
        #     last_save_time = current_time  # Update the last save time

        #out.write(frame)
        cv2.imshow(f"{camera_name}", frame)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    #out.release()
cv2.destroyAllWindows()

process_counting("0", "webcam")