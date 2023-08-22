import cv2
import numpy as np

# Load YOLOv3 weights and configuration files
net = cv2.dnn.readNet("yolov3-custom_7000.weights", "yolov3-custom.cfg")

# Define the names of the output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

# Load the video file
cap = cv2.VideoCapture("video.mp4")

# Define the font and colors for drawing the object labels
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(output_layers), 3))

# Process each frame of the video
while True:
    # Read the frame from the video
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize the frame to a size that is compatible with YOLO
    height, width, channels = frame.shape
    resized = cv2.resize(frame, (416, 416))
    
    # Convert the frame to a blob that YOLO can process
    blob = cv2.dnn.blobFromImage(resized, scalefactor=0.00392, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
    
    # Pass the blob through the network and obtain the detections
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    # Process each detection and draw the object labels
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Draw the bounding box and object label
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                colors = [(0, 255, 255), (0, 0, 255), (255, 0, 0)]
                # print(class_id)
                cv2.rectangle(frame, (x, y), (x + w, y + h), colors[class_id], 2)
                label = f"{class_id}"
                cv2.putText(frame, label, (x, y + 30), font, 3, colors[class_id], 3)
    
    # Display the processed frame
    cv2.imshow("Output", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the resources and close the windows
cap.release()
cv2.destroyAllWindows()
