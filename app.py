import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

import warnings

# Suppress torchvision warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

st.title("Object Detection with Faster RCNN")
classes = {1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
           10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person', 16: 'pottedplant',
           17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}

# load a model, pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)

WEIGHTS_FILE = "faster_rcnn_state.pth"

num_classes = 21

# get the number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Load the trained weights
model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=torch.device('cpu')))

st.write("This model is trained to detect object in this list: ['aeroplane','bicycle','bird','boat','bottle','bus','car','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']. \n Therefore image contains object not in the list may not detected in this app")



# Predefined image options
predefined_images = {
    "Select an option": None,
    "Image 1": "images/t1.jpg",
    "Image 2": "images/t5.jpg",
    "Image 3": "images/t3.jpg",
    "Image 4": "images/t4.jpg",
    "Image 5": "images/t2.jpg",
    "Image 6": "images/t6.jpg",
}

# Image selection widget
selected_option = st.radio("Select an option", list(predefined_images.keys()))

# Check if a predefined image is selected
if selected_option != "Select an option":
    # Load the selected predefined image
    image_path = predefined_images[selected_option]
    image = cv2.imread(image_path)

    if image is not None:

        # Convert OpenCV image to PIL image
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        st.image(pil_image, caption=f"Selected Image: {selected_option}", use_column_width=True)

        # Run object detection
        img_tensor = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).permute(0, 3, 1, 2)

        # Set the model to evaluation mode
        model.eval()

        with torch.no_grad():
            detections = model(img_tensor)

        # Display the results
        for detection in detections:
            boxes = detection['boxes'].cpu().numpy()
            scores = detection['scores'].cpu().numpy()
            labels = detection['labels'].cpu().numpy()

            detection_threshold = 0.6 

            for i, box in enumerate(boxes):
                if scores[i] >= detection_threshold:
                    cv2.rectangle(image,
                                  (int(box[0]), int(box[1])),
                                  (int(box[2]), int(box[3])),
                                  (0, 220, 0), 2)
                    cv2.putText(image, classes[labels[i]], (int(box[0]), int(box[1]) - 5),
                                cv2.FONT_HERSHEY_COMPLEX, 0.7, (220, 0, 0), 1, cv2.LINE_AA)
                    
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(rgb_image, caption="Object Detection Result", use_column_width=True)

    else:
        st.write(f"Error: Unable to load the selected image: {selected_option}")

else:
    uploaded_file = st.file_uploader("Upload your image", type=["jpg","jpeg","png"])

    if uploaded_file is not None:
        # Read the uploaded image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), dtype=np.uint8), 1)

        if image is not None:
            # Convert OpenCV image to PIL image
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            st.image(pil_image, caption="Uploaded Image", use_column_width=True)

            # Run object detection
            img_tensor = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).permute(0, 3, 1, 2)

            # Set the model to evaluation mode
            model.eval()
            with torch.no_grad():
                detections = model(img_tensor)

            # Display the results
            for detection in detections:
                boxes = detection['boxes'].cpu().numpy()
                scores = detection['scores'].cpu().numpy()
                labels = detection['labels'].cpu().numpy()

                detection_threshold = 0.6 

                for i, box in enumerate(boxes):
                    if scores[i] >= detection_threshold:
                        cv2.rectangle(image,
                                    (int(box[0]), int(box[1])),
                                    (int(box[2]), int(box[3])),
                                    (0, 220, 0), 2)
                        cv2.putText(image, classes[labels[i]], (int(box[0]), int(box[1]) - 5),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (220, 0, 0), 1, cv2.LINE_AA)

            # Display the annotated image
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(rgb_image, caption="Object Detection Result", use_column_width=True)
        else:
            st.text("Error loading image. Please select image with format .jpg or .jpeg or .png")

    



# checking the version of the libraries
# print(np.__version__)
# print(torch.__version__)
# print(torchvision.__version__)
# print(cv2.__version__)
# print(cv2.getBuildInformation())































































