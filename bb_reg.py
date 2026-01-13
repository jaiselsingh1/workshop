import torch 
import torchvision 
from torchvision import transforms as T
import cv2 

# VGG-16 model is a convolutional neural network (CNN) architecture that was proposed by the Visual Geometry Group (VGG) at the University of Oxford
# https://www.geeksforgeeks.org/computer-vision/vgg-16-cnn-model/

model = torchvision.models.detection.ssd300_vgg16(pretrained = True)
model.eval()

classnames = []
with open("content/classnames.txt") as f:
    classnames = f.read().splitlines()

def load_image(image_path: str):
    image = cv2.imread(image_path)
    return image

def transform_image(image):
    img_transform = T.ToTensor()
    image_tensor = img_transform(image)
    return image_tensor 

# It performs inference with the model, filters the predicted bounding boxes, scores and labels based on a confidence threshold (default is 0.80) and returns the filtered results.
# The filtered results include bounding boxes (filtered_bbox), corresponding scores (filtered_scores) and class labels (filtered_labels)

def detect_objects(model, image_tensor, conf_threshold = 0.80):
    with torch.no_grad():
        y_pred = model([image_tensor])
    
    bbox, scores, labels = y_pred[0]["boxes"], y_pred[0]["scores"], y_pred[0]["labels"]
    # the squeeze essentially makes it into a 1d vector 
    indices = torch.nonzero(scores > conf_threshold).squeeze(1)

    filtered_bbox = bbox[indices]
    filtered_scores = scores[indices]
    filtered_labels = labels[indices]

    return filtered_bbox, filtered_scores, filtered_labels

def draw_boxes_and_labels(image, bbox, labels, class_names):
    img_copy = image.copy()

    for i in range(len(bbox)):
        x, y, w, h = bbox[i].numpy().astype("int")
        cv2.rectangle(img_copy, (x, y), (w, h), (0, 0, 255), 5)
        class_index = labels[i].numpy().astype('int')
        class_detected = class_names[class_index - 1]
        cv2.putText(img_copy, class_detected, (x, y + 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)

    return img_copy

from google.colab.patches import cv2_imshow 
image_path = "content/mandog.jpg"
# transform image 
img = load_image(image_path)
img_tensor = transform_image(img)
bbox, scores, labels = detect_objects(model, img_tensor)
result_img = draw_boxes_and_labels(img, bbox, labels, classnames)
cv2_imshow(result_img)