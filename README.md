# Automatic-Vehicle-License-Plate-Detection-Using-YOLO-and-PaddleOCR

Flask-based web application for automatic vehicle license plate detection and recognition from traffic videos using a three-tier deep learning pipeline: YOLO for vehicle detection, customized YOLO for plate localization, and PaddleOCR for character recognition.
​
Features
End-to-end ALPR pipeline: Detects vehicles, localizes license plates, and recognizes plate text from uploaded videos.

Three-tier architecture:
Tier 1: YOLO model for vehicle detection.
Tier 2: Custom YOLO model for license plate detection inside vehicle regions.
Tier 3: PaddleOCR for character recognition on cropped plate images.
​
Web interface with Flask: Simple frontend to upload a video and display results as tiles, each showing a cropped plate image with the detected text.
​
Result visualization: Best plate crop per vehicle selected across frames and served as static images with associated text
