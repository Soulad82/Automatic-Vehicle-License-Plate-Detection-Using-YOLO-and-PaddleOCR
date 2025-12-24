import os
import uuid
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for
from ultralytics import YOLO
from paddleocr import PaddleOCR

# ----------------- CONFIG -----------------
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# ----------------- MODELS -----------------
vehicle_model = YOLO('yolov8n.pt')  # coco weights
lp_model = YOLO(r'runs/detect/train5/weights/best.pt')
ocr_model = PaddleOCR(use_textline_orientation=True, lang='en')

vehicle_class_ids = [2, 5, 7]  # car, bus, truck


# ----------------- UTIL FUNCS -----------------
def crop_img_cv(img_cv, box):
    x1, y1, x2, y2 = map(int, box)
    return img_cv[y1:y2, x1:x2]


def resize_plate(img, target_size=(128, 64)):
    return cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)


def image_sharpness(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def get_centroid(box):
    x1, y1, x2, y2 = map(int, box)
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def match_vehicle(prev_centroids, curr_centroid, thresh=60):
    for idx, c in prev_centroids.items():
        dist = np.linalg.norm(np.array(c) - np.array(curr_centroid))
        if dist < thresh:
            return idx
    return None


# ----------------- CORE PROCESSING -----------------
def process_video(video_path, output_dir):
 
    cap = cv2.VideoCapture(video_path)
    target_width, target_height = 1280, 720
    frame_count = 0
    sampling_interval = 5

    best_outputs = {}   # car_id : dict with crop, text, score, sharpness
    prev_centroids = {}  # car_id : centroid
    next_car_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        if w > target_width or h > target_height:
            frame = cv2.resize(frame, (target_width, target_height),
                               interpolation=cv2.INTER_LINEAR)

        frame_count += 1
        do_ocr_crop = (frame_count % sampling_interval == 0)

        vehicle_boxes = []
        vehicle_results = vehicle_model(frame)
        for r in vehicle_results:
            boxes = r.boxes.xyxy
            classes = r.boxes.cls
            for box, cls in zip(boxes, classes):
                if int(cls) in vehicle_class_ids:
                    vehicle_boxes.append(box)

        new_centroids = {}
        for vbox in vehicle_boxes:
            vehicle_crop = crop_img_cv(frame, vbox)
            car_sharpness = image_sharpness(vehicle_crop)

            centroid = get_centroid(vbox)
            car_id = match_vehicle(prev_centroids, centroid)
            if car_id is None:
                car_id = next_car_idx
                next_car_idx += 1
            new_centroids[car_id] = centroid

            lp_results = lp_model(vehicle_crop)
            for plate in lp_results:
                if plate.boxes.xyxy.shape[0] == 0:
                    continue

                lp_box = plate.boxes.xyxy[0]
                x1, y1, x2, y2 = map(int, lp_box)
                vx1, vy1, _, _ = map(int, vbox)
                abs_box = [x1 + vx1, y1 + vy1, x2 + vx1, y2 + vy1]

                if do_ocr_crop:
                    plate_crop = crop_img_cv(frame, abs_box)
                    plate_resized = resize_plate(plate_crop, target_size=(128, 64))

                    # OCR on resized plate
                    result = ocr_model.predict(plate_resized)

                    best_text, best_score = '', 0.0
                    if isinstance(result, list):
                        for res_item in result:
                            texts = res_item.get('rec_texts', [])
                            scores = res_item.get('rec_scores', [])
                            for text, score in zip(texts, scores):
                                if score > best_score:
                                    best_text = text.strip()
                                    best_score = score
                    elif isinstance(result, dict) and 'rec_texts' in result:
                        for text, score in zip(result['rec_texts'], result['rec_scores']):
                            if score > best_score:
                                best_text = text.strip()
                                best_score = score

                    prev = best_outputs.get(car_id)
                    update = False
                    if prev is None:
                        update = True
                    else:
                        if best_score > prev['score']:
                            update = True
                        elif best_score == prev['score'] and car_sharpness > prev['sharpness']:
                            update = True

                    if best_text and update:
                        best_outputs[car_id] = {
                            'crop': plate_resized,  # store plate image
                            'text': best_text,
                            'score': best_score,
                            'sharpness': car_sharpness
                        }

        prev_centroids = new_centroids

    cap.release()

    # Save outputs and build JSON-friendly list
    results = []
    for car_id, info in best_outputs.items():
        # using uuid to avoid name clashes
        filename = f"plate_{car_id}_{uuid.uuid4().hex[:8]}.jpg"
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, info['crop'])

        image_url = url_for('static', filename=f"outputs/{filename}")
        results.append({
            "image_url": image_url,
            "text": info['text']
        })

    return results


# ----------------- ROUTES -----------------
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/process_video', methods=['POST'])
def process_video_route():
    """
    Receives video upload, runs detection, returns JSON:
    { "plates": [ { "image_url": "...", "text": "..."}, ... ] }
    """
    file = request.files.get('video')
    if not file or file.filename == '':
        return jsonify({"error": "No video file uploaded"}), 400

    # Save uploaded video
    video_id = uuid.uuid4().hex
    ext = os.path.splitext(file.filename)[1]
    video_filename = f"{video_id}{ext}"
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
    file.save(video_path)

    # Run processing
    plates = process_video(video_path, app.config['OUTPUT_FOLDER'])

    # Optionally delete video after processing to save space
    # os.remove(video_path)

    return jsonify({"plates": plates})


if __name__ == '__main__':
    # For dev only; set host to '0.0.0.0' if you want LAN access
    app.run(debug=True)
