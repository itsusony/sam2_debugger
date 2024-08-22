from flask import Flask, request, jsonify
from PIL import Image, ImageDraw
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import cv2
import base64
import io

top_path = "/sam2"

app = Flask(__name__)

checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
sam2 = build_sam2(model_cfg, checkpoint)
predictor = SAM2ImagePredictor(sam2)
mask_generator = SAM2AutomaticMaskGenerator(sam2)


def apply_mask(image, mask, color, alpha=0.5):
    if isinstance(mask, dict):
        mask = mask['segmentation']
    mask = mask.astype(bool)
    for c in range(3):
        image[:, :, c] = np.where(mask, 
                                  image[:, :, c] * (1 - alpha) + alpha * color[c] * 255, 
                                  image[:, :, c])
    return image

def draw_mask(image, mask, color, alpha=0.5, borders=True):
    if isinstance(mask, dict):
        mask = mask['segmentation']
    mask = mask.astype(np.uint8)
    masked = apply_mask(image.copy(), mask, color, alpha)

    if borders:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(masked, contours, -1, color, 2)

    return masked

def draw_point(image, x, y, color, size=5):
    cv2.circle(image, (int(x), int(y)), size, color, -1)
    return image

def straighten_mask_edges(mask, epsilon:float=0.02):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    straight_mask = np.zeros_like(mask, dtype=np.uint8)
    
    for contour in contours:
        epsilon = epsilon * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(straight_mask, [approx], 0, 1, -1)
    
    return straight_mask.astype(bool)

def process_image(image, masks, point_coords, input_labels, normalize=False, epsilon:float=0.02):
    for i, mask in enumerate(masks):
        if normalize:
            mask = straighten_mask_edges(mask, epsilon=epsilon)
        
        color = [30/255, 144/255, 255/255]  # Light blue color
        image = draw_mask(image, mask, color, alpha=0.5, borders=True)

    if point_coords is not None and input_labels is not None:
        for point, label in zip(point_coords, input_labels):
            color = (0, 255, 0) if label == 1 else (255, 0, 0)
            image = draw_point(image, point[0], point[1], color)

    return image

@app.route(f'{top_path}/predict', methods=['POST'])
def predict():
    x = request.args.get('x', type=int)
    y = request.args.get('y', type=int)
    x2 = request.args.get('x2', type=int)
    y2 = request.args.get('y2', type=int)
    pm = request.args.get('pm', type=int)

    if 'image' not in request.files:
       return jsonify({"error": "No image file provided"}), 400

    normalize = request.args.get('nml', type=int, default=0)
    epsilon = request.args.get('epsilon', type=float, default=0.02)

    image_file = request.files['image']
    image = Image.open(image_file)
    image = np.array(image.convert("RGB"))


    if x is None and y is None:
        masks = mask_generator.generate(image)
        processed_image = process_image(image, masks, None, None, normalize=bool(normalize), epsilon=epsilon)
    else:
        predictor.set_image(image)
        input_point = np.array([[x, y]])
        input_label = np.array([1])

        if x2 is not None and y2 is not None:
            if pm == 3: # box mode
                input_box = np.array([x, y, x2, y2])
                masks, _, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
            elif pm == 2: # multi points mode
                input_points = np.array([[x,y], [x2,y2]])
                input_labels = np.ones(len(input_points), dtype=int)
                masks, _, _ = predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    multimask_output=False,
                )
            processed_image = process_image(image, masks, None, None, normalize=bool(normalize), epsilon=epsilon)
        else:
            masks, _, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
            )
            processed_image = process_image(image, masks, input_point, input_label, normalize=bool(normalize), epsilon=epsilon)


        # Convert the processed image to base64
    img = Image.fromarray(processed_image)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return jsonify({"image_base64": img_str})

@app.route(f"{top_path}/ping")
def ping():
    return "pong", 200

if __name__ == '__main__':
    app.run(debug=True, port="5001")
