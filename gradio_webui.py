import gradio as gr
import requests
from PIL import Image, ImageDraw
import numpy as np
import io
import base64
import re

# Global variable for API base URL
API_BASE_URL = "https://mlserver.geniee.jp/sam2"

class PointSelector:
    def __init__(self):
        self.points = []

    def select_point(self, image, evt: gr.SelectData):
        x, y = evt.index
        self.points.append((x, y))
        
        if len(self.points) == 1:
            pil_image = Image.fromarray(image.astype('uint8'), 'RGB')
            draw = ImageDraw.Draw(pil_image)
            draw.ellipse([x-5, y-5, x+5, y+5], fill="red", outline="red")
            image_with_point = np.array(pil_image)
            return f"Point selected: ({x}, {y}). Click again to select a second point or click 'Process Image' to use single point.", image_with_point
        elif len(self.points) == 2:
            x1, y1 = self.points[0]
            x2, y2 = self.points[1]
            if x2 < x1:
                xarr = [x1,x2]
                x1 = xarr[1]
                x2 = xarr[0]
            if y2 < y1:
                yarr = [y1,y2]
                y1 = yarr[1]
                y2 = yarr[0]
            pil_image = Image.fromarray(image.astype('uint8'), 'RGB')
            draw = ImageDraw.Draw(pil_image)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            image_with_box = np.array(pil_image)
            self.points = []  # Reset points for next selection
            return f"Box selected: ({x1}, {y1}) to ({x2}, {y2})", image_with_box
        else:
            return "Unexpected state. Please try again.", image

    def reset(self):
        self.points = []

def extract_coordinates(point_info):
    if not point_info or point_info == "No selection":
        return None, None, None, None
    
    # Use regular expression to extract coordinates
    single_point_match = re.search(r'\((\d+),\s*(\d+)\)', point_info)
    box_match = re.search(r'\((\d+),\s*(\d+)\)\s*to\s*\((\d+),\s*(\d+)\)', point_info)
    
    if box_match:
        return tuple(map(int, box_match.groups()))
    elif single_point_match:
        return tuple(map(int, single_point_match.groups())) + (None, None)
    else:
        # Instead of raising an exception, return None values
        return None, None, None, None

def process_image(image, point_info, normalize, epsilon):
    if image is None:
        return "Please upload an image first.", None, None
    
    try:
        # Extract coordinates from point_info
        x1, y1, x2, y2 = extract_coordinates(point_info)

        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image.astype('uint8'), 'RGB')

        # Save the image to a bytes buffer
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Make the API call
        nml_param = 1 if normalize == "Apply Normalization" else 0
        url = f"{API_BASE_URL}/predict?nml={nml_param}&epsilon={epsilon}"
        
        # Add coordinates to URL if they exist
        if x1 is not None and y1 is not None:
            url += f"&x={x1}&y={y1}"
            if x2 is not None and y2 is not None:
                url += f"&x2={x2}&y2={y2}"
        
        files = {"image": ("image.png", img_byte_arr, "image/png")}
        response = requests.post(url, files=files)

        if response.status_code == 200:
            result = response.json()
            image_base64 = result.get("image_base64")
            
            if image_base64:
                # Decode base64 string to image
                image_data = base64.b64decode(image_base64)
                result_image = Image.open(io.BytesIO(image_data))
                
                # Convert PIL Image to numpy array for Gradio
                result_numpy = np.array(result_image)
                
                # Reset the point selector state
                point_selector.reset()
                
                if x1 is None or y1 is None:
                    return "Processing complete. No point or box selected.", image, result_numpy
                elif x2 is None or y2 is None:
                    return f"Processing complete. Point coordinates: ({x1}, {y1}). Ready for new selection.", image, result_numpy
                else:
                    return f"Processing complete. Box coordinates: ({x1}, {y1}) to ({x2}, {y2})", image, result_numpy
            else:
                return f"Error: No image data in response.", image, None
        else:
            return f"API Error: {response.status_code}.", image, None

    except Exception as e:
        return f"Error: {str(e)}", image, None

point_selector = PointSelector()

with gr.Blocks() as demo:
    gr.Markdown("# Image Upload and Optional Point/Box Selection Demo")

    with gr.Row():
        image_input = gr.Image(type="numpy", label="Upload Image and Optionally Select Point/Box", height=400)
        image_with_selection = gr.Image(type="numpy", label="Image with Selection", height=400)
    
    point_info = gr.Textbox(label="Selection Information", value="No selection")

    with gr.Row():
        normalize = gr.Radio(
            ["No Normalization", "Apply Normalization"], 
            label="Edge Normalization", 
            value="No Normalization"
        )
        epsilon = gr.Slider(
            minimum=0.001, 
            maximum=0.1, 
            value=0.02, 
            step=0.001, 
            label="Normalization Strength (Epsilon)"
        )

    process_button = gr.Button("Process Image")
    result_image = gr.Image(label="Result Image", height=600, show_download_button=True)

    def reset_selection(image):
        point_selector.reset()
        return "No selection", image

    def trigger_processing(image, point_info, normalize, epsilon):
        if image is None:
            return "Please upload an image first.", None, None
        return process_image(image, point_info, normalize, epsilon)

    # Set up the event listeners
    image_input.select(
        point_selector.select_point,
        inputs=[image_input],
        outputs=[point_info, image_with_selection]
    )

    # Connect the reset_selection function to the image_input's clear event
    image_input.clear(
        reset_selection,
        inputs=[image_input],
        outputs=[point_info, image_with_selection]
    )

    # Modify the process_button click event
    process_button.click(
        trigger_processing,
        inputs=[image_input, point_info, normalize, epsilon],
        outputs=[point_info, image_with_selection, result_image]
    )

    gr.Markdown("""
    ## Instructions:
    1. Upload an image using the 'Upload Image and Optionally Select Point/Box' component.
    2. (Optional) Click on the image to select a single point, or click twice to select a box.
    3. Select whether to apply edge normalization or not.
    4. Adjust the Normalization Strength (Epsilon) if applying normalization.
    5. Click the 'Process Image' button to send the image (with or without selection) to the API and get the result.
    6. The result image will be displayed below.
    7. After processing, you can make a new selection or upload a new image.
    8. If the result image is not fully visible, you can:
        - Click on the image to open it in full size in a new tab.
        - Use the download button to save and view the full image locally.
    """)

demo.launch(server_name="0.0.0.0")
