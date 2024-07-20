import os
import base64
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from ultralytics import YOLO

class ActionProcessImage(Action):

    def name(self):
        return "action_process_image"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: dict) -> list:

        # Directory paths
        image_dir = "D:/rasamod/images"
        output_dir = "D:/rasamod/output"

        # Get the latest image
        try:
            files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
            latest_image = max(files, key=lambda f: os.path.getmtime(os.path.join(image_dir, f)))
            image_path = os.path.join(image_dir, latest_image)
            output_path = os.path.join(output_dir, "detected.jpg")

            # Create output directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Load YOLOv8 model
            model = YOLO("D:/rasamod/model.pt")
            
            # Perform detection
            results = model(image_path)
            result = results[0]

            # Get original image dimensions
            with Image.open(image_path) as original_image:
                original_width, original_height = original_image.size

            # Save the detected image with the same dimensions as the original image
            fig, ax = plt.subplots(1, 1, figsize=(original_width / 100, original_height / 100), dpi=100)
            ax.imshow(result.plot())
            plt.axis('off')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100)
            plt.close()

            # Open and resize the detected image to match the original image size
            with Image.open(output_path) as detected_image:
                detected_image = detected_image.resize((original_width, original_height))
                detected_image.save(output_path)

            # Encode images to base64
            with open(image_path, "rb") as image_file:
                original_image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
            
            with open(output_path, "rb") as image_file:
                detected_image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

            # Log detected objects
            detected_objects = [model.names[int(cls)] for cls in result.boxes.cls]
            detected_objects_text = ", ".join(detected_objects)

            # Create the message with base64 images
            message = f"I detected the following objects in the image: {detected_objects_text}.\n\n"

            dispatcher.utter_message(
                text=message,
                image={
                    "original": original_image_base64,
                    "detected": detected_image_base64
                }
            )
            return [SlotSet("detected_objects", detected_objects_text)]

        except Exception as e:
            dispatcher.utter_message(text=f"An error occurred while processing the image: {str(e)}")
            return []
