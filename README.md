# YOLO-RASA-Object-Detection-Chatbot

This repository contains a project that integrates YOLOv8 object detection with a Rasa chatbot. The chatbot processes images, detects objects using the YOLO model, and provides descriptions or information about the detected objects through conversational interactions.

## Project Structure

- **actions.py**: Custom actions for processing images and integrating YOLOv8 object detection with the Rasa chatbot.
- **domain.yml**: Configuration file defining the intents, entities, slots, actions, and responses for the Rasa chatbot.
- **data/nlu.yml**: Contains the training data for natural language understanding (NLU) with various intents and examples.
- **data/stories.yml**: Contains the training stories for the dialogue management model.
- **config.yml**: Configuration file for the Rasa pipeline and policies.
- **images/**: Directory to store input images for object detection.
- **output/**: Directory where detected images are saved.
- **model.pt**: Pre-trained YOLOv8 model for object detection.

## Key Concepts

### Intents
Intents represent the purpose or goal of a user’s input. For example:
- **greet**: User greets the chatbot.
- **ask_about_object**: User asks about an object detected in an image.

### Entities
Entities are key pieces of information extracted from a user’s input. For example:
- **object**: The name of the detected object in the image.

### Slots
Slots are used to store information extracted from user inputs or set during the conversation. For example:
- **object**: Stores the name of the detected object.

### Actions
Actions are custom functions that execute specific tasks. For example:
- **action_process_image**: Processes the image, performs object detection, and sends the detected information to the user.

### Responses
Responses define what the chatbot says in reply to user inputs. For example:
- **utter_greet**: "Hello! How can I assist you today?"
- **utter_detected_objects**: "I detected the following objects in the image: {objects}."

## Setup and Usage

1. **Clone the repository:**
   ```sh
   git clone https://github.com/arnabmanandhar/YOLO-RASA-Object-Detection-Chatbot.git
   cd YOLO-RASA-Object-Detection-Chatbot
