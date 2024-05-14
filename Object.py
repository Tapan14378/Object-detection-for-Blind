import cv2
import numpy as np
import pyttsx3
import speech_recognition as sr
import threading
import pygame  # for playing beep sound

# Load pre-trained YOLO model
net = cv2.dnn.readNet("yolov7-tiny.weights", "yolov7-tiny.cfg")
classes = []

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Set up text-to-speech engine
engine = pyttsx3.init()

# Set up speech recognition
recognizer = sr.Recognizer()

# Global variable to control the object detection loop
detect_objects_flag = False
object_to_find = ""

# Initialize pygame mixer for playing beep sound
pygame.mixer.init()
beep_sound = pygame.mixer.Sound('beep-07.wav')

# Function to perform object detection
def detect_objects(frame):
    global object_to_find
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    focal_length = 1000 # Replace this with your camera's focal length
    object_width = 50 # Replace this with the actual width of the object you are looking for
    beep_played = False # Initialize the variable
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == object_to_find:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                distance = (focal_length * object_width) / w # Calculate the distance using the formula
                cv2.putText(frame, f"{distance:.2f} cm", (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # Display the distance on the image


    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{object_to_find}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Play beep sound when object is detected
            if not beep_played: # Check if the beep sound has been played
               beep_sound.play() # Play the beep sound
               beep_played = True # Set the variable to True


# Function to speak
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Function to recognize speech
def listen():
    global detect_objects_flag, object_to_find
    while True:
        with sr.Microphone() as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            # Set the timeout to 5 seconds
            audio = recognizer.listen(source, timeout=2)

        try:
            print("Recognizing...")
            query = recognizer.recognize_google(audio)
            print(f"User: {query}")
            if "find" in query:
                object_to_find = query.split("find ")[1]
                speak(f"Object to find: {object_to_find}")
                detect_objects_flag = True
            elif "stop" in query:
                speak("Object detection stopped.")
                detect_objects_flag = False
            else:
                speak("Command not recognized. Please try again.")

        except sr.UnknownValueError:
            print("Sorry, I did not get that. Please try again.")
        except sr.WaitTimeoutError:
            print("No speech detected. Please try again.")

# Start the listening thread
listen_thread = threading.Thread(target=listen)
listen_thread.start()

# Main loop for object detection
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    if detect_objects_flag:
        detect_objects(frame)
        cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
