import cv2
import numpy as np
import pyttsx3
import os
import speech_recognition as sr
import webbrowser
datetime

# Initialize text-to-speech engine
engine = pyttsx3.init()
recognizer = sr.Recognizer()

# Load face classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize video capture
cap = cv2.VideoCapture(0)

# Load face recognizer if training data exists
recognizer_path = "recognizer/TrainingData.yml"
if os.path.exists(recognizer_path):
    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.read(recognizer_path)
else:
    print("Warning: No training data found. Running in guest mode.")
    rec = None

# Font for text display
font = cv2.FONT_HERSHEY_SIMPLEX

# Load user data
user = {}
data_file = "datatext.txt"
if os.path.exists(data_file):
    with open(data_file, "r") as f:
        for line in f:
            parts = line.strip().split(" ", 2)
            if len(parts) == 3:
                key, name, branch = parts
                user[key] = (name, branch)

# Function to get voice input with 2 attempts
def get_voice_input(prompt):
    """Gets voice input from the user with a maximum of 2 attempts, then falls back to manual input."""
    for attempt in range(2):
        engine.say(prompt)
        engine.runAndWait()
        print(f"Attempt {attempt + 1}: {prompt}")
        
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            try:
                audio = recognizer.listen(source, timeout=5)
                response = recognizer.recognize_google(audio).strip().lower()
                return response if response else None
            except (sr.UnknownValueError, sr.RequestError, sr.WaitTimeoutError):
                print("Could not understand. Try again.")
    
    print("Voice input failed. Please type your response.")
    engine.say("Voice input failed. Please type your response.")
    engine.runAndWait()
    return input("Enter response manually: ").strip().lower()

# Function to open a website
def open_website(site):
    sites = {
        "google": "https://www.google.com",
        "youtube": "https://www.youtube.com",
        "instagram": "https://www.instagram.com",
        "svce": "https://svcengg.edu.in/"
    }
    webbrowser.open(sites.get(site, f"https://www.google.com/search?q={site}"))

# Welcome message
today = datetime.datetime.today().strftime('%A')
engine.say(f"Hi, welcome to SVCE College. Today is {today}.")
engine.runAndWait()

while True:
    status, img = cap.read()
    if not status:
        continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(100, 100))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        if rec:
            id, conf = rec.predict(gray[y:y + h, x:x + w])
            print(f"Detected ID: {id}, Confidence: {conf:.2f}")
            
            if conf < 60:
                user_info = user.get(str(id), ("Unknown", "Unknown"))
                name, branch = user_info
            else:
                id = None
                name, branch = "Unknown", "Unknown"
        else:
            id = None
            name, branch = "Guest", "Visitor"

        if name == "Unknown":
            name = get_voice_input("I don't recognize you. What's your name?")
            if not name:
                continue
            
            branch = get_voice_input(f"Which branch are you from, {name}?")
            if not branch:
                continue
            
            # Assign new ID for unknown users
            if id is None:
                id = max(map(int, user.keys()), default=1000) + 1
            
            user[str(id)] = (name, branch)
            with open(data_file, "a") as f:
                f.write(f"{id} {name} {branch}\n")
            
            engine.say(f"Thank you, {name} from {branch} branch! Your details are recorded.")
            engine.runAndWait()
            
            # Automatically capture and save unknown face
            unknown_face_path = f"unrecognized_faces/{id}_{name}.jpg"
            os.makedirs("unrecognized_faces", exist_ok=True)
            cv2.imwrite(unknown_face_path, img[y:y+h, x:x+w])
            print(f"Saved unrecognized face as {unknown_face_path}")
            
            # Open college website automatically
            open_website("svce")
            engine.say("Follow for latest updates.")
            engine.runAndWait()

        # Display user info
        cv2.putText(img, f"{name} - {branch}", (x, y - 10), font, 0.8, (0, 255, 0), 2)
    
    # Display camera feed
    cv2.imshow('Face Recognition', img)
    
    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()