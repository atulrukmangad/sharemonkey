import cv2, time
# from openai import OpenAI
import pyttsx3
import base64
import numpy as np
import speech_recognition as sr

openai = OpenAI(api_key="sk-proj-sogPZruuyxbtyrNswg8fT3BlbkFJRdNNQ4EWd0EQFW59xcsN")


def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def generate_compliment_from_image(image_data):
    try:
        message = {
            "role":"user",
            "content":"Give me a compliment"
        }
        res = openai.chat.completions.create(
            messages=[message],
            model="gpt-3.5-turbo")
                
        return res.choices[0].message.content
    except Exception as e:
        print(f"Error querying OpenAI: {e}")
    return "Hey...dont just pass by...see what we have for you here...."

if __name__ == "__main__":
    capture = cv2.VideoCapture(0)  
    if not capture.isOpened():
        print("Error: Could not open camera.")
        exit()
    prev_time = time.time()

    while True:
        ret, frame = capture.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        cv2.imshow('Camera Feed', frame)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                roi = frame[y:y+h, x:x+w]
                _, img_encoded = cv2.imencode('.jpg', roi)
                image_data = base64.b64encode(img_encoded).decode('utf-8')

                compliment = generate_compliment_from_image(image_data)
                print(f"Compliment for this person: {compliment}")
                text_to_speech(compliment)

                while True:
                    response = input("Say something:") # We have to change this to Speech when PyAudio is fixed
                    if response:
                        if response == "bye":
                            break
                        response_message = response
                        print(f"Bot: {response_message}")
                        text_to_speech(response_message)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()
