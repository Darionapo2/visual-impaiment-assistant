import pyttsx3

engine = pyttsx3.init()

def read(text):
    engine.say(text)
    engine.runAndWait()