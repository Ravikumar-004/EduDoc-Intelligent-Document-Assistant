import pyttsx3

def generate_voices(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')

    for voice in voices:
        engine.setProperty('voice', voice.id)
        engine.setProperty('rate', 145)
        print(f"Using voice: {voice.name}")
        engine.save_to_file(text, f"assets/voices/{voice.name}.wav")
        engine.runAndWait()

if __name__ == "__main__":
    text = "I'm sorry, there is an issue with our speech recognition system. Please try again later."
    generate_voices(text)
