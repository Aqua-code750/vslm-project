# ==============================================================================
# Mog1 AI Desktop Voice Recognition & Speech Synthesizer Module
# ==============================================================================
import sys
from mog1_agent import Mog1Agent

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def listen_microphone():
    """Captures microphone input from standard Python libraries."""
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("\n🎙️ Listening... (Speak into your microphone now)")
            r.adjust_for_ambient_noise(source, duration=0.5)
            audio = r.listen(source, timeout=5)
            print("⏳ Transcribing speech...")
            query = r.recognize_google(audio)
            print(f"🗣️ You said: '{query}'")
            return query
    except Exception as e:
        print(f"Microphone Notice: {e}")
        return None

def speak_response(text):
    """Converts Mog1 AI text response to spoken voice audio."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 175)
        engine.say(text)
        engine.runAndWait()
    except Exception:
        pass

def start_voice_chat():
    print("=" * 65)
    print("🎙️ Mog1 AI Desktop Voice Chat Assistant")
    print("=" * 65)
    agent = Mog1Agent()

    print("\nPress ENTER to speak into microphone, or type prompt directly.")
    while True:
        try:
            inp = input("\n[Press Enter for Mic / Type prompt / 'exit']: ").strip()
            if inp.lower() == 'exit':
                break
            
            prompt = inp
            if not prompt:
                mic_text = listen_microphone()
                if mic_text:
                    prompt = mic_text
                else:
                    print("No audio captured. Try typing your prompt.")
                    continue

            print(f"\nProcessing prompt with Mog1 AI...")
            response = agent.run(prompt)
            print(f"\n🤖 Mog1 AI Response:\n{response}\n")
            
            # Speak out loud
            speak_response(response)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    start_voice_chat()
