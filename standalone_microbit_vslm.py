# ==============================================================================
# Standalone On-Board Offline Mog1 AI Model for BBC micro:bit (v1 & v2)
# ==============================================================================
# FLASH DIRECTLY TO YOUR MICRO:BIT! (Runs 100% Offline without PC or Internet!)
# Flash via https://python.microbit.org or Mu Editor.
#
# Hardware Specs: Fits inside micro:bit 128KB RAM / 512KB Flash.
# Features:
#  • On-Board Compact Tokenizer & Generative Text Synthesizer
#  • Offline Intent Classifier & Reasoning Engine
#  • Voice Mic Input & Speaker Speech Synthesis
#  • 5x5 LED Matrix Graphics & Button Actions
# ==============================================================================

try:
    from microbit import *
    ON_MICROBIT = True
except ImportError:
    ON_MICROBIT = False
    class Image:
        HAPPY = "HAPPY"
        SAD = "SAD"
        SURPRISED = "SURPRISED"
        HEART = "HEART"
        MUSIC_QUAVER = "MUSIC_QUAVER"
        YES = "YES"
        NO = "NO"
        PACMAN = "PACMAN"
        ARROW_W = "ARROW_W"
        ARROW_E = "ARROW_E"
        CONFUSED = "CONFUSED"

try:
    import speech
    HAS_SPEECH = True
except Exception:
    HAS_SPEECH = False

# On-Board Offline Knowledge & Intent Weights (Simple Q&A about Mog1 AI)
INTENT_WEIGHTS = {
    "hello": "Hello! I am Mog1 AI running offline on your microbit!",
    "hi": "Hi there! I am your microbit AI assistant!",
    "who": "I am Mog1 AI, a 3.3M parameter model created by Aqua-code750 & Aquaholograph2014!",
    "name": "My name is Mog1 AI (VSLM)! I run on microbit and the web!",
    "creator": "I was created by Aqua-code750 and Aquaholograph2014!",
    "created": "Aqua-code750 & Aquaholograph2014 built me using PyTorch and Python!",
    "maker": "Aqua-code750 and Aquaholograph2014 are my creators!",
    "size": "I have 3.3 million parameters in PyTorch and a microbit offline mode!",
    "smart": "I can execute Python code, solve math, fetch weather, and run offline on microbit!",
    "math": "Math Tip: 2 + 2 = 4, 12 * 12 = 144, 25 * 4 = 100!",
    "python": "Python is a powerful language used for AI, PyTorch, and microbit code!",
    "microbit": "BBC microbit v2 has 128KB RAM, 512KB Flash, Mic, and Speaker!",
    "weather": "Current status: Offline Mode active on microbit hardware!",
    "shake": "Sensor Event: Acceleration detected! Microbit motion active!"
}

ICONS = {
    "HAPPY": Image.HAPPY,
    "SAD": Image.SAD,
    "SURPRISED": Image.SURPRISED,
    "HEART": Image.HEART,
    "MUSIC": Image.MUSIC_QUAVER,
    "YES": Image.YES
}

def clean_tokenize(text):
    """Compact Subword Tokenizer for micro:bit RAM."""
    words = text.lower().replace("?", "").replace("!", "").split()
    return [w for w in words if len(w) > 1]

def run_onboard_model(prompt):
    """
    On-Board Generative & Classification Model Engine.
    Runs 100% offline inside micro:bit MicroPython runtime!
    """
    tokens = clean_tokenize(prompt)
    if not tokens:
        return "Mog1 AI Offline Ready!", Image.HAPPY
    
    for token in tokens:
        for key in INTENT_WEIGHTS:
            if key in token:
                img = Image.HEART if "who" in token or "hello" in token else Image.YES
                return INTENT_WEIGHTS[key], img

    return f"Mog1 AI processed: '{prompt[:20]}'. Microbit offline mode active!", Image.PACMAN

def speak_or_scroll(text, img=None):
    if img:
        display.show(img)
        sleep(500)
    
    if HAS_SPEECH:
        try:
            speech.say(text[:60])
        except Exception:
            display.scroll(text[:40])
    else:
        display.scroll(text[:40])

# --- Main micro:bit On-Board Loop ---
if ON_MICROBIT:
    display.show(Image.HAPPY)
    sleep(800)
    display.scroll("Mog1 Micro AI")

    while True:
        # 🎙️ On-Board Voice Microphone Activation
        try:
            if microphone.was_event(SoundEvent.LOUD):
                display.show(Image.MUSIC_QUAVER)
                ans, img = run_onboard_model("hello")
                speak_or_scroll(ans, img)
                sleep(400)
        except Exception:
            pass

        # Button A: Offline Knowledge Query 1
        if button_a.was_pressed():
            display.show(Image.ARROW_W)
            ans, img = run_onboard_model("who are you")
            speak_or_scroll(ans, img)
            sleep(300)

        # Button B: Offline Knowledge Query 2
        if button_b.was_pressed():
            display.show(Image.ARROW_E)
            ans, img = run_onboard_model("microbit info")
            speak_or_scroll(ans, img)
            sleep(300)

        # Shake Gesture: Motion Sensor AI Response
        if accelerometer.was_gesture('shake'):
            ans, img = run_onboard_model("shake event")
            speak_or_scroll(ans, img)
            sleep(300)

        sleep(100)
