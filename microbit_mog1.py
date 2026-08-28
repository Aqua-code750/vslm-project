# ==============================================================================
# Mog1 AI MicroPython Client with Voice & Speaker for BBC micro:bit (v1 & v2)
# ==============================================================================
# Flash this script to your micro:bit using https://python.microbit.org or Mu Editor.
#
# Supports:
# 🎙️ Voice Microphone Activation (Clap / Loud Speech)
# 🔊 Speaker Speech Audio Output
# 🕹️ Buttons A & B + Shake Gestures
# 📺 5x5 LED Matrix Graphics & Text
# ==============================================================================

from microbit import *
import time

display.show(Image.HAPPY)
sleep(1000)
display.scroll("Mog1 Voice AI")

# Set up USB Serial baud rate
uart.init(baudrate=115200)

def show_ai_icon(icon_name):
    if icon_name == "HAPPY":
        display.show(Image.HAPPY)
    elif icon_name == "SAD":
        display.show(Image.SAD)
    elif icon_name == "SURPRISED":
        display.show(Image.SURPRISED)
    elif icon_name == "YES":
        display.show(Image.YES)
    elif icon_name == "NO":
        display.show(Image.NO)
    elif icon_name == "HEART":
        display.show(Image.HEART)
    elif icon_name == "MIC":
        display.show(Image.MUSIC_QUAVER)
    else:
        display.show(Image.PACMAN)

while True:
    # 🎙️ Voice Microphone Activation (Clap / Loud Speech on micro:bit v2)
    try:
        if microphone.was_event(SoundEvent.LOUD):
            show_ai_icon("MIC")
            uart.write("PROMPT:VOICE_ACTIVATED\n")
            sleep(500)
    except Exception:
        pass

    # Button A: Ask Mog1 AI for a smart motivation quote
    if button_a.was_pressed():
        display.show(Image.ARROW_W)
        uart.write("PROMPT:Give a 1-sentence motivation quote for microbit\n")
        sleep(500)

    # Button B: Voice Recognition trigger or system info
    if button_b.was_pressed():
        show_ai_icon("MIC")
        uart.write("PROMPT:VOICE_ACTIVATED\n")
        sleep(500)

    # Shake: Trigger AI gesture event
    if accelerometer.was_gesture('shake'):
        display.show(Image.CONFUSED)
        uart.write("PROMPT:What happens when you shake a microbit?\n")
        sleep(500)

    # Read response from Mog1 AI Serial Bridge
    if uart.any():
        line = uart.readline()
        if line:
            msg = line.decode('utf-8').strip()
            if msg.startswith("ICON:"):
                show_ai_icon(msg.split(":")[1].strip())
            elif msg.startswith("SPEAK:"):
                speech_text = msg.split("SPEAK:")[1].strip()
                # Try micro:bit v2 built-in speaker speech synthesizer
                try:
                    import speech
                    speech.say(speech_text)
                except Exception:
                    display.scroll(speech_text)
            elif msg.startswith("TEXT:"):
                display.scroll(msg.split("TEXT:")[1].strip())
            else:
                display.scroll(msg)

    sleep(100)
