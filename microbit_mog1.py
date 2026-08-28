# ==============================================================================
# Mog1 AI MicroPython Client for BBC micro:bit (v1 & v2)
# ==============================================================================
# Flash this script to your micro:bit using the online MicroPython editor 
# (https://python.microbit.org) or Mu Editor.
#
# It sends button clicks and sensor data over Serial (USB) to Mog1 AI 
# and displays smart responses and icons on the 5x5 LED Matrix!
# ==============================================================================

from microbit import *
import time

display.show(Image.HAPPY)
sleep(1000)
display.scroll("Mog1 AI")

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
    else:
        display.show(Image.PACMAN)

while True:
    # Button A: Ask Mog1 AI for a smart quote / status
    if button_a.was_pressed():
        display.show(Image.ARROW_W)
        uart.write("PROMPT:Give a 1-sentence motivation quote for microbit\n")
        sleep(500)

    # Button B: Request current weather / system info
    if button_b.was_pressed():
        display.show(Image.ARROW_E)
        uart.write("PROMPT:system info\n")
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
            elif msg.startswith("TEXT:"):
                display.scroll(msg.split("TEXT:")[1].strip())
            else:
                display.scroll(msg)

    sleep(100)
