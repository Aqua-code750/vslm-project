import sys
import time
import re
from mog1_agent import Mog1Agent

# Reconfigure stdout for UTF-8 on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def find_microbit_port():
    """Attempts to discover connected BBC micro:bit COM port on Windows/Mac/Linux."""
    try:
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()
        for p in ports:
            if "micro:bit" in p.description.lower() or "mbed" in p.description.lower() or "VID:PID=0D28" in p.hwid:
                return p.device
    except ImportError:
        pass
    return None

def start_microbit_bridge(com_port=None):
    """
    Connects BBC micro:bit to Mog1 AI Autonomous Model Engine.
    Receives button/sensor prompts from micro:bit and sends AI responses & LED icons back!
    """
    print("=" * 65)
    print("🚀 Mog1 AI <-> BBC micro:bit Smart Hardware Bridge")
    print("=" * 65)

    agent = Mog1Agent()
    port = com_port or find_microbit_port()

    if not port:
        print("\n⚠️ No physical BBC micro:bit detected automatically over USB Serial.")
        print("💡 Simulation Mode Active! You can test micro:bit commands below:")
        print("-" * 65)
        
        test_prompts = [
            "Give a 1-sentence motivation quote for microbit",
            "system info",
            "What is 12 * 12?"
        ]
        for p in test_prompts:
            print(f"\n[micro:bit -> Mog1 AI]: {p}")
            res = agent.run(p)
            print(f"[Mog1 AI -> micro:bit]: {res[:120]}...")
        return

    try:
        import serial
        ser = serial.Serial(port, 115200, timeout=1)
        print(f"\n🟢 Successfully connected to BBC micro:bit on port: {port}")
        print("Waiting for micro:bit button presses and sensor events...\n")

        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith("PROMPT:"):
                    prompt = line.replace("PROMPT:", "").strip()
                    print(f"📥 [micro:bit Prompt]: {prompt}")
                    
                    # Run Mog1 AI Model / Agent
                    ai_response = agent.run(prompt)
                    clean_res = re.sub(r'[^\w\s\.\,\!\?]', '', ai_response).strip()
                    
                    print(f"📤 [Mog1 AI Response]: {clean_res[:100]}...")
                    
                    # Send response back to micro:bit 5x5 LED matrix
                    ser.write(f"ICON:HAPPY\n".encode('utf-8'))
                    time.sleep(0.2)
                    ser.write(f"TEXT:{clean_res[:60]}\n".encode('utf-8'))

            time.sleep(0.1)
    except Exception as e:
        print(f"Serial Error: {e}")

if __name__ == "__main__":
    start_microbit_bridge()
