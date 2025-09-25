import multiprocessing
import tkinter as tk
import os
import psutil
import time
from datetime import datetime
import speech_recognition as sr
from transformers import pipeline
import google.generativeai as genai
import pyttsx3
import sys

import queue
import cv2
import platform
import pytesseract
from ultralytics import YOLO
import re

# Pin Definitions
ULTRA_SONIC_TRIGPIN = 22
ULTRA_SONIC_ECHOPIN = 24
IR_SENSOR_1 = 26
IR_SENSOR_2 = 37
RELAY1 = 40
RELAY2 = 38
RELAY3 = 36
RELAY4 = 32
LED_PIN = 18
led_on_flag = False
speaking = False
MIN_DISTANCE = 20  # cm
MAX_DISTANCE = 50  # cm
SPEED_DELAY = 0.2  # Delay for motor control

GEMINI_API_KEY = "AIzaSyC4mgpLtHv6tjBCMSXV_hh1yzsX_kjM_Xk"  # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')  # Or another Gemini model

# Text-to-Speech Setup
engine = pyttsx3.init()
audio_queue = queue.Queue()
last_distance = None  # Track the last announced distance
distance_threshold = 5  # Minimum change in cm to trigger new announcement


def print_log(str):
    start_time = time.time()
    formatted_time = datetime.fromtimestamp(start_time).strftime("%I:%M:%S %p")
    log_entry = f"{formatted_time}\t{str}"
    print(log_entry)
    #with open("robot_log.txt", "a") as log_file:  # "a" mode for append
        #log_file.write(log_entry + "\n")  # Add newline for readability

print_log("Smart Vision Assistant: Interactive Object Detection and Distance Tracking System")


OSNAME = ""


def check_os():
    global OSNAME
    OSNAME = platform.system()
    if OSNAME == 'Windows':
        print_log("Operating system is Windows")
    elif OSNAME == 'Linux':
        print_log("Operating system is Linux")
    elif OSNAME == 'Darwin':
        print_log("Operating system is macOS")
    else:
        print_log("Unknown operating system")
    return OSNAME

check_os()
    
with open("robot_log.txt", "w") as log_file:  # "a" mode for append
    log_file.write("")  # Add newline for readability




# List of COCO class names with indices for YOLOv8
# COCO stands for Common Objects in Context
# Contains 80 common objects
# Images: Over 200,000 labeled images with more than 1.5 million object instances.
coco_class_names = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Example usage with model
yolomodel = YOLO('yolov8n.pt')

def speak(text, rate=150):
    global speaking
    """Converts text to speech with adjustable speed."""
    if not speaking:
        speaking = True
        engine.setProperty('rate', rate)
        engine.say(text)
        engine.runAndWait()
        speaking = False


def get_gemini_response(prompt):
    try:
        modified_prompt = f"{prompt}. Please provide a concise answer, ideally within two short lines."
        response = model.generate_content(modified_prompt)
        gemini_response_text = response.text
        #print_log("Gemini API Response: " + gemini_response_text)
        return gemini_response_text
    except Exception as e:
        print_log(f"Error getting Gemini API response: {e}")

    return "Error in API response."


def setup():
    # GPIO Setup
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)
    
    GPIO.setup(ULTRA_SONIC_TRIGPIN, GPIO.OUT)
    GPIO.setup(ULTRA_SONIC_ECHOPIN, GPIO.IN)

    GPIO.setup(RELAY1, GPIO.OUT)
    GPIO.setup(RELAY2, GPIO.OUT)
    GPIO.setup(RELAY3, GPIO.OUT)
    GPIO.setup(RELAY4, GPIO.OUT)
    GPIO.setup(LED_PIN, GPIO.OUT)
    
    GPIO.setup(IR_SENSOR_1, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(IR_SENSOR_2, GPIO.IN, pull_up_down=GPIO.PUD_UP)

    print_log("HUMAN FOLLOWING RESPONDING ROBOT")

    # Initial relay states
    GPIO.output(RELAY1, GPIO.LOW)
    GPIO.output(RELAY2, GPIO.LOW)
    GPIO.output(RELAY3, GPIO.LOW)
    GPIO.output(RELAY4, GPIO.LOW)


if OSNAME == 'Linux':
    import RPi.GPIO as GPIO

    setup()

'''

camera = cv2.VideoCapture(0)
time.sleep(0.5)
if not camera.isOpened():
    print("Error: Could not open USB camera")
    sys.exit(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = int(camera.get(cv2.CAP_PROP_FPS))
print_log(f"Camera Frame Width={frame_width}\t Height={frame_height}\tFPS={fps}")
camera.release()
'''

# Function to measure distance
def get_distance():
    if OSNAME == 'Windows':
        return 9999
    GPIO.output(ULTRA_SONIC_TRIGPIN, GPIO.LOW)
    time.sleep(0.1)  # Small delay to settle

    # Send 10µs pulse to trigger
    GPIO.output(ULTRA_SONIC_TRIGPIN, GPIO.HIGH)
    time.sleep(0.00001)  # 10 microseconds
    GPIO.output(ULTRA_SONIC_TRIGPIN, GPIO.LOW)

    # Measure the echo pulse
    start_time = time.time()
    stop_time = time.time()

    # Wait for echo to go high (start of pulse)
    while GPIO.input(ULTRA_SONIC_ECHOPIN) == 0:
        start_time = time.time()
        # Timeout after 1 second if no response
        if time.time() - start_time > 1:
            return -1  # Indicate failure

    # Wait for echo to go low (end of pulse)
    while GPIO.input(ULTRA_SONIC_ECHOPIN) == 1:
        stop_time = time.time()
        # Timeout after 1 second if stuck high
        if time.time() - stop_time > 1:
            return -2  # Indicate failure

    # Calculate distance
    time_elapsed = stop_time - start_time
    distance = (time_elapsed * 34300) / 2  # Speed of sound in cm/s 343 m/s
    return int(distance)


def motor_forward():
    """Move robot forward"""
    print_log("Moving Forward...")
    GPIO.output(RELAY1, GPIO.HIGH)  # Left forward
    GPIO.output(RELAY2, GPIO.LOW)
    GPIO.output(RELAY3, GPIO.HIGH)  # Right forward
    GPIO.output(RELAY4, GPIO.LOW)


def motor_reverse():
    """Move robot backward (reverse)"""
    print_log("Moving Reverse...")
    GPIO.output(RELAY1, GPIO.LOW)
    GPIO.output(RELAY2, GPIO.HIGH)  # Left backward
    GPIO.output(RELAY3, GPIO.LOW)
    GPIO.output(RELAY4, GPIO.HIGH)  # Right backward


def motor_stop():
    """Stop robot"""
    print_log("Motor Stop...")
    GPIO.output(RELAY1, GPIO.LOW)
    GPIO.output(RELAY2, GPIO.LOW)
    GPIO.output(RELAY3, GPIO.LOW)
    GPIO.output(RELAY4, GPIO.LOW)


def motor_left():
    """Turn robot left"""
    print_log("Turning Left...")
    GPIO.output(RELAY1, GPIO.LOW)
    GPIO.output(RELAY2, GPIO.HIGH)  # Left backward
    GPIO.output(RELAY3, GPIO.HIGH)  # Right forward
    GPIO.output(RELAY4, GPIO.LOW)


def motor_right():
    """Turn robot right"""
    print_log("Turning Right...")
    GPIO.output(RELAY1, GPIO.HIGH)  # Left forward
    GPIO.output(RELAY2, GPIO.LOW)
    GPIO.output(RELAY3, GPIO.LOW)
    GPIO.output(RELAY4, GPIO.HIGH)  # Right backward
    
# Each task must be in a separate function and outside of the class
def task_robotfollowing(is_automatic):
    blink_led(2)
    if is_automatic == False:
        return 
    ps = psutil.Process(os.getpid())
    ps.cpu_affinity([1])  # CPU core 1
   
    print(f"Running RobotFollowing in Automatic mode on PID {os.getpid()}")
    
    
    try:
        stable_time = 0
        start_time = time.time()
        lb_found = False
        ls_str=''
        
        while True:
            if is_automatic == False:
                break
            output = GPIO.input(IR_SENSOR_1)
            output2 = GPIO.input(IR_SENSOR_2)
            output = 1 - output
            output2 = 1 - output2

            distanceCm = get_distance()
            lb_found = False
            #print(f"Distance: {distanceCm}cm, IR1:{output}, IR2:{output2}")
            # Condition : Stop when too close
            if distanceCm < MIN_DISTANCE - 10:
                lb_found = True
                motor_reverse()
                
                print_log(f"Stop when too close, Distance: {distanceCm}cm")
                speak("Too close", rate=150)
                time.sleep(0.5)
                motor_stop()

            # Condition : Stop
            
            elif output == 1 or output2 == 1:
                if output == 0 and output2 == 1:
                    print_log(f"Turn right, Distance: {distanceCm}cm")
                    speak("Right", rate=150)
                    motor_right()
                elif output == 1 and output2 == 0:
                    print_log(f"Turn left, Distance: {distanceCm}cm")
                    speak("Left", rate=150)
                    motor_left()
            elif distanceCm > MAX_DISTANCE:  # and output == 0 and output2 == 0:
                lb_found = True
                #print_log(f"Stop, Too far!, Distance: {distanceCm}cm")
                #speak("Too far!", rate=150)
                motor_stop()
            elif distanceCm >= MIN_DISTANCE and distanceCm <= MAX_DISTANCE:  # and output == 0 and output2 == 0:
                print_log(f"Forward, Distance: {distanceCm}cm")
                speak("Forward", rate=150)
                motor_forward()

            if lb_found:
                stable_time = int(time.time() - start_time)
            else:
                stable_time = 0  # Reset stability check
                start_time = time.time()
            if stable_time >= 5:
                print_log("Sleep 2 seconds")
                time.sleep(2)
                stable_time = 0
                start_time = time.time()
            else:
                time.sleep(SPEED_DELAY)

    except KeyboardInterrupt:
        print_log("Error is raised")
        #GPIO.cleanup()
    finally:
        
        motor_stop()
        speak("Robot stopped by user", rate=150)
        #GPIO.cleanup()

def task_speech():
    ps = psutil.Process(os.getpid())
    ps.cpu_affinity([2])  # CPU core 2
    print(f"Running Speech on PID {os.getpid()}")
    
    recognizer = sr.Recognizer()

    print_log("🎤 Speak something, and I'll answer!")
    speak("Speak something, and I'll answer!")
    blink_led(2)
    while True:
        try:
            with sr.Microphone(device_index=1) as source:
                print_log("Listening...")
                speak("Listening", 150)
                recognizer.adjust_for_ambient_noise(source, duration=0.5)  # Reduce duration to 0.5 seconds
                audio = recognizer.listen(source)

                # Recognize speech using Google Speech Recognition
                user_input = recognizer.recognize_google(audio)
                print_log(f"You said: {user_input}")
                if "exit" in user_input.lower() or "goodbye" in user_input.lower():
                #if "exit" in user_input.lower():
                    print_log("Goodbye!")
                    speak("Goodbye!", 100)
                    #self.stop()
                    break

                # Get response from free AI model
                response = get_gemini_response(user_input)
                print_log(f"🤖 AI says: {response}")
                blink_led(1)
                speak(response)

        except sr.UnknownValueError:
            print_log("I couldn't understand. Please try again.")
            speak("I couldn't understand. Please try again.", 150)
        except sr.RequestError:
            print_log("Speech recognition service error.")
            speak("Speech recognition service error.")
        except Exception as e:
            print_log(f"An error occurred: {e}")

            speak("An error occurred.")    

def object_detection(frame):
    global yolomodel
    """Detect objects using YOLO"""
    # Resize frame for better performance on Pi
    small_frame = cv2.resize(frame, (320, 240))
    results = yolomodel(small_frame, verbose=False)
    detected_objects = []
    #return frame, detected_objects
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = yolomodel.names[cls]

            if conf >= 0.7:  # Lowered threshold for Pi
                # Scale coordinates back to original size
                scale_x = frame.shape[1] / 320
                scale_y = frame.shape[0] / 240
                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                detected_objects.append(label)
                print_log(f"Detected:{label}")

    return frame, detected_objects



    
def task_objectdetect():
    ps = psutil.Process(os.getpid())
    ps.cpu_affinity([3])  # CPU core 3
    
    print(f"Running ObjectDetect on PID {os.getpid()}")
    
    camera = cv2.VideoCapture(0)
    time.sleep(0.5)
    if not camera.isOpened():
        print("Error: Could not open USB camera")
        sys.exit(0)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


    frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = int(camera.get(cv2.CAP_PROP_FPS))
    print_log(f"Camera Frame Width={frame_width}\t Height={frame_height}\tFPS={fps}")

    last_text = ""
    last_objects = []
    frame_count = 0
    blink_led(2)
    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Error: Could not read frame")
                break
            current_time = time.time()
            '''
            # Text detection (every 2 seconds)
            if current_time % 2 < 0.1:
                text = read_text(frame)
                if text and text != last_text and len(text) > 3:
                    audio_queue.put(f"Text: {text}")
                    last_text = text
            '''        
            if OSNAME == 'Linux':
                frame_count += 1
                if frame_count % 10 != 0:
                    continue
                else:
                    frame_count = 0
            # frame = cv2.resize(frame, (320, 240))
            # Object detection (every 2 frames to save resources)
            
            if int(current_time * 10) % 20 == 0 or 1 == 1:
                frame, detected_objects = object_detection(frame)
                if detected_objects != last_objects and detected_objects:
                    blink_led(2)
                    audio_queue.put(f"Objects: {', '.join(detected_objects)}")
                    last_objects = detected_objects
           

            # Process audio queue
            try:
                while not audio_queue.empty():
                    message = audio_queue.get_nowait()
                    speak(message)
            except queue.Empty:
                pass

            # Optional: Display on screen (comment out if not using display)
            cv2.imshow('Object Detection - Live', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
    

class HumanFollowingRobot:
    def __init__(self, root):
        self.root = root
        self.root.title("Human Following Robot")

        self.task_states = {
            "robotfollowing": None,
            "speech": None,
            "objectdetect": None
        }

        self.buttons = {}
        self.canvas_buttons = {}

        self.canvas_button_actions = {
            "Forward": motor_forward,
            "Reverse": motor_reverse,
            "Left": motor_left,
            "Right": motor_right
        }

        self.option_var = tk.StringVar(value="Automatic")

        self.create_main_buttons()
        self.create_options()
        self.create_canvas_buttons()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_main_buttons(self):
        tasks = [("RobotFollowing", "robotfollowing"),
                 ("Speech", "speech"),
                 ("ObjectDetect", "objectdetect")]

        for idx, (label, name) in enumerate(tasks):
            btn = tk.Button(self.root, text=label, bg="grey", width=15, height=2,
                            command=lambda n=name: self.toggle_task(n))
            btn.grid(row=0, column=idx, padx=10, pady=10)
            self.buttons[name] = btn

    def create_options(self):
        self.option_auto = tk.Radiobutton(self.root, text="Automatic", variable=self.option_var,
                                          value="Automatic", state="normal")
        self.option_manual = tk.Radiobutton(self.root, text="Manual", variable=self.option_var,
                                            value="Manual", state="normal")
        self.option_auto.grid(row=2, column=0, pady=5)
        self.option_manual.grid(row=1, column=0, pady=5)

    def create_canvas_buttons(self):
        self.canvas = tk.Canvas(self.root, width=300, height=200, bg="white")
        self.canvas.grid(row=3, column=0, columnspan=3, pady=20)

        self.radius = 30
        positions = {
            "Forward": (150, 30),
            "Reverse": (150, 90),
            "Left": (70, 60),
            "Right": (230, 60)
        }

        def bind_button(oval_id, text_id, button_name):
            self.canvas.tag_bind(oval_id, "<Button-1>", lambda event: self.toggle_canvas_button(button_name))
            self.canvas.tag_bind(text_id, "<Button-1>", lambda event: self.toggle_canvas_button(button_name))

        for name, (x, y) in positions.items():
            oval = self.canvas.create_oval(x - self.radius, y - self.radius,
                                           x + self.radius, y + self.radius,
                                           fill="grey", outline="black")
            text = self.canvas.create_text(x, y, text=name)

            bind_button(oval, text, name)
            self.canvas_buttons[name] = {"button": oval, "active": False}
            
    def process_end(self, task_name):
        
        proc = self.task_states[task_name]
        if proc and ( proc.is_alive() or ( task_name == "robotfollowing" and self.option_var.get() == "Manual") ):
            if not( task_name == "robotfollowing" and self.option_var.get() == "Manual"):
                proc.terminate()
            
            self.task_states[task_name] = None
            self.buttons[task_name].config(bg="grey")
            print(f"{task_name} stopped")
        if task_name == "robotfollowing":
                motor_stop()
                for btn_name, btn_info in self.canvas_buttons.items():
                    self.canvas.itemconfig(btn_info["button"], fill="grey")
                    self.canvas_buttons[btn_name]["active"] = False
            
    def on_closing(self):
        print("Window is closing... Cleaning up processes.")
        self.process_end("robotfollowing")
        self.process_end("speech")
        self.process_end("objectdetect")
        self.root.destroy()        
        
        
    def toggle_task(self, task_name):
        proc = self.task_states[task_name]
        if proc and ( proc.is_alive() or ( task_name == "robotfollowing" and self.option_var.get() == "Manual") ):
            self.process_end(task_name)
            
                
            #proc.terminate()
            #self.task_states[task_name] = None
            #self.buttons[task_name].config(bg="grey")
            #print(f"{task_name} stopped")
        else:
            if task_name == "robotfollowing":
                if self.option_var.get() == "Automatic":
                    is_automatic = True
                    self.task_states[task_name] = multiprocessing.Process(target=task_robotfollowing, args=(is_automatic,))
                else:
                    is_automatic = False
                    self.task_states[task_name] = multiprocessing.Process(target=task_robotfollowing, args=(is_automatic,))
                    
            elif task_name == "speech":
                self.task_states[task_name] = multiprocessing.Process(target=task_speech)
            elif task_name == "objectdetect":
                self.task_states[task_name] = multiprocessing.Process(target=task_objectdetect)

            self.task_states[task_name].start()
            self.buttons[task_name].config(bg="green")
            print(f"{task_name} started")

    def toggle_canvas_button(self, name):
        
        if not self.task_states["robotfollowing"] or ( not self.task_states["robotfollowing"].is_alive() and self.option_var.get() == "Automatic" ):
            
            print("RobotFollowing task is OFF")
            return

        if self.option_var.get() != "Manual":
            
            print("Not in Manual mode")
            return
            
        motor_stop()
        for btn_name, btn_info in self.canvas_buttons.items():
            if btn_name != name:
                self.canvas.itemconfig(btn_info["button"], fill="grey")
                self.canvas_buttons[btn_name]["active"] = False

        button_info = self.canvas_buttons[name]
        is_active = not button_info["active"]
        button_info["active"] = is_active
        self.canvas.itemconfig(button_info["button"], fill="green" if is_active else "grey")

        if is_active:
            self.canvas_button_actions[name]()

def led_on():
    global  led_on_flag
    if OSNAME == 'Linux':
        global led_on_flag
        if led_on_flag == False:
            led_on_flag = True
            GPIO.output(LED_PIN, GPIO.HIGH)


def led_off():
    global led_on_flag
    if OSNAME == 'Linux':
        global led_on_flag
        if led_on_flag == True:
            led_on_flag = False
            GPIO.output(LED_PIN, GPIO.LOW)


def blink_led(duration=5, interval=0.125):
    
    if OSNAME == 'Linux':
        end_time = time.time() + duration

        while time.time() < end_time:
            led_on()  # LED On
            time.sleep(interval)  # Wait for the specified interval
            led_off()
            time.sleep(interval)  # Wait for the specified interval

        led_off()
   

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    root = tk.Tk()
    app = HumanFollowingRobot(root)
    if OSNAME == 'Linux':
        blink_led()
    root.mainloop()
