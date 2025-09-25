# =========================
# HAND GESTURE MOVEMENT CONTROL (Mediapipe + Cooldown)
# =========================

import cv2
import time
import mediapipe as mp

# -------------------------
# Motor Control (Dummy)
# -------------------------
def motor_forward():
    print("⚙️ Motor: Moving Forward")

def motor_left():
    print("⚙️ Motor: Turning Left")

def motor_right():
    print("⚙️ Motor: Turning Right")

def motor_stop():
    print("⚙️ Motor: Stopped")

def print_log(msg):
    print(f"[LOG] {msg}")

# -------------------------
# Mediapipe Hands Setup
# -------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)  # Detect 1 hand for control
mp_draw = mp.solutions.drawing_utils

# -------------------------
# Cooldown Setup
# -------------------------
last_action_time = 0
cooldown = 5  # seconds

# -------------------------
# Gesture Detection
# -------------------------
def detect_hand_gesture(frame):
    global last_action_time

    h, w, c = frame.shape
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    current_time = time.time()

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            # Get wrist point (landmark 0)
            wrist = handLms.landmark[0]
            cx, cy = int(wrist.x * w), int(wrist.y * h)

            # Only act if cooldown finished
            if current_time - last_action_time >= cooldown:
                if cy < h // 3:
                    print_log("Hand UP → Moving Stop")
                    motor_stop()
                    last_action_time = current_time
                elif cy > 2 * h // 3:
                    print_log("Hand DOWN → Moving Forward")
                    motor_forward()
                    last_action_time = current_time
                elif cx < w // 3:
                    print_log("Hand LEFT → Moving Left")
                    motor_left()
                    last_action_time = current_time
                elif cx > 2 * w // 3:
                    print_log("Hand RIGHT → Moving Right")
                    motor_right()
                    last_action_time = current_time

    return frame

# -------------------------
# Hand Control Loop
# -------------------------
def task_handcontrol():
    cap = cv2.VideoCapture(0)
    time.sleep(0.5)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print_log("Hand Gesture Control Started")
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = detect_hand_gesture(frame)
            cv2.imshow("Hand Control", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        motor_stop()
        print_log("Hand Gesture Control Stopped")

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    task_handcontrol()
