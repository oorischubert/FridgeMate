import asyncio
import cv2
import numpy as np
import nats
from nats.errors import ConnectionClosedError, TimeoutError, NoServersError
import signal
import argparse
import time 
import tensorflow as tf
import os
import csv
Interpreter = tf.lite.Interpreter

# >>> HAND TRACKING IMPORT <<<
import utility.handTracking as htm

# --- Configuration ---
DEFAULT_NATS_URL = "nats://localhost:4222"
FRAMES_SUBJECT = "camera.frames"
RESULTS_SUBJECT = "detection.results" 
WINDOW_NAME = "FridgeMate Stream"

# --- TFLite Model Configuration ---
MODEL_PATH = 'utility/model.tflite'
LABEL_PATH = 'utility/coco_labels.txt'
CONFIDENCE_THRESHOLD = 0.4 
TARGET_LABELS = ["apple", "banana", "cell phone"]

# --- Hand Tracking Configuration ---
MAX_HANDS = 1 # Let's track only one hand for simplicity now
hand_detector = htm.handDetector(maxHands=MAX_HANDS, detectionCon=0.7) # Initialize detector
baseline_hand_width = None # Store the initial reference width
current_relative_distance = 1.0 # Default relative distance
SMOOTHING_FACTOR = 0.3 # Simple exponential smoothing for distance

# --- FPS Measurement Globals ---
prev_frame_time = time.time()  # timestamp of previous processed frame
fps_display = 0.0              # exponentially smoothed FPS value

# --- Global flag for shutdown ---
shutdown_event = asyncio.Event()

# --- Signal Handler ---
def handle_signal(sig, frame):
    print(f"\nReceived signal {sig}, initiating shutdown...")
    shutdown_event.set()
    cv2.destroyAllWindows()

# --- Helper function to load labels ---
def load_labels(path: str):
    label_map = {}
    # --- CSV branch ---------------------------------------------------------
    if os.path.splitext(path)[1].lower() == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue                       # blank line
                if not row[0].strip().isdigit():
                    continue                       # likely a header
                class_id = int(row[0])
                label_text = row[1].strip() if len(row) > 1 else ""
                label_map[class_id] = label_text
        return label_map

    # --- Plain-text branch --------------------------------------------------
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                class_id = int(parts[0])
                label_text = parts[1]
            else:
                class_id = line_num
                label_text = parts[0] if parts else ""
            label_map[class_id] = label_text

    return label_map


# --- NATS Message Handler ---
async def message_handler(msg, nc, interpreter, input_details, labels, input_height, input_width, floating_model):
    """Callback function to handle frames, detect objects, detect hands, and publish results."""
    global baseline_hand_width, current_relative_distance, prev_frame_time, fps_display # Allow modification of globals

    subject = msg.subject
    data = msg.data

    try:
        nparr = np.frombuffer(data, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            print("WARN: Failed to decode JPEG image")
            return
        
        # --- FPS calculation ---
        now = time.time()
        # Use exponential smoothing to reduce jitter in display
        if prev_frame_time is not None:
            instantaneous_fps = 1.0 / max(now - prev_frame_time, 1e-6)
            fps_display = 0.9 * fps_display + 0.1 * instantaneous_fps
        prev_frame_time = now
        
        #img_bgr = cv2.flip(img_bgr, 1)

        original_h, original_w, _ = img_bgr.shape

        # --- HAND TRACKING START ---
        # 1. Find Hands and draw basic landmarks/connections
        img_bgr = hand_detector.findHands(img_bgr, draw=True)
        # 2. Get landmark positions (without drawing default circles/box)
        lmList, bbox = hand_detector.findPosition(img_bgr, draw=False)
        hand_detected_this_frame = False

        if len(lmList) != 0:
            hand_detected_this_frame = True
            # 3. Calculate distance between base of index (5) and pinky (17) as proxy for hand size/distance
            #    We use draw=False because findHands already drew.
            current_width, _, _ = hand_detector.findDistance(5, 17, img_bgr, draw=False)

            if current_width > 10: # Basic check to avoid tiny erroneous widths
                if baseline_hand_width is None:
                    baseline_hand_width = current_width
                    print(f"INFO: Captured baseline hand width: {baseline_hand_width:.2f}")
                    current_relative_distance = 1.0
                else:
                    # Calculate relative distance (smaller width means further away)
                    relative_dist = baseline_hand_width / current_width # Inverted ratio
                    # Apply smoothing
                    current_relative_distance = (SMOOTHING_FACTOR * relative_dist) + \
                                                ((1 - SMOOTHING_FACTOR) * current_relative_distance)

                # Display the relative distance on the image
                cv2.putText(img_bgr, f"Rel Dist: {current_relative_distance:.2f}", (10, original_h - 60),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

        # Optional: If hand disappears, maybe reset baseline? Or let distance stay?
        # else: # No hand detected
        #     if baseline_hand_width is not None:
        #         # Option: Reset baseline after a while? Or keep last distance?
        #         # baseline_hand_width = None # Uncomment to reset baseline when hand disappears
        #         # current_relative_distance = 1.0 # Reset distance
        #         pass # Keep last distance for now
        #     cv2.putText(img_bgr, f"Rel Dist: --", (10, original_h - 60),
        #                 cv2.FONT_HERSHEY_PLAIN, 2, (0, 100, 100), 2)

        # --- HAND TRACKING END ---

        # --- OBJECT DETECTION START ---
        # Preprocess for object detection
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # Model likely expects RGB
        img_resized = cv2.resize(img_rgb, (input_width, input_height))
        input_data = np.expand_dims(img_resized, axis=0) # Add batch dimension

        if floating_model:
            # Assuming normalization to [-1, 1] if float model
            input_data = (np.float32(input_data) - 127.5) / 127.5

        # Perform object detection inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Get object detection results
        boxes   = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]  # Boxes
        classes  = interpreter.get_tensor(interpreter.get_output_details()[1]['index'])[0]  # Scores
        scores = interpreter.get_tensor(interpreter.get_output_details()[2]['index'])[0]  # Classes 

        detected_items_in_frame = []

        for i in range(len(scores)):
            if scores[i] > CONFIDENCE_THRESHOLD:
                class_id = int(classes[i])
                # Assuming 0-based IDs match 0-based labels_map keys
                label = labels.get(class_id, f"unknown_id_{class_id}")

                # DEBUG print for all detections
                # print(f"DEBUG: Detected '{label}' (ID: {class_id}) with score {scores[i]:.2f}")

                if label in TARGET_LABELS:
                    if label not in detected_items_in_frame:
                        detected_items_in_frame.append(label)

                    # Draw GREEN box for target objects
                    ymin, xmin, ymax, xmax = boxes[i]
                    if 0.0 <= ymin < ymax <= 1.0 and 0.0 <= xmin < xmax <= 1.0: # Check validity
                        left, right, top, bottom = (xmin * original_w, xmax * original_w,
                                                    ymin * original_h, ymax * original_h)
                        cv2.rectangle(img_bgr, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
                        cv2.putText(img_bgr, f"{label}: {scores[i]:.2f}", (int(left), int(top) - 7),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Send messages for detected target food items
        for item_label in detected_items_in_frame:
            payload = f"FOOD:{item_label} found!".encode('utf-8') # Added prefix
            # print(f"Sending message to '{RESULTS_SUBJECT}': {payload.decode()}")
            try:
                await nc.publish(RESULTS_SUBJECT, payload)
            except ConnectionClosedError:
                print(f"ERROR: Cannot send '{item_label} found', NATS connection closed.")
            except Exception as e:
                print(f"ERROR: Failed to publish '{item_label} found': {e}")

        # --- OBJECT DETECTION END ---

        # Draw FPS in the top‑left corner
        cv2.putText(img_bgr, f"FPS: {fps_display:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display the image (now includes hand landmarks and object boxes)
        cv2.imshow(WINDOW_NAME, img_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
             print("Quit key pressed in window.")
             shutdown_event.set()

    except Exception as e:
        print(f"ERROR processing frame or in detection: {e}")
        import traceback
        traceback.print_exc()

# --- Main Application Logic ---
async def main_viewer(nats_url, model_path_arg, label_path_arg): # Added arguments
    global baseline_hand_width, current_relative_distance # Declare globals needed
    nc = None
    sub = None
    interpreter_instance = None

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    # --- Load TFLite Model and Labels ---
    if Interpreter is None:
        print("FATAL: TFLite runtime not available. Exiting.")
        return

    try:
        labels = load_labels(label_path_arg) # Use argument
        if labels is None: return # Exit if labels failed to load

        interpreter_instance = Interpreter(model_path=model_path_arg) # Use argument
        interpreter_instance.allocate_tensors()
        input_details = interpreter_instance.get_input_details()
        output_details = interpreter_instance.get_output_details()
        input_height = input_details[0]['shape'][1]
        input_width = input_details[0]['shape'][2]
        floating_model = (input_details[0]['dtype'] == np.float32)
        print(f"Model loaded. Input shape: {input_details[0]['shape']}, Floating model: {floating_model}")
    except Exception as e:
        print(f"FATAL: Could not load TFLite model or labels: {e}")
        print(f"Ensure '{model_path_arg}' and '{label_path_arg}' are correct.")
        return

    try:
        print(f"Connecting to NATS server at {nats_url}...")
        async def disconnected_cb(): print("NATS connection lost...")
        async def reconnected_cb(): print(f"Reconnected to NATS...")
        async def error_cb(e): print(f"NATS Error: {e}")

        nc = await nats.connect(
            nats_url, connect_timeout=5, reconnect_time_wait=2,
            disconnected_cb=disconnected_cb, reconnected_cb=reconnected_cb, error_cb=error_cb
        )
        print(f"Connected to NATS! Client ID: {nc.client_id}")

        # --- Define ASYNC Callback Wrapper ---
        # Pass all necessary components to the message handler
        async def wrapped_message_handler(msg):
            await message_handler(msg, nc, interpreter_instance, input_details, labels, input_height, input_width, floating_model)

        print(f"Subscribing to frames subject: '{FRAMES_SUBJECT}'")
        sub = await nc.subscribe(FRAMES_SUBJECT, cb=wrapped_message_handler)


        print("Viewer running. Waiting for frames...")
        print(f"Displaying stream in window '{WINDOW_NAME}'. Press 'q' in window or Ctrl+C in terminal to exit.")
        await shutdown_event.wait()

        print("Shutdown signal received, exiting...")
        # if 'stdin_sender_task' in locals() and not stdin_sender_task.done(): # Check if task exists before cancelling
        #     stdin_sender_task.cancel()

    # ... (Rest of exception handling and cleanup remains the same) ...
    except NoServersError as e: print(f"FATAL: No NATS servers: {e}")
    except ConnectionRefusedError: print(f"FATAL: NATS connection refused at {nats_url}.")
    except TimeoutError: print(f"FATAL: NATS connection timeout at {nats_url}.")
    except Exception as e:
        print(f"ERROR in main viewer loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cleaning up...")
        if sub:
            try: await sub.unsubscribe()
            except: pass
        if nc and nc.is_connected:
            try: await nc.drain()
            except: pass
        print("NATS connection drained/closed.")
        cv2.destroyAllWindows()
        print("Viewer finished.")


# --- Entry Point ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NATS Camera Stream Viewer with Object and Hand Detection')
    parser.add_argument('--nats_url', type=str, default=DEFAULT_NATS_URL)
    parser.add_argument('--frames_subject', type=str, default=FRAMES_SUBJECT)
    parser.add_argument('--results_subject', type=str, default=RESULTS_SUBJECT)
    parser.add_argument('--model', type=str, default=MODEL_PATH, help="Path to TFLite model file")
    parser.add_argument('--labels', type=str, default=LABEL_PATH, help="Path to labels file")

    args = parser.parse_args()

    # Set global config using arguments
    FRAMES_SUBJECT = args.frames_subject
    RESULTS_SUBJECT = args.results_subject
    # MODEL_PATH and LABEL_PATH are now passed into main_viewer

    print("Starting NATS Stream Viewer with Object & Hand Detection...")
    print(f" --- NATS Server: {args.nats_url}")
    print(f" --- Subscribing To Frames: {FRAMES_SUBJECT}")
    print(f" --- Publishing Results To: {RESULTS_SUBJECT}")
    print(f" --- Using Object Model: {args.model}")
    print(f" --- Using Labels: {args.labels}")

    try:
        # Pass model and label paths from args to main_viewer
        asyncio.run(main_viewer(args.nats_url, args.model, args.labels))
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught in top level.")
    print("Application terminated.")