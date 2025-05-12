import asyncio
import cv2
import numpy as np
import nats
from nats.errors import ConnectionClosedError, TimeoutError, NoServersError
import signal
import argparse
import sys
import time # For timestamping "apple found" messages
import tensorflow as tf
Interpreter = tf.lite.Interpreter

# --- Configuration ---
DEFAULT_NATS_URL = "nats://localhost:4222"
FRAMES_SUBJECT = "camera.frames"
RESULTS_SUBJECT = "detection.results" # Subject to PUBLISH results TO Pi
WINDOW_NAME = "FridgeMate Stream"

# --- TFLite Model Configuration ---
MODEL_PATH = '1.tflite'    # Path to your .tflite model file https://www.kaggle.com/models/tensorflow/efficientdet/tfLite
LABEL_PATH = 'coco_labels.txt'   # Path to your labels file  https://dl.google.com/coral/canned_models/coco_labels.txt
CONFIDENCE_THRESHOLD = 0.45    # Minimum score to consider a detection valid
TARGET_LABELS = ["apple", "banana","cell_phone"]

# --- Global flag for shutdown ---
shutdown_event = asyncio.Event()

# --- Signal Handler ---
def handle_signal(sig, frame):
    print(f"\nReceived signal {sig}, initiating shutdown...")
    shutdown_event.set()
    cv2.destroyAllWindows()

# --- Helper function to load labels ---
def load_labels(path):
    label_map = {}
    with open(path, "r") as f:
        for line_num, line in enumerate(f):
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                class_id = int(parts[0])
                label = parts[1]
            else:
                class_id = line_num
                label = parts[0] if parts else ""
            label_map[class_id] = label
    print(label_map)
    return label_map

# --- NATS Message Handler (includes nc, interpreter, input_details, labels) ---
async def message_handler(msg, nc, interpreter, input_details, labels, input_height, input_width, floating_model):
    """Callback function to handle received frames, detect apples, and publish results."""
    subject = msg.subject
    data = msg.data

    try:
        nparr = np.frombuffer(data, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # OpenCV uses BGR by default

        if img_bgr is None:
            print("WARN: Failed to decode JPEG image")
            return

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # Model likely expects RGB
        img_resized = cv2.resize(img_rgb, (input_width, input_height))
        input_data = np.expand_dims(img_resized, axis=0) # Add batch dimension

        if floating_model:
            input_data = (np.float32(input_data) - 127.5) / 127.5 # Normalize to [-1, 1]

        # Perform inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Get detection results
        boxes   = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]  # Bounding boxes
        classes  = interpreter.get_tensor(interpreter.get_output_details()[1]['index'])[0]  # Detection scores
        scores = interpreter.get_tensor(interpreter.get_output_details()[2]['index'])[0]  # Class IDs
        # num_detections = interpreter.get_tensor(interpreter.get_output_details()[3]['index'])[0] # Sometimes available

        detected_items_in_frame = [] # Store items found in this frame
        original_h, original_w, _ = img_bgr.shape

        for i in range(len(scores)):
            if scores[i] > CONFIDENCE_THRESHOLD:
               # print(boxes,classes,scores)
                class_id = int(classes[i])
                label = labels.get(class_id, "unknown")
                
                #print("LABEL",label,"CLASS_ID",class_id)
                #print(f"DEBUG: Detected '{label}' (ID: {class_id}) with score {scores[i]:.2f}")
                
                # --- MODIFICATION START ---
                if label in TARGET_LABELS: # Check if the detected label is in our list
                    # apple_detected_this_frame = True # Old flag
                    if label not in detected_items_in_frame: # Avoid duplicate messages for the same item type per frame
                        detected_items_in_frame.append(label)

                    #print(f"INFO: Detected {label} with score {scores[i]:.2f}")

                    # Optional: Draw bounding box
                    ymin, xmin, ymax, xmax = boxes[i]
                    left, right, top, bottom = (xmin * original_w, xmax * original_w,
                                                ymin * original_h, ymax * original_h)
                    cv2.rectangle(img_bgr, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
                    cv2.putText(img_bgr, f"{label}: {scores[i]:.2f}", (int(left), int(top) - 7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    # If you only care about the first apple or banana found in a frame, you could break here.
                    # If you want to detect all apples and bananas, remove any 'break'.
                # --- MODIFICATION END ---
        
        # Send messages for all detected target items in this frame
        for item_label in detected_items_in_frame:
            payload = f"{item_label} found!".encode('utf-8')
            print(f"Sending message to '{RESULTS_SUBJECT}': {payload.decode()}")
            try:
                await nc.publish(RESULTS_SUBJECT, payload)
            except ConnectionClosedError:
                print(f"ERROR: Cannot send '{item_label} found', NATS connection closed.")
            except Exception as e:
                print(f"ERROR: Failed to publish '{item_label} found': {e}")

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
async def main_viewer(nats_url):
    nc = None
    sub = None
    interpreter_instance = None # Keep variable name distinct

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    # --- Load TFLite Model and Labels ---
    if Interpreter is None:
        print("FATAL: TFLite runtime not available. Exiting.")
        return

    try:
        labels = load_labels(LABEL_PATH)
        interpreter_instance = Interpreter(model_path=MODEL_PATH)
        interpreter_instance.allocate_tensors()
        input_details = interpreter_instance.get_input_details()
        output_details = interpreter_instance.get_output_details() # Keep for potential debugging
        input_height = input_details[0]['shape'][1]
        input_width = input_details[0]['shape'][2]
        floating_model = (input_details[0]['dtype'] == np.float32)
        print(f"Model loaded. Input shape: {input_details[0]['shape']}, Floating model: {floating_model}")
    except Exception as e:
        print(f"FATAL: Could not load TFLite model or labels: {e}")
        print(f"Ensure '{MODEL_PATH}' and '{LABEL_PATH}' are in the correct location.")
        return

    try:
        print(f"Connecting to NATS server at {nats_url}...")
        # (NATS Connection logic with callbacks - keep as you had it, or use this simplified one)
        async def disconnected_cb(): print("NATS connection lost...")
        async def reconnected_cb(): print(f"Reconnected to NATS...")
        async def error_cb(e): print(f"NATS Error: {e}")

        nc = await nats.connect(
            nats_url, connect_timeout=5, reconnect_time_wait=2,
            disconnected_cb=disconnected_cb, reconnected_cb=reconnected_cb, error_cb=error_cb
        )
        print(f"Connected to NATS! Client ID: {nc.client_id}")

        # --- Define ASYNC Callback Wrapper ---
        async def wrapped_message_handler(msg):
            await message_handler(msg, nc, interpreter_instance, input_details, labels, input_height, input_width, floating_model)

        print(f"Subscribing to frames subject: '{FRAMES_SUBJECT}'")
        sub = await nc.subscribe(FRAMES_SUBJECT, cb=wrapped_message_handler)

        # --- Optional: Send messages via keyboard input (from your previous version) ---
        async def send_messages_stdin():
            print("INFO: stdin message sender started. Type 'exit' in terminal to stop.")
            while not shutdown_event.is_set():
                try:
                    # Use asyncio.to_thread for blocking sys.stdin.readline in Python 3.9+
                    if sys.version_info >= (3, 9):
                        user_input = await asyncio.to_thread(sys.stdin.readline)
                    else: # Fallback for older Python, might still block a bit
                        user_input = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)

                    user_input = user_input.strip()
                    if not user_input and shutdown_event.is_set(): # Check if shutdown happened while reading
                        break
                    if user_input.lower() == 'exit':
                        print("INFO: 'exit' typed in terminal, initiating shutdown.")
                        shutdown_event.set()
                        break
                    if user_input: # Only send if there's actual input
                        await nc.publish(RESULTS_SUBJECT, user_input.encode())
                        print(f"Sent via stdin: {user_input}")
                except ConnectionClosedError:
                    print("WARN: stdin sender - NATS connection closed.")
                    break # Exit task if connection is closed
                except Exception as e:
                    if not shutdown_event.is_set(): # Avoid error messages during normal shutdown
                        print(f"ERROR in stdin sender: {e}")
                    break # Exit task on other errors too


        stdin_sender_task = asyncio.create_task(send_messages_stdin())

        print("Viewer running. Waiting for frames...")
        print(f"Displaying stream in window '{WINDOW_NAME}'. Press 'q' in window or Ctrl+C in terminal to exit.")
        print("Detection for 'apple' is active. Type messages in *this terminal* and press Enter to send them to Pi.")
        await shutdown_event.wait()

        print("Shutdown signal received, exiting...")
        if not stdin_sender_task.done():
            stdin_sender_task.cancel() # Attempt to cancel stdin task

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
            except: pass # Ignore errors on unsubscribe during shutdown
        if nc and nc.is_connected:
            try: await nc.drain()
            except: pass
        print("NATS connection drained/closed.")
        cv2.destroyAllWindows()
        print("Viewer finished.")

# --- Entry Point ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NATS Camera Stream Viewer with Apple Detection')
    parser.add_argument('--nats_url', type=str, default=DEFAULT_NATS_URL)
    parser.add_argument('--frames_subject', type=str, default=FRAMES_SUBJECT)
    parser.add_argument('--results_subject', type=str, default=RESULTS_SUBJECT)
    parser.add_argument('--model', type=str, default=MODEL_PATH, help="Path to TFLite model file")
    parser.add_argument('--labels', type=str, default=LABEL_PATH, help="Path to labels file")

    args = parser.parse_args()

    FRAMES_SUBJECT = args.frames_subject
    RESULTS_SUBJECT = args.results_subject
    MODEL_PATH = args.model
    LABEL_PATH = args.labels

    print("Starting NATS Stream Viewer with Apple Detection...")
    print(f" --- NATS Server: {args.nats_url}")
    print(f" --- Subscribing To Frames: {FRAMES_SUBJECT}")
    print(f" --- Publishing Results To: {RESULTS_SUBJECT}")
    print(f" --- Using Model: {MODEL_PATH}")
    print(f" --- Using Labels: {LABEL_PATH}")

    try:
        asyncio.run(main_viewer(args.nats_url))
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught in top level.")
    print("Application terminated.")
    