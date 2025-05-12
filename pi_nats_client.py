import cv2
import numpy as np
import asyncio
from asyncio import Queue
import nats
from nats.errors import ConnectionClosedError, TimeoutError, NoServersError
import signal
import argparse
import os
import time

# Attempt to import PiCamera specific library
try:
    from picamera2 import Picamera2
    picamera2_imported = True
except ImportError:
    picamera2_imported = False
    print("INFO: picamera2 library not found. Will try OpenCV for camera.")


# --- Configuration ---
DEFAULT_NATS_URL = "nats://localhost:4222" # CHANGE if your NATS server is elsewhere
FRAMES_SUBJECT = "camera.frames"       # Subject to publish frames TO
RESULTS_SUBJECT = "detection.results"  # Subject to receive results FROM
CAMERA_INDEX = 0                       # Default USB camera index if not using PiCamera
FRAME_WIDTH = 640                      # Lower resolution is better for streaming
FRAME_HEIGHT = 480
JPEG_QUALITY = 75                      # Adjust quality vs size (0-100)
PUBLISH_DELAY_SECONDS = 0.05           # Target ~20 FPS (adjust as needed)

# --- Global flag for shutdown ---
shutdown_event = asyncio.Event()

# Buffer for outgoing JPEG frames (capture → publish). Keeps only a few newest.
outgoing_frames: Queue = Queue(maxsize=3)

# --- Signal Handler ---
def handle_signal(sig, frame):
    print(f"\nReceived signal {sig}, initiating shutdown...")
    shutdown_event.set()

# --- Camera Setup ---
def setup_camera(use_picamera, width, height):
    """Initializes and returns the camera object."""
    camera = None
    is_picamera = False
    if use_picamera and picamera2_imported:
        try:
            print("Setting up PiCamera2...")
            picam2 = Picamera2()
            # Configure for video recording, low-res preview helpful for performance
            config = picam2.create_video_configuration(
                main={"size": (width, height), "format": "RGB888"},
                lores={"size": (320, 240), "format": "YUV420"}, # Small lores stream if needed
                encode="main" # Encode the main stream
            )
            picam2.configure(config)
            # Limit sensor FPS to ~10 fps to lighten load
            picam2.set_controls({"FrameDurationLimits": (100000, 100000)})  # 10 fps
            # Optional: Tweak sensor settings for framerate/exposure if needed
            # picam2.set_controls({"FrameRate": 30}) # Example: Request 30fps
            picam2.start()
            time.sleep(1.0) # Allow camera to initialize
            print("PiCamera2 setup complete.")
            camera = picam2
            is_picamera = True
        except Exception as e:
            print(f"ERROR: Failed to initialize PiCamera2: {e}")
            print("Falling back to OpenCV for USB Cam...")
            is_picamera = False # Ensure fallback flag is correct
            pass # Let it try OpenCV next

    if not is_picamera: # Either PiCam failed or wasn't requested/available
        print(f"Setting up OpenCV VideoCapture (Index: {CAMERA_INDEX})...")
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            print(f"ERROR: Cannot open camera at index {CAMERA_INDEX}")
            return None, False # Return None if camera fails
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # Optional: Try setting FPS for USB cams, might not always work
        # cap.set(cv2.CAP_PROP_FPS, 30)
        print("OpenCV VideoCapture setup complete.")
        camera = cap
        is_picamera = False

    return camera, is_picamera

# --- Frame Capture ---
def capture_frame(camera, is_picamera):
    """Captures a single frame (as RGB) from the camera object."""
    if is_picamera:
        # capture_array returns RGB numpy array directly
        frame = camera.capture_array("main") # Capture from main stream
        return frame
    else:
        # OpenCV captures BGR by default
        ret, frame_bgr = camera.read()
        if not ret:
            print("WARN: Failed to grab frame from OpenCV camera")
            return None
        # Convert BGR to RGB if needed downstream, but sending BGR is fine too
        # frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # return frame_rgb
        return frame_bgr # Send BGR directly, server can handle it

async def capture_loop(camera, is_picamera):
    """Capture frames, JPEG‑encode, and push latest into outgoing queue."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    while not shutdown_event.is_set():
        frame = capture_frame(camera, is_picamera)
        if frame is None:
            await asyncio.sleep(0.05)
            continue
        ok, encoded = cv2.imencode('.jpg', frame, encode_param)
        if not ok:
            continue
        # Keep only the most recent frame
        if outgoing_frames.full():
            _ = outgoing_frames.get_nowait()
        await outgoing_frames.put(encoded.tobytes())
        await asyncio.sleep(PUBLISH_DELAY_SECONDS)

async def publish_loop(nc):
    """Publish encoded JPEG bytes from queue to NATS."""
    while not shutdown_event.is_set():
        try:
            data = await outgoing_frames.get()
            await nc.publish(FRAMES_SUBJECT, data)
        except Exception as e:
            print(f"ERROR publishing frame: {e}")
            await asyncio.sleep(0.1)

# --- NATS Message Handler ---
async def message_handler(msg):
    """Callback function to handle received messages."""
    subject = msg.subject
    data = msg.data.decode()
    print(f"Received result on '{subject}': {data}")
    # --- Add your logic here ---
    # e.g., display on an LCD, print formatted output, etc.

# --- Main Application Logic ---
async def main_loop(nats_url, use_pi_cam):
    nc = None # NATS connection object
    camera = None # Camera object

    try:
        # --- Setup Camera ---
        camera, is_picamera = setup_camera(use_pi_cam, FRAME_WIDTH, FRAME_HEIGHT)
        if camera is None:
            print("FATAL: Could not initialize any camera. Exiting.")
            return # Exit if no camera

        # --- Connect to NATS ---
        print(f"Connecting to NATS server at {nats_url}...")
        try:
            nc = await nats.connect(nats_url, connect_timeout=5, reconnect_time_wait=2)
            print(f"Connected to NATS! Client ID: {nc.client_id}")
        except NoServersError as e:
            print(f"FATAL: Could not connect to any of the specified NATS servers: {e}")
            return
        except ConnectionRefusedError:
            print(f"FATAL: Connection refused by NATS server at {nats_url}.")
            return
        except TimeoutError:
            print(f"FATAL: Timeout connecting to NATS server at {nats_url}.")
            return
        except Exception as e:
            print(f"FATAL: An unexpected error occurred during NATS connection: {e}")
            return


        # --- Setup Subscription ---
        print(f"Subscribing to results subject: '{RESULTS_SUBJECT}'")
        await nc.subscribe(RESULTS_SUBJECT, cb=message_handler)

        # Launch producer/consumer tasks
        capture_task = asyncio.create_task(capture_loop(camera, is_picamera))
        publish_task = asyncio.create_task(publish_loop(nc))

        # Wait for shutdown
        await shutdown_event.wait()

        # Cancel background tasks gracefully
        capture_task.cancel()
        publish_task.cancel()
        await asyncio.gather(capture_task, publish_task, return_exceptions=True)

        print("Shutdown signal received, exiting publish loop.")

    except Exception as e:
        print(f"ERROR in main loop: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
    finally:
        print("Cleaning up...")
        if nc and nc.is_connected:
            print("Draining and closing NATS connection...")
            await nc.drain() # Waits for outgoing buffer to clear before closing
            print("NATS connection closed.")
        elif nc and not nc.is_connected:
            print("NATS connection already closed.")

        # Release camera
        if camera:
            if is_picamera:
                 print("Stopping PiCamera2.")
                 camera.stop()
            else:
                 print("Releasing OpenCV VideoCapture.")
                 camera.release()
        print("Cleanup complete.")


# --- Entry Point ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Raspberry Pi NATS Camera Streamer')
    parser.add_argument('--nats_url', type=str, default=DEFAULT_NATS_URL,
                        help=f'NATS server URL (default: {DEFAULT_NATS_URL})')
    parser.add_argument('--use_usb_cam', action='store_true',
                        help='Force use of USB webcam via OpenCV instead of PiCamera')
    args = parser.parse_args()

    # Determine if we should try using PiCamera
    use_pi = picamera2_imported and not args.use_usb_cam

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print("Starting NATS Camera Client...")
    print(f" --- Target NATS Server: {args.nats_url}")
    print(f" --- Publishing Frames To: {FRAMES_SUBJECT}")
    print(f" --- Listening for Results On: {RESULTS_SUBJECT}")
    print(f" --- Using {'PiCamera2' if use_pi else 'OpenCV (USB Cam)'}")

    # Run the main asynchronous loop
    try:
        asyncio.run(main_loop(args.nats_url, use_pi))
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught in top level, ensuring shutdown.")
        # Event should already be set by signal handler, but set again just in case
        shutdown_event.set()
        # asyncio.run should handle cleanup via finally block in main_loop

    print("Application finished.")

# sudo apt install pip
# sudo apt install python3-opencv python3-numpy
# pip install nats-py --break-system-packages
# sudo apt install python3-picamera2

# sudo apt update && sudo apt install -y pip python3-opencv python3-numpy python3-picamera2 && pip install nats-py --break-system-packages

# python pi_nats_client.py --nats_url nats://192.168.1.22:4222

# Issues with camera not being opened:
# sudo apt install lsof
# lsof /dev/video0
# kill -9 <PID>