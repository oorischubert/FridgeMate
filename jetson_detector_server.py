import cv2
import numpy as np
import signal
import argparse
import time
import os # Kept for path checks if needed by utility, can be removed if not

# >>> HAND TRACKING IMPORT <<<
# Ensure this path is correct relative to where you run the script.
# If 'utility' is in the same directory as this script, it should work.
try:
    import utility.handTracking as htm
except ImportError:
    print("ERROR: Failed to import utility.handTracking as htm.")
    print("Ensure 'utility' directory is in the Python path and contains handTracking.py")
    htm = None # To allow script to load but fail gracefully if hand tracking is not used

# --- Configuration ---
WINDOW_NAME = "FridgeMate Hand Tracking Stream"
DEFAULT_CAMERA_INDEX = 0 # Default USB camera index

# --- Hand Tracking Configuration ---
MAX_HANDS = 1
hand_detector = None # Initialize later if htm is available
baseline_hand_width = None
current_relative_distance = 1.0 # Default relative distance
SMOOTHING_FACTOR = 0.3 # Simple exponential smoothing for distance

# --- FPS Measurement Globals ---
prev_frame_time = 0.0 # Initialize to 0.0, will be set on first frame
fps_display = 0.0              # exponentially smoothed FPS value

# --- Global flag for shutdown ---
shutdown_requested = False

# --- Signal Handler ---
def handle_signal(sig, frame):
    """Handles termination signals."""
    global shutdown_requested
    print(f"\nReceived signal {sig}, initiating shutdown...")
    shutdown_requested = True

def process_frame_hand_tracking(img_bgr):
    """Processes a single frame for hand tracking."""
    global baseline_hand_width, current_relative_distance, prev_frame_time, fps_display, hand_detector

    if img_bgr is None:
        print("WARN: process_frame_hand_tracking received a None image.")
        return None

    original_h, original_w, _ = img_bgr.shape

    # --- FPS calculation ---
    now = time.time()
    if prev_frame_time > 0: # Ensure prev_frame_time is initialized and not the first frame
        instantaneous_fps = 1.0 / max(now - prev_frame_time, 1e-6) # Avoid division by zero
        fps_display = 0.9 * fps_display + 0.1 * instantaneous_fps
    prev_frame_time = now
    
    # --- HAND TRACKING START ---
    if hand_detector: # Only run if hand_detector was initialized successfully
        # 1. Find Hands and draw basic landmarks/connections
        img_bgr = hand_detector.findHands(img_bgr, draw=True)
        # 2. Get landmark positions (without drawing default circles/box)
        lmList, bbox = hand_detector.findPosition(img_bgr, draw=False)

        if len(lmList) != 0:
            try:
                # 3. Calculate distance between base of index (5) and pinky (17) as proxy for hand size/distance
                current_width, _, _ = hand_detector.findDistance(5, 17, img_bgr, draw=False)

                if current_width > 10: # Basic check to avoid tiny erroneous widths
                    if baseline_hand_width is None:
                        baseline_hand_width = current_width
                        print(f"INFO: Captured baseline hand width: {baseline_hand_width:.2f}")
                        current_relative_distance = 1.0
                    else:
                        # Calculate relative distance (smaller width means further away)
                        relative_dist = baseline_hand_width / max(current_width, 1e-6) # Avoid division by zero
                        # Apply smoothing
                        current_relative_distance = (SMOOTHING_FACTOR * relative_dist) + \
                                                    ((1 - SMOOTHING_FACTOR) * current_relative_distance)
                    
                # Display the relative distance on the image
                cv2.putText(img_bgr, f"Rel Dist: {current_relative_distance:.2f}", (10, original_h - 30), # Adjusted y-pos
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
            except Exception as e:
                # This can happen if findDistance fails (e.g., landmarks not visible)
                print(f"ERROR in hand distance calculation: {e}")
                # Optionally display an error or "--" for distance if calculation fails
                cv2.putText(img_bgr, f"Rel Dist: Error", (10, original_h - 30),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        # Optional: If hand disappears, display placeholder for distance
        # else: # No hand detected
        #     cv2.putText(img_bgr, f"Rel Dist: --", (10, original_h - 30),
        #                 cv2.FONT_HERSHEY_PLAIN, 2, (0, 100, 100), 2)
    # --- HAND TRACKING END ---

    # Draw FPS in the top‑left corner
    cv2.putText(img_bgr, f"FPS: {fps_display:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return img_bgr

# --- Main Application Logic ---
def main_local_camera_hand_tracking(camera_index):
    """Main function to run local camera viewer with hand tracking."""
    global shutdown_requested, hand_detector, prev_frame_time # Allow modification

    # Register signal handlers
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Initialize Hand Detector if htm module is available
    if htm:
        try:
            hand_detector = htm.handDetector(maxHands=MAX_HANDS, detectionCon=0.7)
            print("INFO: Hand detector initialized.")
        except Exception as e:
            print(f"ERROR: Could not initialize handDetector: {e}")
            hand_detector = None # Ensure it's None if initialization fails
    else:
        print("WARN: Hand tracking module (htm) not loaded. Hand detection will be skipped.")


    # --- Initialize Camera ---
    print(f"INFO: Attempting to open camera at index {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    
    # Try GStreamer pipeline if direct V4L2 access fails (common on Jetson for CSI cameras)
    if not cap.isOpened() and camera_index == 0: # Only try GStreamer for default index 0
        print(f"INFO: Camera at index {camera_index} failed. Trying GStreamer pipeline for Jetson camera...")
        # Common GStreamer pipeline for Jetson CSI cameras
        # Adjust width, height, framerate as needed for your camera
        gst_pipeline = (
            "nvarguscamerasrc sensor-id=0 ! " # sensor-id might be 0 or 1 depending on your setup
            "video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! "
            "nvvidconv flip-method=0 ! " # flip-method: 0=none, 1=counterclockwise, 2=rotate180, 3=clockwise
            "video/x-raw, width=(int)1280, height=(int)720, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink drop=1"
        )
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print("INFO: Successfully opened camera using GStreamer pipeline.")
        else:
            print("FATAL: Still cannot open camera using GStreamer pipeline. Check camera connection and drivers.")
            return
    elif not cap.isOpened():
        print(f"FATAL: Cannot open camera at index {camera_index}. Exiting.")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    print(f"INFO: Displaying stream in window '{WINDOW_NAME}'. Press 'q' in window or Ctrl+C in terminal to exit.")

    prev_frame_time = time.time() # Initialize prev_frame_time before the loop starts

    while not shutdown_requested:
        ret, frame = cap.read()
        if not ret:
            print("WARN: Cannot read frame from camera. End of stream or camera error?")
            # Consider adding a counter to break after several failed attempts
            time.sleep(0.1) # Wait a bit before trying again
            continue 

        try:
            processed_frame = process_frame_hand_tracking(frame)

            if processed_frame is not None:
                cv2.imshow(WINDOW_NAME, processed_frame)
            else:
                print("WARN: Frame processing returned None.")
                # Optionally display the original frame if processing fails
                # cv2.imshow(WINDOW_NAME, frame) 

        except Exception as e:
            print(f"ERROR during frame processing or display: {e}")
            import traceback
            traceback.print_exc()
            # break # Optionally break the loop on critical errors

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("INFO: 'q' key pressed in window. Initiating shutdown...")
            shutdown_requested = True
            break 

    # --- Cleanup ---
    print("INFO: Cleaning up and closing resources...")
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    print("INFO: Application terminated.")

# --- Entry Point ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Local Camera Stream with Hand Tracking')
    parser.add_argument('--camera_index', type=int, default=DEFAULT_CAMERA_INDEX,
                        help=f"Index of the camera to use (default: {DEFAULT_CAMERA_INDEX})")
    # Add other arguments if needed, e.g., for hand tracking parameters

    args = parser.parse_args()

    print("Starting Local Camera Stream with Hand Tracking...")
    print(f" --- Using Camera Index: {args.camera_index}")
    
    try:
        main_local_camera_hand_tracking(args.camera_index)
    except KeyboardInterrupt:
        print("\nINFO: KeyboardInterrupt caught in main. Exiting gracefully.")
        shutdown_requested = True # Ensure flag is set for cleanup if loop was exited abruptly
    except Exception as e:
        print(f"FATAL ERROR in __main__: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure cleanup happens even if main function exits unexpectedly
        if 'cap' in locals() and cap.isOpened(): # Check if cap was defined and is open
            cap.release()
        cv2.destroyAllWindows()
        print("INFO: Main process finished.")
