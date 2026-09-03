"""
Simple Face Mask Detection with OpenCV Haar Cascades.

Run it with:  python main.py
Press "q" to quit.
"""

import os
import sys

import cv2
import numpy as np


CASCADE_FILENAME = "haarcascade_frontalface_default.xml"

MASK_SKIN_RATIO_THRESHOLD = 0.5

CAMERA_INDEX = 0


def load_face_cascade():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(script_dir, CASCADE_FILENAME)

    bundled_path = os.path.join(cv2.data.haarcascades, CASCADE_FILENAME)

    if os.path.exists(local_path):
        cascade_path = local_path
    elif os.path.exists(bundled_path):
        cascade_path = bundled_path
    else:
        print("ERROR: Could not find the Haar Cascade file.")
        print(f"  Looked for a local copy at: {local_path}")
        print(f"  Looked for the bundled copy at: {bundled_path}")
        print("  Fix: copy 'haarcascade_frontalface_default.xml' next to main.py,")
        print("  or reinstall OpenCV with:  pip install opencv-python")
        sys.exit(1)

    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        print("ERROR: Found the cascade file but failed to load it.")
        print(f"  Path: {cascade_path}")
        print("  The file may be corrupted. Try reinstalling opencv-python.")
        sys.exit(1)

    print(f"Loaded face cascade from: {cascade_path}")
    return face_cascade


def skin_ratio(bgr_region):
    if bgr_region.size == 0:
        return 0.0

    ycrcb = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2YCrCb)

    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)

    skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

    skin_pixels = int(cv2.countNonZero(skin_mask))
    total_pixels = bgr_region.shape[0] * bgr_region.shape[1]
    if total_pixels == 0:
        return 0.0
    return skin_pixels / total_pixels


def is_wearing_mask(face_bgr):
    height = face_bgr.shape[0]
    mid = height // 2

    upper_half = face_bgr[0:mid, :]
    lower_half = face_bgr[mid:height, :]

    upper_skin = skin_ratio(upper_half)
    lower_skin = skin_ratio(lower_half)

    if upper_skin <= 0.01:
        mask_present = False
    else:
        mask_present = (lower_skin < upper_skin * MASK_SKIN_RATIO_THRESHOLD)

    debug_info = {"upper_skin": upper_skin, "lower_skin": lower_skin}
    return mask_present, debug_info


def draw_result(frame, x, y, w, h, mask_present):
    if mask_present:
        color = (0, 200, 0)
        label = "MASK DETECTED"
        warning = None
    else:
        color = (0, 0, 255)
        label = "NO MASK"
        warning = "PLEASE WEAR A MASK"

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    label_y = y - 10 if y - 10 > 10 else y + h + 20
    cv2.putText(frame, label, (x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if warning is not None:
        cv2.putText(frame, warning, (x, y + h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main():
    face_cascade = load_face_cascade()

    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        print("ERROR: Could not open the webcam.")
        print(f"  Tried camera index {CAMERA_INDEX}.")
        print("  Make sure a camera is connected and not in use by another app,")
        print("  or change CAMERA_INDEX near the top of this file.")
        sys.exit(1)

    print("Webcam opened. Press 'q' in the video window to quit.")

    while True:
        ok, frame = camera.read()
        if not ok or frame is None:
            print("ERROR: Failed to read a frame from the webcam. Stopping.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
        )

        for (x, y, w, h) in faces:
            face_region = frame[y:y + h, x:x + w]
            mask_present, _ = is_wearing_mask(face_region)
            draw_result(frame, x, y, w, h, mask_present)

        cv2.putText(frame, "Press 'q' to quit", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Face Mask Detection (Haar Cascade demo)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    print("Stopped. Webcam released and windows closed.")


if __name__ == "__main__":
    main()
