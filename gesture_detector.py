import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


# --- Part 1: Detect and Crop Hand from Image ---

def detect_and_crop_hand(image_path: str, show_steps: bool = False):
    """
    Detects a hand in a static image based on skin color,
    and crops it out.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None

    # 1. Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 2. Define a common range for skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # 3. Create a binary mask
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # 4. Clean up the mask
    mask = cv2.GaussianBlur(mask, (5, 5), 100)

    # 5. Find contours (Robust fix for different OpenCV versions)
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    if not contours:
        print("No skin-like contours found.")
        return None

    # 6. Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # 7. Get the bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)

    # 8. Crop the original image
    cropped_hand = image[y:y + h, x:x + w]

    if show_steps:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        crop_rgb = cv2.cvtColor(cropped_hand, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(img_rgb)

        plt.subplot(1, 3, 2)
        plt.title("Skin Mask")
        plt.imshow(mask, cmap='gray')

        plt.subplot(1, 3, 3)
        plt.title("Cropped Hand")
        plt.imshow(crop_rgb)
        plt.show()

    return cropped_hand


# --- Part 2: Track and Recognize in Real-Time Video ---

def recognize_in_realtime():
    """
    Captures video from webcam, tracks the hand, and counts fingers.
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        roi_x, roi_y, roi_w, roi_h = 300, 100, 300, 300
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
        roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv_roi, lower_skin, upper_skin)

        mask = cv2.GaussianBlur(mask, (5, 5), 100)

        # 4. Find contours (Robust fix for different OpenCV versions)
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        try:
            if contours:
                hand_contour = max(contours, key=cv2.contourArea)

                x, y, w, h = cv2.boundingRect(hand_contour)
                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 255), 2)

                hull = cv2.convexHull(hand_contour, returnPoints=False)
                defects = cv2.convexityDefects(hand_contour, hull)

                finger_count = 0
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(hand_contour[s][0])
                        end = tuple(hand_contour[e][0])
                        far = tuple(hand_contour[f][0])

                        a = np.linalg.norm(np.array(start) - np.array(end))
                        b = np.linalg.norm(np.array(start) - np.array(far))
                        c = np.linalg.norm(np.array(end) - np.array(far))

                        angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))

                        if angle <= np.pi / 2:  # 90 degrees
                            finger_count += 1

                gesture = f"Fingers: {finger_count + 1}"

                if finger_count == 0:
                    if cv2.contourArea(hand_contour) > 1000:
                        gesture = "Fingers: 1"
                    else:
                        gesture = "Fist"

                cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        except Exception as e:
            # This handles frames with no contours found
            pass  # Keep this for error handling

        # --- FIX 2: Correct Indentation for Display ---
        # Show the main frame (Must be outside the try/except block)
        cv2.imshow("Gesture Detection", frame)

        # Press 'q' to quit (Must be outside the try/except block)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# --- Main execution ---
if __name__ == "__main__":
    print("--- Running Image Detection (Part 1) ---")

    # Using forward slashes for the file path
    image_file = 'C:/Users/Jay/Downloads/CV_Image.jpg'

    print(f"Loading image from: {image_file}")
    cropped_image = detect_and_crop_hand(image_file, show_steps=True)

    if cropped_image is not None:
        print("Hand cropped successfully.")
    else:
        print("Could not detect a hand in the image.")

    print("\n--- Running Real-Time Recognition (Part 2) ---")
    print("Put your hand inside the green box.")
    print("Press 'q' in the video window to quit.")

    recognize_in_realtime()