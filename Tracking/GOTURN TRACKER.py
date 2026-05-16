import cv2
import os
import sys
import time

# Define GOTURN model paths (relative to current directory)
PROTOTXT = "opencv_extra/testdata/dnn/goturn.prototxt"
CAFFEMODEL = "opencv_extra/testdata/dnn/goturn.caffemodel"

def validate_goturn_model():
    if not os.path.exists(PROTOTXT) or not os.path.exists(CAFFEMODEL):
        print("\n❌ GOTURN model files not found.")
        print("Expected files:")
        print(f"  - {os.path.abspath(PROTOTXT)}")
        print(f"  - {os.path.abspath(CAFFEMODEL)}")
        print("Please place them in the paths shown above.\n")
        sys.exit(1)

def main():
    validate_goturn_model()

    # Set up GOTURN parameters
    params = cv2.TrackerGOTURN_Params()
    params.modelTxt = os.path.abspath(PROTOTXT)
    params.modelBin = os.path.abspath(CAFFEMODEL)

    # Create GOTURN tracker
    tracker = cv2.TrackerGOTURN_create(params)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Failed to open webcam.")
        return

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read from webcam.")
        return

    # Let user select ROI
    bbox = cv2.selectROI("Select Object to Track", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    # Initialize tracker
    tracker.init(frame, bbox)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Track object
        start = time.time()
        success, bbox = tracker.update(frame)
        end = time.time()

        if success:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"GOTURN: {1000 * (end - start):.1f} ms"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Tracking Lost", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.imshow("GOTURN Tracker", frame)

        key = cv2.waitKey(1)
        if key == 27:  # ESC key to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()