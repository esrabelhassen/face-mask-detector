import cv2
from ultralytics import YOLO

model = YOLO(r"C:\Users\belha\Downloads\best.pt")

GREEN  = (0, 255, 0)
RED    = (0, 0, 255)
YELLOW = (0, 255, 255)

CLASS_COLORS = {
    "with_mask":    GREEN,
    "without_mask": RED,
}

cap = cv2.VideoCapture(0)
print("Running — press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf  = float(box.conf[0])
        cls   = int(box.cls[0])
        label = model.names[cls]
        color = CLASS_COLORS.get(label, YELLOW)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{label} {conf:.0%}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, color, 2
        )

    cv2.imshow("mask Detector /press Q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
