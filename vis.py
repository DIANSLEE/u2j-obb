from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("C:/Users/dians/Desktop/ultralytics-main/runs/obb/train22/weights/best.pt")

results = model.predict("C:/academic/paper/result_r2/104imgjson/IMG_20230507_085933.jpg")

for result in results:
    img = result.orig_img.copy()

    if result.obb is not None and len(result.obb) > 0:
        xywhr = result.obb.xywhr.cpu().numpy()

        for box in xywhr:
            cx, cy, w, h, angle_rad = box
            angle_deg = np.degrees(angle_rad)

            rect = ((cx, cy), (w, h), angle_deg)
            pts = cv2.boxPoints(rect)
            pts = np.int32(pts)
            cv2.drawContours(img, [pts], 0, (0, 255, 0), 2)

            arrow_len = 80
            ex = int(cx + arrow_len * np.cos(angle_rad))
            ey = int(cy + arrow_len * np.sin(angle_rad))
            cv2.arrowedLine(img, (int(cx), int(cy)), (ex, ey), (0, 165, 255), 3, tipLength=0.3)
            cv2.putText(img, f'{angle_deg:.1f}', (int(cx)+5, int(cy)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 2)

    cv2.imwrite("result_arrow.jpg", img)
