import cv2
from ultralytics import YOLO
from config import (DET_MODEL_PATH, POSE_MODEL_PATH, IMAGE_PATH, SAVE_PATH, 
                    CONF_THRESHOLD, IOU_THRESHOLD, DEVICE)
from pose_utils import draw_pose


def main():
    # Load models
    det_model = YOLO(DET_MODEL_PATH)
    pose_model = YOLO(POSE_MODEL_PATH)

    # Load image
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"Error: Could not load image {IMAGE_PATH}")
        return

    # Detect people
    det_results = det_model(
        image,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        classes=[0],   # class 0 = person in COCO
        device=DEVICE,
        verbose=False
    )

    # Copy for visualization
    vis_image = image.copy()

    # Draw bounding boxes for detected people
    for result in det_results:
        for box in result.boxes:   # YOLO stores detections in result.boxes
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)  # bounding box coords
            conf = float(box.conf[0])  # confidence score

            # Draw rectangle
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_image, f"Person {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Pose estimation
    pose_results = pose_model(
        image,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        classes=[0],
        device=DEVICE,
        verbose=False
    )

    # Draw keypoints
    for result in pose_results:
        if result.keypoints is not None:
            vis_image = draw_pose(vis_image, result.keypoints.xy, result.keypoints.conf)

    # Save result
    cv2.imwrite(SAVE_PATH, vis_image)
    print(f"Output saved at {SAVE_PATH}")


if __name__ == "__main__":
    main()
