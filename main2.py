import cv2
from ultralytics import YOLO
from config import (DET_MODEL_PATH, POSE_MODEL_PATH, VIDEO_PATH, SAVE_PATH2, CONF_THRESHOLD, IOU_THRESHOLD, DEVICE)
from pose_utils import draw_pose

def main():
    # Load models
    det_model = YOLO(DET_MODEL_PATH)
    pose_model = YOLO(POSE_MODEL_PATH)

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH}")
        return

    # Get video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    # Video writer to save output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    out = cv2.VideoWriter(SAVE_PATH2, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect people
        det_results = det_model(
            frame,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            classes=[0],
            device=DEVICE,
            verbose=False
        )

        # Copy frame for visualization
        vis_frame = frame.copy()

        # Draw bounding boxes
        for result in det_results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis_frame, f"Person {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Pose estimation
        pose_results = pose_model(
            frame,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            classes=[0],
            device=DEVICE,
            verbose=False
        )

        # Draw keypoints
        for result in pose_results:
            if result.keypoints is not None:
                vis_frame = draw_pose(vis_frame, result.keypoints.xy, result.keypoints.conf)

        # Write frame to output video
        out.write(vis_frame)

        # Optional: display frame in real-time
        cv2.imshow('Pose Estimation', vis_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Output video saved at {SAVE_PATH2}")

if __name__ == "__main__":
    main()
