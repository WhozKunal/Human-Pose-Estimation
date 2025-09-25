import cv2

# COCO skeleton connections (17 keypoints)
SKELETON = [
    (0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),
    (6,8),(8,10),(5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16)
]

def draw_pose(image, keypoints_xy, keypoints_conf, threshold=0.5):
    """Draws keypoints and skeleton on the image."""
    for kpts, confs in zip(keypoints_xy, keypoints_conf):
        kpts, confs = kpts.cpu().numpy(), confs.cpu().numpy()
        
        # Draw keypoints
        for i, (x, y) in enumerate(kpts):
            if confs[i] > threshold:
                cv2.circle(image, (int(x), int(y)), 4, (0, 0, 255), -1)
        
        # Draw skeleton
        for (s, e) in SKELETON:
            if confs[s] > threshold and confs[e] > threshold:
                start_pt = tuple(map(int, kpts[s]))
                end_pt   = tuple(map(int, kpts[e]))
                cv2.line(image, start_pt, end_pt, (255, 0, 0), 2)
    return image
