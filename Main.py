import cv2
import numpy as np
import time
import json
import os
from scipy.spatial import cKDTree
from skimage.registration import phase_cross_correlation
from skimage import img_as_float
import math
FLAG_PATH = 'tracking_enabled.flag'

def set_tracking(enabled: bool):
    tmp_path = FLAG_PATH + '.tmp'
    with open(tmp_path, 'w') as f:
        f.write('1' if enabled else '0')
    os.replace(tmp_path, FLAG_PATH)

def is_tracking_enabled():
    try:
        with open(FLAG_PATH, 'r') as f:
            return f.read().strip() == '1'
    except:
        return False

def t_tracking_flag():
    set_tracking(not is_tracking_enabled())

def save_offset(dx, dy, angle):
    data = {'dx': float(dx), 'dy': float(dy), 'angle': float(angle)}
    with open('offsets_tmp.json', 'w') as f:
        json.dump(data, f)
    os.replace('offsets_tmp.json', 'offsets.json')

def add_waypoint(wp_list, pts, ang, gray_frame):
    if len(pts) == 0:
        return
    wp_list.append({
        'points': pts.copy(),
        'angle': ang,
        'frame_hash': hash(gray_frame.tobytes())
    })

def adaptive_good_features(gray, min_features=100, max_features=1000):
    mean_val, std_val = cv2.meanStdDev(gray)
    quality_level = max(0.01, 0.1 * (1 - std_val / 50))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    height, width = gray.shape
    area = height * width
    num_features = max(min_features, min(max_features, int(area / 500)))
    min_distance = max(5, int(np.sqrt(area / num_features)))
    pts = cv2.goodFeaturesToTrack(
        enhanced,
        maxCorners=num_features,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=7,
        useHarrisDetector=False
    )
    return pts

def points_moved(p1, p2, thresh):
    if p1.shape != p2.shape or len(p1) == 0 or len(p2) == 0:
        return True
    d = np.linalg.norm(p1 - p2, axis=1)
    return np.mean(d) > thresh

def angle_diff(a, b):
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d

def mean_offset(p_new, p_ref):
    if len(p_new) == 0 or len(p_ref) == 0:
        return 0.0, 0.0
    tree = cKDTree(p_ref)
    d, idx = tree.query(p_new)
    offs = p_new - p_ref[idx]
    return np.mean(offs, axis=0)

def find_similar(current, waypoints, pos_thresh=5.0, angle_thresh=5.0):
    best_idx = -1
    best_score = float('inf')
    for i, wp in enumerate(waypoints):
        if len(wp['points']) == 0 or len(current) == 0:
            continue
        tree = cKDTree(wp['points'])
        d, _ = tree.query(current)
        pos_score = np.mean(d)
        ang_score = angle_diff(wp['angle'], 0)
        total_score = pos_score + 0.3 * ang_score
        if total_score < best_score and pos_score < pos_thresh and ang_score < angle_thresh:
            best_score = total_score
            best_idx = i
    return best_idx

def is_returned_to_anchor(good_new, anchor_pts, current_angle, anchor_angle, frame, anchor_frame):
    offset_x, offset_y = mean_offset(good_new, anchor_pts)
    pos_drift = np.hypot(offset_x, offset_y)
    if pos_drift > 3.0:
        return False
    ang_drift = angle_diff(current_angle, anchor_angle)
    if ang_drift > 3.0:
        return False
    try:
        shift, _, _ = phase_cross_correlation(
            img_as_float(anchor_frame), img_as_float(frame), upsample_factor=10
        )
        if np.linalg.norm(shift) > 2.0:
            return False
    except:
        return False
    return True

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    lw, lh = 720, 576
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    anchor_gray = None
    anchor_pts = None
    anchor_ang = 0.0
    anchor_frame_full = None
    waypoints = []
    last_wp_time = 0
    wp_interval = 1.0
    tracking = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (lw, lh))
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        if not is_tracking_enabled():
            anchor_gray = None
            anchor_pts = None
            anchor_ang = 0.0
            anchor_frame_full = None
            waypoints.clear()
            tracking = False
            vis = frame_resized.copy()
            cv2.putText(vis, "TRACKING OFF (t to toggle)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Tracker", vis)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            if k == ord('t'):
                t_tracking_flag()
            continue

        if not tracking:
            pts = adaptive_good_features(gray, min_features=150)
            if pts is not None and len(pts) > 100:
                anchor_gray = gray.copy()
                anchor_pts = pts.reshape(-1, 2).copy()
                anchor_ang = 0.0
                anchor_frame_full = frame_resized.copy()
                waypoints.clear()
                add_waypoint(waypoints, anchor_pts, anchor_ang, anchor_gray)
                last_wp_time = time.time()
                tracking = True
            cv2.waitKey(1)
            continue

        next_pts, st, err = cv2.calcOpticalFlowPyrLK(anchor_gray, gray, anchor_pts.reshape(-1, 1, 2), None, **lk_params)
        if next_pts is None or st is None:
            continue
        st = st.flatten()
        good_new = next_pts[st == 1].reshape(-1, 2)
        good_old = anchor_pts[st == 1].reshape(-1, 2)
        if len(good_new) < 50:
            continue

        H, inliers = cv2.estimateAffinePartial2D(
            good_old.reshape(-1, 1, 2),
            good_new.reshape(-1, 1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0
        )
        current_ang = math.degrees(math.atan2(H[1, 0], H[0, 0])) if H is not None else anchor_ang

        now = time.time()

        if points_moved(good_new, waypoints[-1]['points'], 3.0) and now - last_wp_time > wp_interval:
            add_waypoint(waypoints, good_new, current_ang, gray)
            last_wp_time = now

        similar_idx = find_similar(good_new, waypoints, pos_thresh=5.0, angle_thresh=5.0)
        if similar_idx != -1 and similar_idx < len(waypoints) - 1:
            waypoints = waypoints[:similar_idx + 1]
            last_wp_time = now

        if is_returned_to_anchor(good_new, anchor_pts, current_ang, anchor_ang, gray, anchor_gray):
            waypoints = [waypoints[0]]
            last_wp_time = now

        total_dx, total_dy = mean_offset(good_new, anchor_pts)
        total_ang = angle_diff(current_ang, anchor_ang)
        save_offset(total_dx, total_dy, total_ang)

        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for i, wp in enumerate(waypoints):
            color = (255, 0, 0) if i > 0 else (0, 255, 0)
            for p in wp['points']:
                cv2.circle(vis, (int(p[0]), int(p[1])), 3, color, -1)
        for p in good_new:
            cv2.circle(vis, (int(p[0]), int(p[1])), 2, (0, 255, 255), -1)

        cv2.putText(vis, f"Waypoints: {len(waypoints)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, f"dx:{total_dx:.2f} dy:{total_dy:.2f} ang:{total_ang:.2f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if len(waypoints) == 1:
            cv2.putText(vis, "ANCHOR LOCKED", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Tracker", vis)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if k == ord('t'):
            t_tracking_flag()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()