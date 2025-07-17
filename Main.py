import cv2
import numpy as np
import time
import json
import os
import logging
import math
from scipy.spatial import cKDTree

logging.basicConfig(filename='anchor_tracking.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

FLAG_PATH = 'tracking_enabled.flag'

def set_tracking(enabled: bool):
    tmp_path = FLAG_PATH + '.tmp'
    try:
        with open(tmp_path, 'w') as f:
            f.write('1' if enabled else '0')
        os.replace(tmp_path, FLAG_PATH)
        logging.info(f"Tracking {'enabled' if enabled else 'disabled'}")
    except Exception as e:
        logging.error(f"Error writing tracking flag: {e}")

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
    try:
        with open('offsets_tmp.json', 'w') as f:
            json.dump(data, f)
        os.replace('offsets_tmp.json', 'offsets.json')
        logging.debug(f"Saved offset dx={dx:.2f}, dy={dy:.2f}, angle={angle:.2f}")
    except Exception as e:
        logging.error(f"Error saving offsets: {e}")

def add_waypoint(wp_list, pts, ang):
    wp_list.append({'points': pts.copy(), 'angle': ang})
    logging.info(f"Added waypoint #{len(wp_list)-1} angle={ang:.2f}")

def points_moved(p1, p2, thresh):
    if p1.shape != p2.shape or len(p1) == 0:
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
    return offs[:, 0].mean(), offs[:, 1].mean()

def find_similar(current, waypoints, thresh):
    for i, wp in enumerate(waypoints):
        if len(wp['points']) == 0 or len(current) == 0:
            continue
        tree = cKDTree(wp['points'])
        d, _ = tree.query(current)
        if d.mean() < thresh:
            return i
    return -1

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    lw, lh = 720, 576
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    anchor_gray = None
    anchor_pts = None
    anchor_ang = 0.0

    waypoints = []
    last_wp_time = 0
    wp_interval = 1.0

    tracking = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if not is_tracking_enabled():
            anchor_gray = None
            anchor_pts = None
            anchor_ang = 0.0
            waypoints.clear()
            tracking = False

            vis = cv2.resize(frame, (lw, lh))
            cv2.putText(vis, "TRACKING OFF (t to toggle)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Tracker", vis)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            if k == ord('t'):
                t_tracking_flag()
            continue

        gray = cv2.cvtColor(cv2.resize(frame, (lw, lh)), cv2.COLOR_BGR2GRAY)

        if not tracking:
            pts = cv2.goodFeaturesToTrack(gray, maxCorners=2000,
                                          qualityLevel=0.1, minDistance=10)
            if pts is not None and len(pts) > 0:
                anchor_gray = gray.copy()
                anchor_pts = pts.reshape(-1, 2).copy()
                anchor_ang = 0.0

                waypoints.clear()
                add_waypoint(waypoints, anchor_pts, anchor_ang)
                last_wp_time = time.time()
                tracking = True
                logging.info("Tracking started and anchor fixed")
            cv2.waitKey(1)
            continue

        # Оптический поток с anchor_gray на текущий кадр, для anchor_pts
        next_pts, st, err = cv2.calcOpticalFlowPyrLK(anchor_gray, gray, anchor_pts.reshape(-1, 1, 2), None, **lk_params)
        if next_pts is None or st is None:
            continue
        st = st.flatten()
        good_new = next_pts[st == 1].reshape(-1, 2)
        good_old = anchor_pts[st == 1].reshape(-1, 2)
        if len(good_new) < 10:
            continue

        # Оценка поворота и смещения между anchor_pts и good_new
        H, inliers = cv2.estimateAffinePartial2D(
            good_old.reshape(-1, 1, 2), good_new.reshape(-1, 1, 2),
            method=cv2.RANSAC, ransacReprojThreshold=3)
        ang = math.degrees(math.atan2(H[1, 0], H[0, 0])) if H is not None else anchor_ang

        now = time.time()

        # Проверяем, насколько новый набор точек близок к последнему waypoint
        last_wp = waypoints[-1]
        if points_moved(good_new, last_wp['points'], 3.0) and now - last_wp_time > wp_interval:
            # Добавляем новый waypoint с текущими точками и углом относительно якоря
            add_waypoint(waypoints, good_new, ang)
            last_wp_time = now

        # Проверяем, не совпадает ли текущий набор с каким-то предыдущим waypoint
        similar_idx = find_similar(good_new, waypoints, thresh=5.0)
        if similar_idx != -1 and similar_idx < len(waypoints) - 1:
            logging.info(f"Current frame matches waypoint #{similar_idx}, trimming waypoints after it")
            waypoints = waypoints[:similar_idx + 1]
            last_wp_time = now

        # Проверяем, вернулись ли к якорю (среднее расстояние точек)
        tree = cKDTree(anchor_pts)
        dists, idxs = tree.query(good_new)
        dist_to_anchor = dists.mean()
        if dist_to_anchor < 5.0 and len(waypoints) > 1:
            logging.info("Returned close to anchor, resetting waypoints except anchor")
            # Оставляем только якорь
            waypoints = [waypoints[0]]
            last_wp_time = now

        # Сохраняем смещение и угол относительно якоря
        total_dx, total_dy = mean_offset(good_new, anchor_pts)
        total_ang = angle_diff(ang, anchor_ang)
        save_offset(total_dx, total_dy, total_ang)

        # Визуализация
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for wp in waypoints:
            for p in wp['points']:
                cv2.circle(vis, tuple(p.astype(int)), 3, (255, 0, 0), -1)
        for p in anchor_pts:
            cv2.circle(vis, tuple(p.astype(int)), 4, (0, 255, 0), -1)

        cv2.putText(vis, f"Waypoints: {len(waypoints)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, f"dx:{total_dx:.2f} dy:{total_dy:.2f} ang:{total_ang:.2f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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
