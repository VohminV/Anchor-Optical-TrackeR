import cv2
import numpy as np
import time
import json
import os
import logging
import math

logging.basicConfig(filename='anchor_tracking.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')

FLAG_PATH = '/home/orangepi/Documents/YOLO/tracking_enabled.flag'

def set_tracking(enabled: bool):
    tmp_path = FLAG_PATH + '.tmp'
    try:
        with open(tmp_path, 'w') as f:
            f.write('1' if enabled else '0')
        os.replace(tmp_path, FLAG_PATH)
    except Exception as e:
        logging.error(f"Ошибка записи флага трекинга: {e}")

def is_tracking_enabled():
    if not os.path.exists(FLAG_PATH):
        return False
    try:
        with open(FLAG_PATH, 'r') as f:
            return f.read().strip() == '1'
    except Exception as e:
        logging.error(f"Ошибка чтения флага трекинга: {e}")
        return False

def t_tracking_flag():
    current = is_tracking_enabled()
    set_tracking(not current)

def save_offset(avg_x, avg_y, angle):
    data = {
        'x': float(avg_x),
        'y': float(avg_y),
        'angle': float(angle)
    }
    tmp_filename = '/home/orangepi/Documents/YOLO/offsets_tmp.json'
    final_filename = '/home/orangepi/Documents/YOLO/offsets.json'
    try:
        with open(tmp_filename, 'w') as f:
            json.dump(data, f)
        os.replace(tmp_filename, final_filename)
    except Exception as e:
        logging.error(f"Error saving offsets: {e}")

def init_kalman():
    kf = cv2.KalmanFilter(3, 3)  # состояния: dx, dy, angle
    kf.transitionMatrix = np.eye(3, dtype=np.float32)
    kf.measurementMatrix = np.eye(3, dtype=np.float32)
    kf.processNoiseCov = np.eye(3, dtype=np.float32) * 1e-4
    kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-2
    kf.statePre = np.zeros((3,1), dtype=np.float32)
    kf.statePost = np.zeros((3,1), dtype=np.float32)
    return kf

def find_nearest_waypoint(position, waypoints):
    if len(waypoints) == 0:
        return None, -1
    dists = [np.linalg.norm(position - wp) for wp in waypoints]
    idx = np.argmin(dists)
    return waypoints[idx], idx

def remove_waypoint(waypoints, idx):
    if 0 <= idx < len(waypoints):
        waypoints.pop(idx)

def main():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Anchor Optical Flow Tracker", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Anchor Optical Flow Tracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру")
        return

    proc_width, proc_height = 720, 576

    lk_params = dict(winSize=(21, 21),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    first_gray = None
    first_pts = None
    tracking_initialized = False
    prev_time = time.time()
    prev_angle = 0.0
    alpha = 0.3  # сглаживание угла
    kalman = init_kalman()

    waypoints = [np.array([0, 0])]
    last_waypoint_update = time.time()

    anchor_offset = None  # Якорь — первый накопленный оффсет
    total_offset = np.array([0.0, 0.0])  # Накопленное смещение относительно якоря

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Не удалось получить кадр с камеры")
            break

        tracking_enabled = is_tracking_enabled()

        frame_proc = cv2.resize(frame, (proc_width, proc_height))
        gray = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2GRAY)

        if not tracking_enabled:
            first_gray = None
            first_pts = None
            tracking_initialized = False
            prev_angle = 0.0
            kalman = init_kalman()
            anchor_offset = None  # Сброс якоря
            total_offset = np.array([0.0, 0.0])  # Сброс смещения при отключении трекинга

            vis = frame_proc.copy()
            cv2.putText(vis, "Tracking disabled (press 't' to enable)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Anchor Optical Flow Tracker", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t'):
                t_tracking_flag()
            continue

        if not tracking_initialized:
            pts = cv2.goodFeaturesToTrack(gray, maxCorners=300, qualityLevel=0.015, minDistance=7)
            if pts is not None and len(pts) > 0:
                first_gray = gray.copy()
                first_pts = pts
                tracking_initialized = True
                prev_angle = 0.0
                kalman = init_kalman()
                anchor_offset = None  # Якорь будет установлен при первом смещении
                total_offset = np.array([0.0, 0.0])
            else:
                vis = frame_proc.copy()
                cv2.putText(vis, "No keypoints for initialization", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Anchor Optical Flow Tracker", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('t'):
                    t_tracking_flag()
                continue

        next_pts, status, error = cv2.calcOpticalFlowPyrLK(first_gray, gray, first_pts, None, **lk_params)

        if next_pts is None or status is None:
            logging.warning("Оптический поток не найден, сброс трекинга")
            first_gray = None
            first_pts = None
            tracking_initialized = False
            prev_angle = 0.0
            kalman = init_kalman()
            anchor_offset = None
            total_offset = np.array([0.0, 0.0])
            continue

        status = status.flatten()
        good_new = next_pts[status == 1].reshape(-1, 2)
        good_old = first_pts[status == 1].reshape(-1, 2)

        if len(good_new) < 10:
            logging.warning("Слишком мало отслеживаемых точек, сброс трекинга")
            first_gray = None
            first_pts = None
            tracking_initialized = False
            prev_angle = 0.0
            kalman = init_kalman()
            anchor_offset = None
            total_offset = np.array([0.0, 0.0])
            continue

        H, inliers = cv2.estimateAffinePartial2D(
            good_old.reshape(-1, 1, 2), good_new.reshape(-1, 1, 2),
            method=cv2.RANSAC, ransacReprojThreshold=3, maxIters=2000)

        angle = prev_angle
        if H is not None and inliers is not None and np.count_nonzero(inliers) > 5:
            raw_angle_rad = math.atan2(H[1, 0], H[0, 0])
            raw_angle = math.degrees(raw_angle_rad)

            angle_diff = ((raw_angle - prev_angle + 180) % 360) - 180
            angle = prev_angle + alpha * angle_diff
            if angle > 180:
                angle -= 360
            elif angle < -180:
                angle += 360

        dxs = good_new[:, 0] - good_old[:, 0]
        dys = good_new[:, 1] - good_old[:, 1]
        avg_dx = np.mean(dxs)
        avg_dy = np.mean(dys)

        # Если якорь не установлен, установить первый кадр смещения
        if anchor_offset is None:
            anchor_offset = np.array([avg_dx, avg_dy])
            total_offset = np.array([0.0, 0.0])  # смещение относительно якоря — в нуле
        else:
            # Накопленное смещение относительно якоря
            total_offset += np.array([avg_dx, avg_dy]) - anchor_offset

        nearest_wp, nearest_idx = find_nearest_waypoint(total_offset, waypoints)

        if nearest_wp is not None and np.linalg.norm(total_offset - nearest_wp) < 5.0:
            logging.info(f"Waypoint {nearest_idx} достигнут и удалён")
            remove_waypoint(waypoints, nearest_idx)

        current_time = time.time()
        if current_time - last_waypoint_update > 1.0:
            last_waypoint_update = current_time
            if len(waypoints) == 0:
                waypoints.extend([
                    np.array([total_offset[0], total_offset[1]])
                ])
                logging.info("Добавлены новые waypoints")

        if nearest_wp is not None:
            offset_to_node = nearest_wp - total_offset
        else:
            offset_to_node = np.array([0.0, 0.0])

        measurement = np.array([[np.float32(offset_to_node[0])],
                                [np.float32(offset_to_node[1])],
                                [np.float32(angle)]])

        predicted = kalman.predict()
        corrected = kalman.correct(measurement)
        filtered_dx, filtered_dy, filtered_angle = corrected.flatten()

        prev_angle = filtered_angle

        save_offset(filtered_dx, filtered_dy, prev_angle)

        vis = frame_proc.copy()
        center = (proc_width // 2, proc_height // 2)
        offset_point = (
            int(center[0] - filtered_dx * 1.5),
            int(center[1] - filtered_dy * 1.5)
        )
        cv2.arrowedLine(vis, center, offset_point, (0, 255, 0), 2, tipLength=0.3)
        cv2.circle(vis, center, 5, (0, 0, 255), -1)
        cv2.putText(vis, f"Offset X: {filtered_dx:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f"Offset Y: {filtered_dy:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f"Angle: {prev_angle:.2f}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        fps = 1.0 / (current_time - prev_time) if current_time != prev_time else 0
        prev_time = current_time
        cv2.putText(vis, f"FPS: {fps:.1f}", (10, proc_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Anchor Optical Flow Tracker", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            pts = cv2.goodFeaturesToTrack(gray, maxCorners=300, qualityLevel=0.015, minDistance=7)
            if pts is not None and len(pts) > 0:
                first_pts = pts
                first_gray = gray.copy()
                prev_angle = 0.0
                tracking_initialized = True
                kalman = init_kalman()
                anchor_offset = None
                total_offset = np.array([0.0, 0.0])
        elif key == ord('t'):
            t_tracking_flag()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
