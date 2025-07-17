import cv2
import numpy as np
import time
import json
import os
import logging
import math

logging.basicConfig(filename='anchor_tracking.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')

FLAG_PATH = 'tracking_enabled.flag'

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

def save_offset(offset_dx, offset_dy, angle):
    data = {
        'dx': float(offset_dx),
        'dy': float(offset_dy),
        'angle': float(angle)
    }
    tmp_filename = 'offsets_tmp.json'
    final_filename = 'offsets.json'
    try:
        with open(tmp_filename, 'w') as f:
            json.dump(data, f)
        os.replace(tmp_filename, final_filename)
    except Exception as e:
        logging.error(f"Error saving offsets: {e}")

def add_waypoint(waypoints, points, angle):
    waypoints.append({'points': points.copy(), 'angle': angle})

def points_diff(points1, points2, threshold=3.0):
    if points1 is None or points2 is None:
        return True
    if points1.shape != points2.shape:
        return True
    diff = np.linalg.norm(points1 - points2, axis=1)
    mean_diff = np.mean(diff)
    return mean_diff > threshold

def angle_diff(a1, a2):
    diff = abs(a1 - a2) % 360
    if diff > 180:
        diff = 360 - diff
    return diff

def smooth_points(points1, points2, alpha=0.3):
    return alpha * points1 + (1 - alpha) * points2

def find_returned_node(waypoints, current_points, threshold=2.0):
    for i, node in enumerate(waypoints):
        if not points_diff(current_points, node['points'], threshold=threshold):
            return i
    return None

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру")
        return

    proc_width, proc_height = 720, 576
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    first_gray = None
    first_pts = None
    tracking_initialized = False

    anchor_points = None
    anchor_angle = None
    waypoints = []
    last_waypoint_time = 0
    waypoint_add_interval = 1.0  # секунды

    accumulated_dx = 0.0
    accumulated_dy = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tracking_enabled = is_tracking_enabled()
        frame_proc = cv2.resize(frame, (proc_width, proc_height))
        gray = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2GRAY)

        if not tracking_enabled:
            first_gray = None
            first_pts = None
            tracking_initialized = False
            anchor_points = None
            anchor_angle = None
            waypoints.clear()
            last_waypoint_time = 0
            accumulated_dx = 0
            accumulated_dy = 0

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
            pts = cv2.goodFeaturesToTrack(gray, maxCorners=2000, qualityLevel=0.1, minDistance=10)
            if pts is not None and len(pts) > 0:
                first_gray = gray.copy()
                first_pts = pts
                tracking_initialized = True
                anchor_points = None
                anchor_angle = None
                waypoints.clear()
                last_waypoint_time = time.time()
                accumulated_dx = 0
                accumulated_dy = 0
            continue

        next_pts, status, error = cv2.calcOpticalFlowPyrLK(first_gray, gray, first_pts, None, **lk_params)

        if next_pts is None or status is None:
            tracking_initialized = False
            anchor_points = None
            anchor_angle = None
            waypoints.clear()
            last_waypoint_time = 0
            accumulated_dx = 0
            accumulated_dy = 0
            continue

        status = status.flatten()
        good_new = next_pts[status == 1].reshape(-1, 2)
        good_old = first_pts[status == 1].reshape(-1, 2)

        if len(good_new) < 10:
            # слишком мало точек для надёжного трекинга
            tracking_initialized = False
            pass

        H, inliers = cv2.estimateAffinePartial2D(good_old.reshape(-1, 1, 2), good_new.reshape(-1, 1, 2),
                                                 method=cv2.RANSAC, ransacReprojThreshold=3, maxIters=2000)

        angle = 0.0
        if H is not None and inliers is not None and np.count_nonzero(inliers) > 5:
            raw_angle = math.degrees(math.atan2(H[1, 0], H[0, 0]))
            angle = raw_angle

        # Рассчёт среднего смещения
        dxs = good_new[:, 0] - good_old[:, 0]
        dys = good_new[:, 1] - good_old[:, 1]
        avg_dx = np.mean(dxs)
        avg_dy = np.mean(dys)

        current_time = time.time()

        if anchor_points is None:
            anchor_points = good_new.copy()
            anchor_angle = angle
            add_waypoint(waypoints, anchor_points, anchor_angle)
            last_waypoint_time = current_time
            accumulated_dx = 0
            accumulated_dy = 0
            logging.info(f"Инициализирован якорь с углом {anchor_angle:.2f}")

        else:
            # Добавляем новый узел, если прошло достаточно времени и точки сильно отличаются
            if (current_time - last_waypoint_time) > waypoint_add_interval:
                if points_diff(good_new, anchor_points, threshold=3.0) and angle_diff(angle, anchor_angle) > 5.0:
                    add_waypoint(waypoints, good_new, angle)
                    last_waypoint_time = current_time
                    anchor_points = smooth_points(anchor_points, good_new, alpha=0.3)
                    anchor_angle = (anchor_angle * 0.7 + angle * 0.3) % 360
                    logging.info(f"Добавлен новый узел с углом {angle:.2f}")

            # Проверяем возврат к уже пройденному узлу
            idx = find_returned_node(waypoints, good_new, threshold=2.0)
            if idx is not None and idx < len(waypoints) - 1:
                # Удаляем узлы, которые идут после найденного — "возврат назад"
                removed = waypoints[idx+1:]
                waypoints = waypoints[:idx+1]
                logging.info(f"Возврат к узлу {idx}, удалено {len(removed)} узлов")

            # Аккумулируем смещения
            accumulated_dx += avg_dx
            accumulated_dy += avg_dy

        # Сохраняем текущие накопленные смещения и отклонение по углу относительно anchor_angle
        offset_dx = accumulated_dx
        offset_dy = accumulated_dy
        offset_angle = (angle - anchor_angle + 180) % 360 - 180  # нормализуем в [-180,180]

        save_offset(offset_dx, offset_dy, offset_angle)

        # Визуализация
        vis = frame_proc.copy()

        for wp in waypoints:
            for p in wp['points']:
                cv2.circle(vis, (int(p[0]), int(p[1])), 3, (255, 0, 0), -1)

        if anchor_points is not None:
            for p in anchor_points:
                cv2.circle(vis, (int(p[0]), int(p[1])), 4, (37, 225, 88), -1)

        cv2.putText(vis, f"Tracking Enabled", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, f"Waypoints: {len(waypoints)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, f"Accum dx: {offset_dx:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, f"Accum dy: {offset_dy:.2f}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, f"Offset angle: {offset_angle:.2f}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            t_tracking_flag()

        cv2.imshow("Anchor Optical Flow Tracker", vis)

        # Обновляем точки для следующего кадра
        first_gray = gray.copy()
        first_pts = good_new.reshape(-1, 1, 2)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
