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
    kf = cv2.KalmanFilter(3, 3)
    kf.transitionMatrix = np.eye(3, dtype=np.float32)
    kf.measurementMatrix = np.eye(3, dtype=np.float32)
    kf.processNoiseCov = np.eye(3, dtype=np.float32) * 1e-4
    kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-2
    kf.statePre = np.zeros((3, 1), dtype=np.float32)
    kf.statePost = np.zeros((3, 1), dtype=np.float32)
    return kf

def add_waypoint(waypoints, offset_pos, angle):
    # Добавляем waypoint с позицией (2D) и углом
    waypoints.append({'pos': np.array(offset_pos), 'angle': angle})

def draw_opposite_arrow(img, start_point, direction_vector, length=50, color=(0, 255, 0), thickness=2):
    norm = np.linalg.norm(direction_vector)
    if norm == 0:
        return img
    dir_norm = -direction_vector / norm  # обратное направление
    end_point = (int(start_point[0] + dir_norm[0] * length), int(start_point[1] + dir_norm[1] * length))
    cv2.arrowedLine(img, (int(start_point[0]), int(start_point[1])), end_point, color, thickness, tipLength=0.3)
    return img

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
    kalman = init_kalman()

    anchor_offset = None
    total_offset = np.array([0.0, 0.0, 0.0])
    current_waypoint_index = 0
    waypoints = []
    last_waypoint_time = 0

    last_angle = None
    angle_threshold = 15  # градусов — порог резкого поворота камеры

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tracking_enabled = is_tracking_enabled()
        frame_proc = cv2.resize(frame, (proc_width, proc_height))
        gray = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2GRAY)

        if not tracking_enabled:
            # Сбрасываем всё при отключенном трекинге
            first_gray = None
            first_pts = None
            tracking_initialized = False
            kalman = init_kalman()
            anchor_offset = None
            total_offset = np.array([0.0, 0.0, 0.0])
            waypoints.clear()
            current_waypoint_index = 0
            last_angle = None

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
                kalman = init_kalman()
                anchor_offset = None
                total_offset = np.array([0.0, 0.0, 0.0])
                waypoints.clear()
                current_waypoint_index = 0
                last_waypoint_time = time.time()
                last_angle = None
            continue

        next_pts, status, error = cv2.calcOpticalFlowPyrLK(first_gray, gray, first_pts, None, **lk_params)

        if next_pts is None or status is None:
            tracking_initialized = False
            anchor_offset = None
            total_offset = np.array([0.0, 0.0, 0.0])
            waypoints.clear()
            current_waypoint_index = 0
            last_angle = None
            continue

        status = status.flatten()
        good_new = next_pts[status == 1].reshape(-1, 2)
        good_old = first_pts[status == 1].reshape(-1, 2)

        if len(good_new) < 10:
            continue

        H, inliers = cv2.estimateAffinePartial2D(good_old.reshape(-1, 1, 2), good_new.reshape(-1, 1, 2),
                                                 method=cv2.RANSAC, ransacReprojThreshold=3, maxIters=2000)

        angle = 0.0
        if H is not None and inliers is not None and np.count_nonzero(inliers) > 5:
            raw_angle = math.degrees(math.atan2(H[1, 0], H[0, 0]))
            angle = raw_angle

        dxs = good_new[:, 0] - good_old[:, 0]
        dys = good_new[:, 1] - good_old[:, 1]
        avg_dx = np.mean(dxs)
        avg_dy = np.mean(dys)

        # Фильтрация Калмана (упрощённая)
        measured = np.array([[avg_dx], [avg_dy], [angle]], dtype=np.float32)
        kalman.correct(measured)
        predicted = kalman.predict()
        filtered_dx, filtered_dy, filtered_angle = predicted.flatten()

        if anchor_offset is None:
            # Инициализация якоря
            anchor_offset = np.array([filtered_dx, filtered_dy, filtered_angle])
            add_waypoint(waypoints, [0, 0], filtered_angle)  # Первый узел в начале
            last_waypoint_time = time.time()
            last_angle = filtered_angle

        else:
            # Проверяем резкий поворот камеры
            if last_angle is not None:
                angle_diff = abs(filtered_angle - last_angle)
                if angle_diff > angle_threshold:
                    # Резкий поворот — добавляем waypoint, anchor не сбрасываем
                    offset_since_anchor = np.array([filtered_dx, filtered_dy, filtered_angle]) - anchor_offset
                    add_waypoint(waypoints, offset_since_anchor[:2], filtered_angle)
                    logging.info(f"Добавлен waypoint из-за резкого поворота: угол сдвинулся на {angle_diff:.2f} град")
                    last_waypoint_time = time.time()

            last_angle = filtered_angle

        # Добавляем узлы примерно раз в 5 секунд (по времени)
        current_time = time.time()
        if current_time - last_waypoint_time >= 5.0:
            offset_since_anchor = np.array([filtered_dx, filtered_dy, filtered_angle]) - anchor_offset
            add_waypoint(waypoints, offset_since_anchor[:2], filtered_angle)
            last_waypoint_time = current_time

        # Проходим по узлам (лесенке) плавно к якарю
        if len(waypoints) > 0:
            current_wp = waypoints[current_waypoint_index]['pos']
            offset_vec = np.array([filtered_dx, filtered_dy])
            dist = np.linalg.norm(offset_vec - current_wp)
            if dist < 15:
                current_waypoint_index += 1
                if current_waypoint_index >= len(waypoints):
                    # Дошли до конца — очищаем waypoints и сбрасываем индекс
                    waypoints.clear()
                    current_waypoint_index = 0
                    anchor_offset = np.array([filtered_dx, filtered_dy, filtered_angle])
                    logging.info("Возврат к якарю завершён, waypoints очищены")

        # Сохраняем общий оффсет для внешнего использования
        total_offset = np.array([filtered_dx, filtered_dy, filtered_angle])

        # Визуализация
        vis = frame_proc.copy()
        for wp in waypoints:
            cv2.circle(vis, (int(wp['pos'][0] + proc_width//2), int(wp['pos'][1] + proc_height//2)), 5, (255, 0, 0), -1)

        cv2.putText(vis, f"Tracking Enabled", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, f"Waypoint: {current_waypoint_index}/{len(waypoints)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, f"Filtered dx: {total_offset[0]:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, f"Filtered dy: {total_offset[1]:.2f}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, f"Filtered angle: {total_offset[2]:.2f}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        center_point = (proc_width // 2, proc_height // 2)
        direction = np.array([filtered_dx, filtered_dy])
        vis = draw_opposite_arrow(vis, center_point, direction, length=50, color=(0, 255, 0), thickness=2)

        cv2.putText(vis, f"Waypoints: {len(waypoints)} Current WP idx: {current_waypoint_index}", (10, proc_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Anchor Optical Flow Tracker", vis)

        save_offset(total_offset[0], total_offset[1], total_offset[2])

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            t_tracking_flag()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
