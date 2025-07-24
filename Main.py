import cv2
import numpy as np
import time
import json
import os
import math
from scipy.spatial import cKDTree
from skimage.registration import phase_cross_correlation
from skimage import img_as_float

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
    # Handle case where gray_frame is None
    frame_hash = hash(gray_frame.tobytes()) if gray_frame is not None else 0
    wp_list.append({
        'points': pts.copy(),
        'angle': ang,
        'frame_hash': frame_hash
    })

def adaptive_good_features(gray, min_features=100, max_features=1000, wind_factor=1.0):
    # Apply CLAHE for contrast enhancement in low-light conditions
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Calculate adaptive parameters based on image characteristics
    mean_val, std_val = cv2.meanStdDev(gray)
    quality_level = max(0.01, 0.1 * (1 - std_val / 50))
    
    height, width = gray.shape
    area = height * width
    
    # Increase feature count based on wind factor (stronger wind = more features needed)
    num_features = max(min_features, min(max_features, int((area / 500) * wind_factor)))
    min_distance = max(5, int(np.sqrt(area / num_features)))
    
    # Detect features with enhanced image
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
        # Убран учет угла из сравнения
        total_score = pos_score
        if total_score < best_score and pos_score < pos_thresh:
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

def rope_ladder_waypoint_management(waypoints, current_points, current_angle, distance_threshold=10.0):
    """
    Rope ladder-inspired waypoint management:
    - Add waypoints when moving away from anchor (climbing the rope)
    - Remove waypoints when returning closer to anchor (descending the rope)
    """
    if len(waypoints) == 0:
        return waypoints
    
    # Calculate distance from current position to anchor
    anchor_pts = waypoints[0]['points']
    current_to_anchor_dist = np.mean(np.linalg.norm(current_points - np.mean(anchor_pts, axis=0), axis=1))
    
    # Find the closest existing waypoint
    closest_dist = float('inf')
    closest_idx = -1
    for i, wp in enumerate(waypoints):
        wp_center = np.mean(wp['points'], axis=0)
        current_center = np.mean(current_points, axis=0)
        dist = np.linalg.norm(wp_center - current_center)
        if dist < closest_dist:
            closest_dist = dist
            closest_idx = i
    
    # If we're far from existing waypoints, add a new one (climbing)
    if closest_dist > distance_threshold:
        # Only add if we're moving away from anchor
        if len(waypoints) > 1:
            last_wp_center = np.mean(waypoints[-1]['points'], axis=0)
            current_center = np.mean(current_points, axis=0)
            anchor_center = np.mean(anchor_pts, axis=0)
            
            # Check if we're moving away from anchor
            last_dist_to_anchor = np.linalg.norm(last_wp_center - anchor_center)
            current_dist_to_anchor = np.linalg.norm(current_center - anchor_center)
            
            if current_dist_to_anchor > last_dist_to_anchor + 5.0:  # Moving away
                add_waypoint(waypoints, current_points, current_angle, None)
    
    # If we're close to an existing waypoint that's not the anchor, remove waypoints beyond it (descending)
    elif closest_idx > 0:  # Not the anchor
        # Remove waypoints after the closest one
        waypoints = waypoints[:closest_idx + 1]
    
    return waypoints

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    # Set frame dimensions for consistent processing
    lw, lh = 720, 576
    # LK optical flow parameters optimized for aerial navigation
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    # System state variables
    anchor_gray = None
    anchor_pts = None
    anchor_ang = 0.0
    anchor_frame_full = None
    waypoints = []
    last_wp_time = 0
    wp_interval = 1.0  # Waypoint creation interval in seconds
    tracking = False
    prev_gray = None
    prev_pts = None
    wind_magnitude_history = []
    anchor_returned = False  # Флаг для отслеживания возвращения к анкеру

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize and convert to grayscale
        frame_resized = cv2.resize(frame, (lw, lh))
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        # Check if tracking is disabled
        if not is_tracking_enabled():
            # Reset all tracking state
            anchor_gray = None
            anchor_pts = None
            anchor_ang = 0.0
            anchor_frame_full = None
            waypoints.clear()
            tracking = False
            prev_gray = None
            prev_pts = None
            wind_magnitude_history.clear()
            anchor_returned = False
            
            # Display status
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

        # Calculate wind magnitude using optical flow
        wind_magnitude = 0.0
        if prev_gray is not None and prev_pts is not None:
            wind_magnitude = calculate_optical_flow_magnitude(prev_gray, gray, prev_pts)
            wind_magnitude_history.append(wind_magnitude)
            if len(wind_magnitude_history) > 10:  # Keep last 10 measurements
                wind_magnitude_history.pop(0)
        
        # Calculate adaptive wind factor
        avg_wind_magnitude = np.mean(wind_magnitude_history) if wind_magnitude_history else 0.0
        wind_factor = 1.0 + min(avg_wind_magnitude / 10.0, 2.0)  # Cap at 3x features
        
        # Store current frame for next iteration
        prev_gray = gray.copy()
        
        # Initialize tracking if not already started
        if not tracking:
            pts = adaptive_good_features(gray, min_features=150, wind_factor=wind_factor)
            if pts is not None and len(pts) > 100:
                # Set anchor frame and points
                anchor_gray = gray.copy()
                anchor_pts = pts.reshape(-1, 2).copy()
                anchor_ang = 0.0
                anchor_frame_full = frame_resized.copy()
                waypoints.clear()
                add_waypoint(waypoints, anchor_pts, anchor_ang, anchor_gray)
                last_wp_time = time.time()
                tracking = True
                prev_pts = anchor_pts.copy()
                anchor_returned = False
            cv2.waitKey(1)
            continue

        # Calculate optical flow from anchor to current frame
        next_pts, st, err = cv2.calcOpticalFlowPyrLK(anchor_gray, gray, anchor_pts.reshape(-1, 1, 2), None, **lk_params)
        if next_pts is None or st is None:
            continue
            
        # Filter valid points
        st = st.flatten()
        good_new = next_pts[st == 1].reshape(-1, 2)
        good_old = anchor_pts[st == 1].reshape(-1, 2)
        
        # Update prev_pts for wind detection
        prev_pts = good_new.copy()
        
        # Check if we have enough points for reliable tracking
        if len(good_new) < 50:
            # If too few points, increase feature density due to potential wind
            pts = adaptive_good_features(gray, min_features=200, wind_factor=wind_factor*1.5)
            if pts is not None:
                prev_pts = pts.reshape(-1, 2).copy()
            continue

        # Estimate affine transformation to get rotation
        H, inliers = cv2.estimateAffinePartial2D(
            good_old.reshape(-1, 1, 2),
            good_new.reshape(-1, 1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0
        )
        
        # Calculate current angle from transformation matrix
        current_ang = math.degrees(math.atan2(H[1, 0], H[0, 0])) if H is not None else anchor_ang

        now = time.time()

        # Apply rope ladder waypoint management
        waypoints = rope_ladder_waypoint_management(waypoints, good_new, current_ang, distance_threshold=15.0)

        # Add new waypoint if points have moved significantly and enough time has passed
        if points_moved(good_new, waypoints[-1]['points'], 3.0) and now - last_wp_time > wp_interval:
            add_waypoint(waypoints, good_new, current_ang, gray)
            last_wp_time = now

        # Check for similar waypoints to detect loops
        similar_idx = find_similar(good_new, waypoints, pos_thresh=5.0)
        if similar_idx != -1 and similar_idx < len(waypoints) - 1:
            # Trim waypoints after the similar one (loop closure)
            waypoints = waypoints[:similar_idx + 1]
            last_wp_time = now

        # Check if we've returned to the anchor point
        anchor_returned = False
        if is_returned_to_anchor(good_new, anchor_pts, current_ang, anchor_ang, gray, anchor_gray):
            # Reset to only anchor waypoint
            waypoints = [waypoints[0]]
            last_wp_time = now
            anchor_returned = True

        # Calculate total offset from anchor
        total_dx, total_dy = mean_offset(good_new, anchor_pts)
        
        # Сохраняем угол только при возвращении к анкеру, иначе 0
        if anchor_returned:
            total_ang = angle_diff(current_ang, anchor_ang)
        else:
            total_ang = 0.0
            
        save_offset(total_dx, total_dy, total_ang)

        # Visualization
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Draw waypoints (rope ladder visualization)
        for i, wp in enumerate(waypoints):
            color = (255, 0, 0) if i > 0 else (0, 255, 0)  # Green for anchor, blue for others
            for p in wp['points']:
                cv2.circle(vis, (int(p[0]), int(p[1])), 3, color, -1)
                
        # Draw connections between waypoints (rope ladder rungs)
        for i in range(len(waypoints) - 1):
            if len(waypoints[i]['points']) > 0 and len(waypoints[i+1]['points']) > 0:
                pt1 = tuple(np.mean(waypoints[i]['points'], axis=0).astype(int))
                pt2 = tuple(np.mean(waypoints[i+1]['points'], axis=0).astype(int))
                cv2.line(vis, pt1, pt2, (255, 255, 0), 2)
                
        # Draw current tracked points
        for p in good_new:
            cv2.circle(vis, (int(p[0]), int(p[1])), 2, (0, 255, 255), -1)

        # Display information
        cv2.putText(vis, f"Waypoints: {len(waypoints)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, f"dx:{total_dx:.2f} dy:{total_dy:.2f} ang:{total_ang:.2f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(vis, f"Wind Factor: {wind_factor:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Indicate if anchor is locked
        if len(waypoints) == 1:
            cv2.putText(vis, "ANCHOR LOCKED", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Tracker", vis)

        # Handle keyboard input
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if k == ord('t'):
            t_tracking_flag()

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

def calculate_optical_flow_magnitude(prev_gray, curr_gray, prev_pts):
    """Calculate median optical flow magnitude to detect wind conditions"""
    if prev_pts is None or len(prev_pts) < 10:
        return 0.0
    
    # Calculate optical flow
    next_pts, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    if next_pts is None or st is None:
        return 0.0
    
    # Filter valid points
    st = st.flatten()
    good_new = next_pts[st == 1].reshape(-1, 2)
    good_old = prev_pts[st == 1].reshape(-1, 2)
    
    if len(good_new) < 5:
        return 0.0
    
    # Calculate flow magnitudes
    flow_vectors = good_new - good_old
    magnitudes = np.linalg.norm(flow_vectors, axis=1)
    
    # Return median magnitude as wind indicator
    return np.median(magnitudes)

if __name__ == "__main__":
    main()