import math
import time

class SocialDistancingLogic:
    def __init__(self, threshold_pixels=150, history_size=3):
        self.threshold = threshold_pixels
        self.history_size = history_size
        self.prev_persons = [] # List of {id, midpoint}
        self.violation_counts = {} # (id1, id2) -> frames_active
        self.next_id = 0
        self.frame_dims = (640, 480)

    def get_body_scale(self, kps):
        """Calculates scale based on hip-to-shoulder distance."""
        def get_kp(kps_list, kp_id):
            return next((kp for kp in kps_list if kp and kp['id'] == kp_id), None)
            
        l_shoulder = get_kp(kps, 5)
        r_shoulder = get_kp(kps, 6)
        l_hip = get_kp(kps, 11)
        r_hip = get_kp(kps, 12)
        
        if all([l_shoulder, r_shoulder, l_hip, r_hip]):
            # Average shoulder to average hip
            sh_mid = ((l_shoulder['x'] + r_shoulder['x'])/2, (l_shoulder['y'] + r_shoulder['y'])/2)
            hip_mid = ((l_hip['x'] + r_hip['x'])/2, (l_hip['y'] + r_hip['y'])/2)
            return self.calculate_distance(sh_mid, hip_mid)
        return None

    def get_hip_midpoint(self, keypoints):
        left_hip = next((kp for kp in keypoints if kp and kp['id'] == 11), None)
        right_hip = next((kp for kp in keypoints if kp and kp['id'] == 12), None)

        if left_hip and right_hip:
            return ((left_hip['x'] + right_hip['x']) / 2, (left_hip['y'] + right_hip['y']) / 2)
        return None

    def calculate_distance(self, p1, p2):
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

    def process_frame(self, detections, dims=(640, 480)):
        self.frame_dims = dims
        current_persons = []
        
        # 1. Tracking: Match detections to previous frame (Nearest Neighbor)
        available_prev = self.prev_persons.copy()
        
        for person_kps in detections:
            midpoint = self.get_hip_midpoint(person_kps)
            if not midpoint: continue
            
            best_match = None
            min_dist = 150 # Max pixels for a match
            
            for prev in available_prev:
                d = self.calculate_distance(midpoint, prev['midpoint'])
                if d < min_dist:
                    min_dist = d
                    best_match = prev
            
            if best_match is None:
                best_match_id = self.next_id
                self.next_id += 1
            else:
                best_match_id = best_match['id']
                available_prev.remove(best_match)
                
            scale = self.get_body_scale(person_kps) or 100
            safe_radius = (self.threshold / 100.0) * scale * 1.2
                
            current_persons.append({
                'id': best_match_id,
                'midpoint': midpoint,
                'keypoints': person_kps,
                'scale': scale,
                'safe_radius': safe_radius
            })

        # 2. Distance Logic with Body-Scale Normalization
        active_violations = []
        num_current = len(current_persons)
        for i in range(num_current):
            for j in range(i + 1, num_current):
                p1, p2 = current_persons[i], current_persons[j]
                dist = self.calculate_distance(p1['midpoint'], p2['midpoint'])
                
                # Violation occurs if the distance is less than the sum of their perspective-aware radii
                if dist < (p1['safe_radius'] + p2['safe_radius']):
                    v_key = tuple(sorted((p1['id'], p2['id'])))
                    self.violation_counts[v_key] = self.violation_counts.get(v_key, 0) + 1
                    
                    # 3. Temporal Smoothing: Only report if persistent
                    if self.violation_counts[v_key] >= self.history_size:
                        # Find indices in current frame for frontend
                        active_violations.append((i, j, dist))
                else:
                    v_key = tuple(sorted((p1['id'], p2['id'])))
                    self.violation_counts[v_key] = max(0, self.violation_counts.get(v_key, 0) - 1)

        # Cleanup old violation counts
        current_ids = {p['id'] for p in current_persons}
        self.violation_counts = {k: v for k, v in self.violation_counts.items() 
                               if k[0] in current_ids and k[1] in current_ids and v > 0}

        # 4. Action Detection (Backend isolated)
        actions = [self.detect_action(p['keypoints']) for p in current_persons]
        
        self.prev_persons = current_persons
        
        # Prepare data for return
        midpoints = [p['midpoint'] for p in current_persons]
        ids = [p['id'] for p in current_persons]
        radii = [p['safe_radius'] for p in current_persons]
        
        return midpoints, active_violations, actions, ids, radii

    def detect_action(self, kps):
        if not kps or len(kps) < 17: return "unknown"
        def get_kp(kps_list, kp_id):
            return next((kp for kp in kps_list if kp and kp['id'] == kp_id), None)
            
        head = get_kp(kps, 0)
        l_shoulder = get_kp(kps, 5); r_shoulder = get_kp(kps, 6)
        l_wrist = get_kp(kps, 9); r_wrist = get_kp(kps, 10)
        l_hip = get_kp(kps, 11); r_hip = get_kp(kps, 12)
        l_knee = get_kp(kps, 13); r_knee = get_kp(kps, 14)
        l_ankle = get_kp(kps, 15); r_ankle = get_kp(kps, 16)

        upper_body = head or l_shoulder or r_shoulder
        if not upper_body or not l_hip: return "detecting..."

        hip_width = abs(r_hip['x'] - l_hip['x']) if r_hip else 1

        if l_wrist and r_wrist and l_shoulder and r_shoulder:
            if l_wrist['y'] < l_shoulder['y'] and r_wrist['y'] < r_shoulder['y']: return "arms raised"

        if l_ankle and r_ankle and upper_body:
            height = abs(r_ankle['y'] - upper_body['y'])
            if height < 100 and height < hip_width * 2: return "fallen"
            if abs(l_hip['y'] - l_ankle['y']) < height * 0.4: return "sitting"

        # Walking Detection (Uses ankles if available, otherwise knees)
        leg_spread = 0
        if l_ankle and r_ankle:
            leg_spread = abs(l_ankle['x'] - r_ankle['x'])
        elif l_knee and r_knee:
            leg_spread = abs(l_knee['x'] - r_knee['x']) * 1.2 # Boost knee spread to match ankle scale
            
        arm_swing = False
        if l_wrist and r_wrist:
            wrist_spread = abs(l_wrist['x'] - r_wrist['x'])
            if wrist_spread > hip_width * 2.0: arm_swing = True
            
        if leg_spread > hip_width * 1.5 or arm_swing: return "walking"
                
        return "standing"
