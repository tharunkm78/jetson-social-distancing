#!/usr/bin/env python3
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
# (Modified for Social Distancing & Animus UI Integration)
#

import sys
import os
import argparse
import logging
import threading
import time
import atexit
from flask import Flask, render_template, Response, jsonify, request
from engine.social_logic import SocialDistancingLogic
from engine.logger import DataLogger

import cv2

# Attempt to load jetson-inference natively
try:
    import jetson_inference
    import jetson_utils
    HAS_JETSON = True
except ImportError:
    from engine.pose_engine import WindowsPoseEngine
    HAS_JETSON = False

parser = argparse.ArgumentParser(description="Locate human poses and calculate social distancing violations.")
parser.add_argument("--network", type=str, default="resnet18-body")
parser.add_argument("--input", type=str, default="/dev/video0")
parser.add_argument("--threshold", type=float, default=0.05)
parser.add_argument("--distance", type=int, default=150)

try:
    args = parser.parse_known_args()[0]
except:
    sys.exit(0)

app = Flask(__name__)
logic = SocialDistancingLogic(threshold_pixels=args.distance)
logger = DataLogger(log_dir="results", throttle_seconds=1.0)

# Global Shared State
state = {
    'frame_jpeg': None,
    'midpoints': [],
    'violations': [],
    'actions': [],
    'ids': [],
    'radii': [],
    'raw_detections': [],
    'width': 640,
    'height': 480,
    'device': "JETSON ORIN NANO"
}
state_lock = threading.Lock()

def run_processing():
    global state
    
    if HAS_JETSON:
        # poseNet automatically parses sys.argv for --threshold
        net = jetson_inference.poseNet(args.network, sys.argv)
        camera = jetson_utils.videoSource(args.input, argv=sys.argv)
    else:
        net = WindowsPoseEngine()
        # On Windows, fallback to webcam 0 if the default jetson string is passed
        input_src = 0 if args.input == "/dev/video0" else args.input
        camera = cv2.VideoCapture(input_src)

    while True:
        try:
            if HAS_JETSON:
                img = camera.Capture()
                if img is None:
                    if str(args.input) not in ["/dev/video0", "0"]:
                        print("End of video clip reached.")
                        break
                    continue
                
                # 2. Process with hardware acceleration
                poses = net.Process(img, overlay="none")
                
                detections = []
                for pose in poses:
                    person_kps = []
                    for i in range(18):
                        idx = pose.FindKeypoint(i)
                        if idx < 0:
                            person_kps.append(None)
                            continue
                        kp = pose.Keypoints[idx]
                        person_kps.append({'id': i, 'x': float(kp.x), 'y': float(kp.y), 'confidence': 1.0})
                    detections.append(person_kps)
                
                w, h = img.width, img.height
                
                # CPU Bottleneck Optimization: cudaToNumpy -> cvtColor -> imencode takes 50-100ms for 720p/1080p.
                # Downscaling drastically improves performance and reduces MJPEG stream lag.
                img_np = jetson_utils.cudaToNumpy(img)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                
                stream_w = 640
                stream_h = int(h * (stream_w / float(w)))
                img_stream = cv2.resize(img_bgr, (stream_w, stream_h), interpolation=cv2.INTER_NEAREST)
                
                _, buffer = cv2.imencode('.jpg', img_stream, [cv2.IMWRITE_JPEG_QUALITY, 65])
                
            else:
                success, frame = camera.read()
                if not success:
                    if isinstance(input_src, str) and str(input_src) != "0":
                        print("End of video clip reached.")
                        break
                    continue
                h, w = frame.shape[:2]
                detections = net.process_frame(frame)
                _, buffer = cv2.imencode('.jpg', frame)

            # Logic & Logging
            # Logic & Logging
            if HAS_JETSON:
                scale_x = stream_w / float(w)
                scale_y = stream_h / float(h)
                scaled_detections = []
                for person in detections:
                    scaled_person = []
                    for kp in person:
                        if kp:
                            scaled_kp = kp.copy()
                            scaled_kp['x'] *= scale_x
                            scaled_kp['y'] *= scale_y
                            scaled_person.append(scaled_kp)
                        else:
                            scaled_person.append(None)
                    scaled_detections.append(scaled_person)
                detections = scaled_detections
                w, h = stream_w, stream_h

            midpoints, violations, actions, person_ids, radii = logic.process_frame(detections, dims=(w, h))
            logger.log_data(violations, actions)

            # --- Results Video Writing ---
            disp_frame = img_stream.copy() if HAS_JETSON else frame.copy()
            
            skeleton_edges = [[5,6],[5,11],[6,12],[11,12],[5,7],[7,9],[6,8],[8,10],[11,13],[13,15],[12,14],[14,16]]
            for idx, person in enumerate(detections):
                is_violating = any(v[0] == idx or v[1] == idx for v in violations)
                color = (0, 0, 139) if is_violating else (89, 160, 197) # BGR for AC Gold
                
                # Draw skeleton
                for s, e in skeleton_edges:
                    p1 = person[s] if s < len(person) else None
                    p2 = person[e] if e < len(person) else None
                    if p1 and p2 and p1.get('confidence', 0) > 0.1 and p2.get('confidence', 0) > 0.1:
                        cv2.line(disp_frame, (int(p1['x']), int(p1['y'])), (int(p2['x']), int(p2['y'])), color, 2)
                
                # Draw keypoints
                for kp in person:
                    if kp and kp.get('confidence', 0) > 0.1:
                        cv2.circle(disp_frame, (int(kp['x']), int(kp['y'])), 3, (77, 77, 255) if is_violating else (0, 215, 255), -1)
                        
            # Draw radii
            for idx, mid in enumerate(midpoints):
                r = int(radii[idx]) if radii else int(logic.threshold / 2)
                is_violating = any(v[0] == idx or v[1] == idx for v in violations)
                color = (0, 0, 139) if is_violating else (89, 160, 197)
                cv2.circle(disp_frame, (int(mid[0]), int(mid[1])), r, color, 1)
                
            # Draw violation lines
            for v in violations:
                p1 = midpoints[v[0]]
                p2 = midpoints[v[1]]
                cv2.line(disp_frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 139), 2)
                
            if not hasattr(run_processing, 'video_writer'):
                os.makedirs('results', exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                
                max_idx = 0
                for f in os.listdir('results'):
                    if f.startswith('output') and f.endswith('.mp4'):
                        num_str = f[6:-4]
                        if num_str.isdigit():
                            max_idx = max(max_idx, int(num_str))
                            
                filename = f'results/output{max_idx + 1}.mp4'
                run_processing.video_writer = cv2.VideoWriter(filename, fourcc, 30.0, (int(w), int(h)))
            
            run_processing.video_writer.write(disp_frame)
            # ---------------------------

            # Update State
            with state_lock:
                state['frame_jpeg'] = buffer.tobytes()
                state['midpoints'] = midpoints
                state['violations'] = violations
                state['actions'] = actions
                state['ids'] = person_ids
                state['radii'] = radii
                state['raw_detections'] = detections
                # CRITICAL: Match reported dimensions to the actual streamed JPEG resolution
                if HAS_JETSON:
                    state['width'] = stream_w
                    state['height'] = stream_h
                else:
                    state['width'] = w
                    state['height'] = h
        except Exception as e:
            print(f"Error in processing thread: {e}")
            time.sleep(0.1)

    print("Processing loop ended.")
    if hasattr(run_processing, 'video_writer') and run_processing.video_writer is not None:
        run_processing.video_writer.release()
        print("Video writer released.")
    
    os._exit(0)

# Start background thread
thread = threading.Thread(target=run_processing, daemon=True)
thread.start()

def generate_frames():
    while True:
        with state_lock:
            if state['frame_jpeg'] is None:
                time.sleep(0.1)
                continue
            frame = state['frame_jpeg']
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03) # Cap stream to ~30 FPS to prevent browser buffer flooding
        time.sleep(0.03) # Limit stream FPS to ~30

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def status():
    with state_lock:
        return jsonify({
            "midpoints": state['midpoints'],
            "violations": state['violations'],
            "actions": state['actions'],
            "ids": state['ids'],
            "num_people": len(state['midpoints']),
            "device": state['device'],
            "width": state['width'],
            "height": state['height'],
            "threshold": logic.threshold,
            "raw_detections": state['raw_detections']
        })

@app.route('/api/config', methods=['POST'])
def update_config():
    data = request.json
    if 'threshold' in data:
        logic.threshold = int(data['threshold'])
        return jsonify({"status": "success", "threshold": logic.threshold})
    return jsonify({"status": "error"}), 400

def cleanup():
    if hasattr(run_processing, 'video_writer') and run_processing.video_writer is not None:
        try:
            run_processing.video_writer.release()
            print("Cleanup: Video writer released gracefully.")
        except Exception as e:
            pass

atexit.register(cleanup)

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    finally:
        cleanup()
