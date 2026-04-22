import cv2
import os
from ultralytics import YOLO

class PoseEngine:
    """Base class for pose engines."""
    def process_frame(self, frame):
        raise NotImplementedError("Subclasses must implement process_frame")

class WindowsPoseEngine(PoseEngine):
    """Pose engine using YOLOv8-pose for Windows fallback."""
    def __init__(self):
        # Load a lightweight YOLOv8-pose model
        self.model = YOLO('yolov8n-pose.pt')

    def process_frame(self, frame):
        """Processes a frame and returns a list of detections."""
        results = self.model(frame, verbose=False, conf=0.5)
        
        detections = []
        for result in results:
            if result.keypoints is not None:
                for person_kps_tensor in result.keypoints.data:
                    person_kps = []
                    # person_kps_tensor: [17, 3] (x, y, conf)
                    for idx, kp in enumerate(person_kps_tensor):
                        person_kps.append({
                            'id': idx,
                            'x': float(kp[0]),
                            'y': float(kp[1]),
                            'confidence': float(kp[2])
                        })
                    detections.append(person_kps)
            
        return detections

class JetsonPoseEngine(PoseEngine):
    """Realistic pose engine using dustynv/jetson-inference for Jetson Orin Nano."""
    def __init__(self):
        try:
            import jetson_inference
            import jetson_utils
            # Load the ResNet18 body pose model as requested
            self.net = jetson_inference.poseNet("resnet18-body", threshold=0.15)
            self.jetson_utils = jetson_utils
            print("Jetson PoseNet loaded successfully.")
        except ImportError:
            print("jetson_inference not found. This engine is only for Jetson hardware.")
            self.net = None
            self.jetson_utils = None

    def process_frame(self, frame):
        if not self.net or not self.jetson_utils:
            return []
            
        # 1. Convert OpenCV BGR numpy array to RGBA cudaImage
        # jetson-inference expects RGBA format for processing
        rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        cuda_img = self.jetson_utils.cudaFromNumpy(rgba_frame)
        
        # 2. Process image with poseNet
        # We don't need the built-in overlay since we use the HTML5 Canvas UI
        poses = self.net.Process(cuda_img, overlay="none")
        
        # 3. Extract keypoints into our standardized dictionary format
        detections = []
        for pose in poses:
            person_kps = []
            # Fill with 17 empty slots to match COCO format expectation if missing
            for i in range(18):
                idx = pose.FindKeypoint(i)
                if idx < 0:
                    person_kps.append(None)
                    continue
                
                kp = pose.Keypoints[idx]
                person_kps.append({
                    'id': i,
                    'x': float(kp.x),
                    'y': float(kp.y),
                    'confidence': 1.0 # jetson-inference filtered by threshold already
                })
            detections.append(person_kps)
            
        return detections
