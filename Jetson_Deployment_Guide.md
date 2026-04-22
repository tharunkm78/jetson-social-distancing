# JETSON ORIN NANO DEPLOYMENT GUIDE
## Project: Animus Social Distancing & Pose Estimation

This guide provides the exact steps to deploy and run the Animus system on an NVIDIA Jetson Orin Nano (15W) using the `dustynv/jetson-inference` Docker environment.

---

### 1. Launch the Docker Container
Run the following command from your **host terminal** to start the container with camera access and your project folder mounted.

```bash
# Replace <PATH_TO_REPO> with the actual path to your Keypoints_AI folder
# Example: /home/jetson/Keypoints_AI

docker run --runtime nvidia -it --rm \
    --network host \
    --volume <PATH_TO_REPO>:/Keypoints_AI \
    --device /dev/video0 \
    dustynv/jetson-inference:r35.2.1
```

---

### 2. Enter the Project Directory
Once inside the docker container's bash prompt, navigate to your project:

```bash
cd /Keypoints_AI
```

---

### 3. Install Runtime Dependencies
The base Jetson-Inference image may not contain Flask. Install it along with other requirements:

```bash
pip3 install flask
```

---

### 4. Execute the Animus System
Run the unified script. This script will automatically detect the Jetson hardware and enable TensorRT hardware acceleration.

```bash
python3 posenet_socialdistancing.py --network=resnet18-body --input=/dev/video0
```

**Common Flags:**
- `--network`: Specifies the pose model (e.g., `resnet18-body`, `resnet18-hand`).
- `--input`: The camera URI (typically `/dev/video0`).
- `--distance`: The initial social distancing threshold in pixels (default: 150).

---

### 5. Access the Animus Dashboard
The system is now acting as a web server. Open a web browser on your **PC or Tablet** that is on the same network as the Jetson.

**URL:** `http://<JETSON_IP_ADDRESS>:5000`

---

### 6. Managing Logs
- The system will start a fresh log session every time you run it.
- **Violation Logs**: `/Keypoints_AI/logs/violations_log.csv`
- **Action Logs**: `/Keypoints_AI/logs/action_log.csv`

---

### Troubleshooting
- **No Camera Found**: Ensure `--device /dev/video0` was included in the `docker run` command.
- **Permission Denied**: If you cannot write logs, run `chmod -R 777 /Keypoints_AI` from the host.
- **Model Download**: On the first run, the Jetson will automatically download the `resnet18-body` model if it's not present. Ensure the Jetson has an active internet connection.

---

**Status: READY FOR DEPLOYMENT**
