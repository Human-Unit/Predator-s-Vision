# 🌌 Yautja-Vision Elite Bio-Mask HUD

An **elite, military-grade alien combat visor** built with Python and OpenCV. This system features a high-performance multiprocessing architecture, gesture-based controls, real-time target tracking, and immersive cinematic HUD overlays.

---

## 🚀 Key Features

### 🛠️ High-Performance Architecture
*   **Multiprocessing Pipeline:** Heavy vision processing and HUD rendering are offloaded to a separate worker process to bypass the Python GIL, ensuring a consistent 30+ FPS.
*   **Hybrid Target Tracking:** Combines Haar Cascades for target discovery with a high-speed **MOSSE Tracker** for fluid, low-latency reticle acquisition.
*   **GPU Acceleration:** Utilizes OpenCV's Transparent API (`cv2.UMat`) to leverage **OpenCL** hardware acceleration for compute-heavy filters.

### 🎭 Visual Modes & Modules
*   **Thermal Vision:** High-contrast heat signature mapping using the JET color space.
*   **Night Vision:** Yautja-style light amplification using CLAHE with a specialized blue/gray cinematic tint.
*   **Tactical Zoom:** Sharp Lanczos4 center-crop magnification (up to 5x).
*   **Optical Cloaking:** Shimmering Gaussian distortion with edge refraction logic.
*   **Motion Tracker Radar:** Cinematic corner radar that "pings" and visualizes real-time pixel movement.
*   **AUTO-TARGET:** Automated head and eye tracking that locks the reticle onto detected prey.

---

## ✋ Gesture Control Mapping

The system uses **MediaPipe** to allow instant mode switching via hand gestures. No keyboard required during combat.

| Gesture | Visor Action | Effect |
| :--- | :--- | :--- |
| **Open Palm** | 🟢 **CLOAK** | Engage optical cloaking distortion. |
| **Peace Sign** | 🔴 **THERMAL** | Switch to high-contrast thermal vision. |
| **Pointing** | 🎯 **AIM HUD** | Activate the static aiming reticle. |
| **Rock On** | 🤖 **AUTO-TARGET** | Enable automated target tracking. |
| **Thumbs Up** | 🔵 **NIGHT VISION**| Activate Yautja light amplification. |
| **OK Sign** | 🔍 **ZOOM** | Toggle 2x Tactical Zoom. |
| **Fist** | ⚪ **RESET** | Return to Normal Diagnostic Vision. |

---

## 📂 Installation & Setup

### 1. Requirements
*   Python 3.10 or higher.
*   A webcam (for **Active Mode**).
*   An OpenAI-compatible LLM server (optional, for natural language commands).

### 2. Quick Install
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory (refer to `.env.example`). This allows you to configure your model and API endpoint:
```dotenv
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=http://127.0.0.1:1234/v1
OPENAI_MODEL=qwen/qwen2.5-vl-7b
```
*Note: If using LM Studio or Ollama, any string can be used as the API key.*

### 4. Custom Audio
The system dynamically loads audio assets. Drop your own `.wav` or `.mp3` files into the `sounds/` directory to customize the experience:
*   `MODE_CHANGE.wav` -> Visor transition sound.
*   `AIM_MODE.wav`   -> Target acquisition mode sound.
*   `TARGET_LOCK.wav`-> Played when a face is successfully locked.

---

## 🕹️ How to Use

Run the application from your terminal:
```bash
python main.py
```

### Select Your Mode:
1.  **Active Mode [1]:** Real-time webcam feed with full tracking and gesture support.
2.  **Passive Mode [2]:** Visualise effects on a static image file (default: `photo.jpg`).

### Controls (OpenCV Window Focused):
*   **Gesture:** Perform any of the mapped hand gestures in front of the camera.
*   **'C' Key:** Pause the feed to enter a natural language command in the terminal (e.g., *"Zoom in 3x"* or *"Activate infrared"*).
*   **'Record' Command:** Type `record` in the terminal to toggle saving the HUD feed to `combat_log.mp4`.
*   **'Q' Key:** Safely shut down the telemetry feed and exit.

---

## 📂 Project Structure
*   `main.py`: Entry point and multiprocessing orchestration.
*   `core/vision_engine.py`: Computer vision filter pipelines.
*   `core/detection.py`: Hybrid tracking and hand gesture recognition.
*   `core/audio_engine.py`: Pygame-based sound management.
*   `ui/hud_overlays.py`: Cinematic HUD layers, radar, and lens post-processing.
*   `config/settings.py`: Central configuration and color constants.

---
*Developed for the ultimate hunter.*
