# Adaptive Motion Navigation in 3D Environments 🎮

> **Diploma Thesis**  
> **Student:** Tzortzakis Alkiviadis
> **Supervisor:** Sintoris Christos  
> **University:** University of Patras

## Abstract
This thesis investigates the feasibility and performance of using motion-based controllers (Nintendo Joy-Con/Pro Controller) for precise navigation in desktop 3D environments. By implementing advanced signal processing (**1€ Filter**) and dynamic adaptive mapping (**"Sniper Mode"**), we aim to bridge the gap between VR locomotion and traditional mouse/keyboard inputs.

## Key Features
- **Low-Latency Driver:** Custom Python driver for Joy-Con/Pro Controller (via `hidapi`) reading raw IMU data at 66Hz.
- **Advanced Filtering:** Real-time comparison between **Kalman Filter** and **OneEuro Filter** for jitter reduction vs. lag minimization.
- **Adaptive Precision:** Dynamic sensitivity adjustment based on velocity and user intent (Clutch/Trigger mechanism).
- **Unity Testbed:** A UDP-connected 3D environment for Fitts' Law evaluation tasks.

## Installation & Setup

### 1. Hardware Requirements
- Nintendo Joy-Con (L/R) or Switch Pro Controller.
- Bluetooth Adapter (Bluetooth 4.0+).

### 2. Python Environment
```bash
# Clone the repository
git clone https://github.com/[your-username]/adaptive-motion-thesis.git

# Install dependencies
pip install -r requirements.txt