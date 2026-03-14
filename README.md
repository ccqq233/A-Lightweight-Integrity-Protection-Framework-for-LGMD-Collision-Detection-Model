# A-Lightweight-Integrity-Protection-Framework-for-LGMD-Collision-Detection-Model
 Addressed security vulnerabilities of the LGMD collision detection model in embedded systems against physical-layer data tampering attacks.

## Key Features

* **Dual-End Authentication**: Implements keyed BLAKE3 hashing to verify the integrity of both input image frames and output decision signals.
* **Anti-Replay Protection**: Incorporates a timestamp verification to ensure data freshness and defend against packet injection.
* **Resilient Design**: Includes a fallback mechanism that maintains detection continuity during single-frame verification failures.
* **Security Circuit-Breaker**: A sliding-window monitor that automatically halts the system if it detects 2 consecutive failures or 3 failures within 5 frames.
* **High Performance**: Leverages the speed of BLAKE3 to maintain near-real-time processing speeds suitable for embedded applications.

## System Architecture

The framework compares two implementations:
1.  **`clean_LGMD1.py`**: The baseline 2006 IEEE TNN LGMD model.
2.  **`blake_LGMD.py`**: The protected model with integrated cryptographic verification and fail-safe logic.

### Security Workflow
1.  **Input Verification**: Each frame is hashed with a secret key and timestamp before processing.
2.  **LGMD Computation**: The model calculates membrane potential ($\kappa_f$) and spikes.
3.  **Output Verification**: The resulting spike signal is authenticated before being sent to the actuator.
4.  **Halt Logic**: If verification fails repeatedly, the system triggers a "System Halted" state to prevent accidents.


## Getting Started

### Prerequisites
* Python 3.x
* OpenCV (`cv2`)
* NumPy
* Matplotlib
* [blake3](https://pypi.org/project/blake3/)

### Installation
```bash
pip install opencv-python numpy matplotlib blake3
```

## References
LGMD Model: Based on the 2006 IEEE Transactions on Neural Networks classic LGMD collision detection model.
