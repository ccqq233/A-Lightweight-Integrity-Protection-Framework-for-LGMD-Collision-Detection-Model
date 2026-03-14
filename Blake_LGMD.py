import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import struct
import secrets
import blake3  # pip install blake3
import os
from collections import deque

# Set plotting parameters
plt.rcParams['axes.unicode_minus'] = False 

# ===================== Security Configuration =====================
SHARED_KEY = b'\x8f\x12\x9a\x78\xe6\x54\x32\x10\x98\x76\x54\x32\x10\xab\xcd\xef\x8f\x12\x9a\x78\xe6\x54\x32\x10\x98\x76\x54\x32\x10\xab\xcd\xef' # 32 bytes
TIME_THRESHOLD_MS = 10
DOMAIN_INPUT = b'\x00'
DOMAIN_OUTPUT = b'\x01'

# Global storage for decision logging
kappa_values = []
spike_values = []
collision_flag = []
halt_frame_index = None  # Record frame index where system halted
current_video_name = None


def plot_result():
    """Plot Membrane Potential, Spikes, and Collision Flags (3 vertical subplots)"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=False, figsize=(6, 4),
                                        gridspec_kw={'height_ratios': [4, 1, 1]})
    x = np.arange(len(kappa_values))
    
    # Ensure X-axis starts from 0 and bars are fully visible
    if len(kappa_values) > 0:
        n = len(kappa_values)
        xlim_left = 0
        xlim_right = n
        ax1.set_xlim(xlim_left, xlim_right)
        ax2.set_xlim(xlim_left, xlim_right)
        ax3.set_xlim(xlim_left, xlim_right)
        
        # Set integer ticks for the bottom subplot
        step = max(1, n // 10)
        ax3.set_xticks(np.arange(0, n, step))
    
    # Subplot 1: Membrane Potential + Threshold
    ax1.plot(x, kappa_values, 'b-', linewidth=2, label=r'$\kappa_f$ (Membrane Potential)')
    ax1.axhline(y=0.8, color='g', linestyle='--', linewidth=2, label='Threshold $T_s$')
    # Placeholder for vertical line indicating specific event frame
    ax1.axvline(x=104, color='grey', linestyle='--', linewidth=1, alpha=0.7)

    ax1.set_xticks([])              
    ax1.set_ylim(0.4, 1.0)          
    ax1.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) 
    ax1.set_ylabel('Membrane Pot.', fontsize=10)
    ax1.set_title('', fontsize=14)

    # Subplot 2: Spike Bar Chart (0/1 values)
    ax2.bar(x, spike_values, width=0.5, color='red', alpha=0.7, label='Spike')
    ax2.set_xticks([]) 
    ax2.set_ylim(0, 2)                      
    ax2.set_yticks([0, 1])                   
    ax2.set_ylabel('Spike', fontsize=10, labelpad=14)
    
    # Subplot 3: Collision Detection Flags (0/1 values)
    ax3.bar(x, collision_flag, width=0.5, color='orange', alpha=0.7, label='Collision Detected')
    ax3.set_ylim(0, 2)
    ax3.set_yticks([1])
    ax3.set_xlabel('Frame Index', fontsize=12)
    ax3.set_ylabel('Collision', fontsize=10, labelpad=14)
    
    # Adjust layout to minimize gap between subplots
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.02)          
    
    # Save the result figure
    if current_video_name is not None:
        output_dir = r".\Result of Verification of The Role of Probability\blake"  #Save path
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, current_video_name + ".png")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Result plot saved to: {save_path}")
    else:
        print("Warning: Video name not found, plot not saved.")
    
    plt.show()


def generate_input_tag(frame_bytes, timestamp_ms, key):
    ts_bytes = struct.pack('<Q', timestamp_ms)
    message = DOMAIN_INPUT + frame_bytes + ts_bytes
    return blake3.blake3(message, key=key).digest()


def verify_input_tag(received_bytes, received_ts, received_tag, key, current_ts, time_threshold):
    if abs(current_ts - received_ts) > time_threshold:
        return False
    ts_bytes = struct.pack('<Q', received_ts)
    message = DOMAIN_INPUT + received_bytes + ts_bytes
    expected_tag = blake3.blake3(message, key=key).digest()
    return secrets.compare_digest(expected_tag, received_tag)


def generate_output_tag(spike, kappa, timestamp_ms, key):
    spike_byte = struct.pack('B', spike)
    kappa_bytes = struct.pack('<f', kappa)
    ts_bytes = struct.pack('<Q', timestamp_ms)
    message = DOMAIN_OUTPUT + spike_byte + kappa_bytes + ts_bytes
    return blake3.blake3(message, key=key).digest()


def verify_output_tag(received_spike, received_kappa, received_ts, received_tag, key, current_ts, time_threshold):
    if abs(current_ts - received_ts) > time_threshold:
        return False
    spike_byte = struct.pack('B', received_spike)
    kappa_bytes = struct.pack('<f', received_kappa)
    ts_bytes = struct.pack('<Q', received_ts)
    message = DOMAIN_OUTPUT + spike_byte + kappa_bytes + ts_bytes
    expected_tag = blake3.blake3(message, key=key).digest()
    return secrets.compare_digest(expected_tag, received_tag)


class LGMD2006:
    """Replication of IEEE TNN 2006 Classic LGMD Model (FFM removed, Ts=0.8)"""

    def __init__(self, first_frame, dt):
        self.dt = dt
        self.rows, self.cols = first_frame.shape
        self.n_cell = self.rows * self.cols

        self.W_I = 0.8
        self.C_de = 0.5
        self.T_de = 15
        self.DeltaT_lt = 0.03
        self.nsp = 3
        self.nts = 5
        self.T_mp = 0.8 
        self.Cw = 4 
        self.Delta_c = 0.01
        self.T_FO = 1000
        self.alpha_ffi = 0.02
        self.n_p = 1
        self.mu = 1

        self.w_I = np.array([[0.125, 0.25, 0.125],
                             [0.25, 0.0, 0.25],
                             [0.125, 0.25, 0.125]])
        self.w_e = np.ones((3, 3)) / 9.0

        self.L = [first_frame.copy(), first_frame.copy()]
        self.Pf = [np.zeros_like(first_frame), np.zeros_like(first_frame)]
        self.E = np.zeros_like(first_frame)
        self.If = np.zeros_like(first_frame)
        self.Sf = np.zeros_like(first_frame)
        self.Ce = np.zeros_like(first_frame)
        self.Gf = np.zeros_like(first_frame)
        self.Gf_tilde = np.zeros_like(first_frame)

        self.Ff = 0.0
        self.T_FFI = self.T_FO
        self.T_s = self.T_mp
        self.spikes = np.zeros(self.nts)

    def photoreceptor_layer(self, gray_frame):
        self.L[0] = self.L[1].copy()
        self.L[1] = gray_frame.copy()
        p1 = 1.0 / (1 + np.exp(self.mu * 1))
        delta_L = self.L[1] - self.L[0]
        self.Pf[0] = self.Pf[1].copy()
        self.Pf[1] = p1 * self.Pf[0] + delta_L

    def ei_layer(self):
        self.E = self.Pf[1].copy()
        self.If = cv2.filter2D(self.Pf[0], -1, self.w_I, borderType=cv2.BORDER_REPLICATE)

    def sum_layer(self):
        self.Sf = self.E - self.W_I * self.If
        self.Sf = np.maximum(self.Sf, 0)

    def grouping_layer(self):
        self.Ce = cv2.filter2D(self.Sf, -1, self.w_e, borderType=cv2.BORDER_REPLICATE)
        max_Ce = np.max(np.abs(self.Ce))
        omega = self.Delta_c + max_Ce / self.Cw
        self.Gf = self.Sf * self.Ce / omega
        self.Gf_tilde = np.where(self.Gf * self.C_de >= self.T_de, self.Gf, 0)

    def lgmd_cell(self):
        Kf = np.sum(np.abs(self.Gf_tilde))
        self.kappa_f = 1.0 / (1 + np.exp(-Kf / self.n_cell))

    def compute_spike(self):
        """Calculate spike based on current membrane potential (without history update)"""
        return 1 if self.kappa_f >= self.T_s else 0

    def update_collide(self, spike):
        """Update history with current spike and return collision flag"""
        self.spikes = np.roll(self.spikes, -1)
        self.spikes[-1] = spike
        self.collide = 1 if np.sum(self.spikes) >= self.nsp else 0
        return self.collide

    def forward(self, gray_frame):
        self.photoreceptor_layer(gray_frame)
        self.ei_layer()
        self.sum_layer()
        self.grouping_layer()
        self.lgmd_cell()
        spike = self.compute_spike()          
        return self.kappa_f, spike            



def main():
    global halt_frame_index, current_video_name

    # Most recent trusted values passing output verification
    last_accepted_kappa = 0.0
    last_accepted_spike = 0
    last_accepted_collide = 0

    # Most recent forward calculation results (independent of verification)
    last_forward_kappa = 0.0
    last_forward_spike = 0

    # ===================== Tamper Simulation Rules =====================
    # Input Tampering: {Target Frame Index: Source Frame Index to Swap}
    tamper_rules ={98:86,99:87,100:88,101:89,102:90,103:91,104:92,105:93,106:94,107:95,108:96,109:97}
    
    # Output Tampering: {Frame Indices to flip output spike signal}
    output_tamper_frames = {}   
    # ===================================================================


    #Path of the video
    video_path = r".\"

    video_file = os.path.basename(video_path)
    current_video_name = os.path.splitext(video_file)[0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file!")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    dt = 1000.0 / fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"FPS: {fps:.1f} | Frame Interval: {dt:.1f}ms | Total Frames: {total_frames}")

    # ========== Pre-load all frames into memory (BGR, uint8, float32) ==========
    all_frames_bgr = []      
    all_frames_uint8 = []    
    all_frames_float = []    
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        all_frames_bgr.append(frame_bgr.copy())
        frame_uint8 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        frame_float = frame_uint8.astype(np.float32)
        all_frames_uint8.append(frame_uint8)
        all_frames_float.append(frame_float)
    cap.release()
    print(f"Read {len(all_frames_uint8)} frames.")

    # Validate tamper rules
    max_frame_idx = len(all_frames_uint8)
    for target, src in list(tamper_rules.items()):
        if not (1 <= target <= max_frame_idx and 1 <= src <= max_frame_idx):
            print(f"Warning: Tamper rule {target}->{src} out of range, ignoring.")
            del tamper_rules[target]

    # ========== Initialize First Frame ==========
    first_frame_uint8 = all_frames_uint8[0]
    first_frame_float = all_frames_float[0]

    ts_send = int(time.time() * 1000)
    frame_bytes = first_frame_uint8.tobytes()
    tag_input = generate_input_tag(frame_bytes, ts_send, SHARED_KEY)

    ts_receive = int(time.time() * 1000)
    if not verify_input_tag(frame_bytes, ts_send, tag_input, SHARED_KEY, ts_receive, TIME_THRESHOLD_MS):
        print("First frame verification failed. Secure startup aborted!")
        return

    lgmd = LGMD2006(first_frame_float, dt)

    kappa, spike = lgmd.forward(first_frame_float)
    last_forward_kappa, last_forward_spike = kappa, spike

    ts_out_send = int(time.time() * 1000)
    tag_output = generate_output_tag(spike, kappa, ts_out_send, SHARED_KEY)
    ts_out_receive = int(time.time() * 1000)

    if verify_output_tag(spike, kappa, ts_out_send, tag_output, SHARED_KEY, ts_out_receive, TIME_THRESHOLD_MS):
        collide = lgmd.update_collide(spike)
        exec_last_kappa, exec_last_spike, exec_last_collide = kappa, spike, collide
        last_accepted_kappa, last_accepted_spike, last_accepted_collide = kappa, spike, collide
    else:
        print("First frame output verification failed, using safe defaults.")
        exec_last_kappa, exec_last_spike, exec_last_collide = 0.0, 0, 0

    kappa_values.append(exec_last_kappa)
    spike_values.append(exec_last_spike)
    collision_flag.append(exec_last_collide)

    frame_count = 1
    start_time = time.time()

    input_fail_count = 0
    output_fail_count = 0
    system_halted = False
    halt_frame_index = None

    # Sliding window to track failures in the last 5 frames
    recent_failures = deque(maxlen=5)
    recent_failures.append(False)

    # Main Processing Loop
    for idx in range(1, len(all_frames_uint8)):
        current_frame_idx = idx + 1

        if system_halted:
            exec_last_kappa, exec_last_spike, exec_last_collide = 0.0, 0, 0
            kappa_values.append(exec_last_kappa)
            spike_values.append(exec_last_spike)
            collision_flag.append(exec_last_collide)
            frame_count += 1
            print(f"Frame {frame_count:3d} | SYSTEM HALTED - output 0")
            cv2.imshow("Input Frame", all_frames_bgr[idx])
            if cv2.waitKey(10) & 0xFF == 27:
                break
            continue

        frame_bgr = all_frames_bgr[idx].copy()
        frame_uint8 = all_frames_uint8[idx].copy()
        frame_float = all_frames_float[idx].copy()

        # Input Auth: Generate tag for original frame
        ts_send = int(time.time() * 1000)
        frame_bytes = frame_uint8.tobytes()
        tag_input = generate_input_tag(frame_bytes, ts_send, SHARED_KEY)

        # Apply Input Tampering (modification after tag generation)
        if current_frame_idx in tamper_rules:
            src_idx = tamper_rules[current_frame_idx]
            if 1 <= src_idx <= max_frame_idx:
                frame_bgr = all_frames_bgr[src_idx - 1].copy()
                frame_uint8 = all_frames_uint8[src_idx - 1].copy()
                frame_float = all_frames_float[src_idx - 1].copy()
                print(f"Frame {current_frame_idx} TAMPERED with Content of Frame {src_idx}")

        ts_receive = int(time.time() * 1000)
        input_valid = verify_input_tag(frame_uint8.tobytes(), ts_send, tag_input, SHARED_KEY, ts_receive, TIME_THRESHOLD_MS)

        # Determine kappa/spike based on input verification
        if input_valid:
            kappa, spike = lgmd.forward(frame_float)
            last_forward_kappa, last_forward_spike = kappa, spike
            input_fail_count = 0
        else:
            print(f"Frame {current_frame_idx} Input Auth Failed: Using last forward result")
            kappa, spike = last_forward_kappa, last_forward_spike
            input_fail_count += 1

        # Output Auth
        ts_out_send = int(time.time() * 1000)
        tag_output = generate_output_tag(spike, kappa, ts_out_send, SHARED_KEY)

        # Apply Output Tampering: Flip spike signal (tag remains unchanged)
        if current_frame_idx in output_tamper_frames:
            spike = 1 - spike
            print(f"Frame {current_frame_idx} Output Spike TAMPERED (Flipped)")

        ts_out_receive = int(time.time() * 1000)
        output_valid = verify_output_tag(spike, kappa, ts_out_send, tag_output, SHARED_KEY, ts_out_receive, TIME_THRESHOLD_MS)

        if output_valid:
            output_fail_count = 0
        else:
            print(f"Frame {current_frame_idx} Output Auth Failed")
            output_fail_count += 1

        # Circuit-breaking Logic
        frame_fail = (not input_valid) or (not output_valid)
        recent_failures.append(frame_fail)

        stop_now = False
        reason = ""
        if len(recent_failures) >= 2 and recent_failures[-1] and recent_failures[-2]:
            stop_now = True
            reason = "Consecutive failures detected"
        elif len(recent_failures) >= 3 and sum(recent_failures) >= 3:
            stop_now = True
            reason = f"Failure threshold reached ({sum(recent_failures)} in last 5 frames)"

        if stop_now:
            system_halted = True
            halt_frame_index = current_frame_idx
            print(f"{reason}. SYSTEM HALTED at Frame: {halt_frame_index}")
            exec_last_kappa, exec_last_spike, exec_last_collide = 0.0, 0, 0
        else:
            if output_valid:
                # Auth passed -> Update history and adopt new values
                collide = lgmd.update_collide(spike)
                exec_last_kappa, exec_last_spike, exec_last_collide = kappa, spike, collide
                last_accepted_kappa, last_accepted_spike, last_accepted_collide = kappa, spike, collide
            else:
                # Auth failed -> Fallback to last trusted state
                exec_last_kappa, exec_last_spike, exec_last_collide = last_accepted_kappa, last_accepted_spike, last_accepted_collide

        # Final Logging
        kappa_values.append(exec_last_kappa)
        spike_values.append(exec_last_spike)
        collision_flag.append(exec_last_collide)

        frame_count += 1
        print(f"Frame {frame_count:3d} | Kappa: {kappa:.4f} | Spike: {spike} | Collide: {exec_last_collide} | Final: {exec_last_spike} | InFail: {input_fail_count} | OutFail: {output_fail_count}")

        cv2.imshow("Input Frame", frame_bgr)
        if cv2.waitKey(10) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    total_time = time.time() - start_time
    print(f"\nProcessing Complete: {frame_count} frames | Total Time: {total_time:.2f}s | Avg FPS: {frame_count / total_time:.1f}")
    plot_result()


if __name__ == "__main__":
    main()
