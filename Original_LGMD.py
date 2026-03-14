import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Set plotting parameters
plt.rcParams['axes.unicode_minus'] = False

# Global storage for result plotting
kappa_values = []
spike_values = []
collision_flag = []
current_video_name = None

def plot_result():
    """Plot Membrane Potential, Spikes, and Collision Flags (3 vertical subplots)"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=False, figsize=(6, 4),
                                        gridspec_kw={'height_ratios': [4, 1, 1]})
    x = np.arange(len(kappa_values))
    if len(kappa_values) > 0:
        n = len(kappa_values)
        xlim_left = 0
        xlim_right = n
        ax1.set_xlim(xlim_left, xlim_right)
        ax2.set_xlim(xlim_left, xlim_right)
        ax3.set_xlim(xlim_left, xlim_right)
        step = max(1, n // 10)
        ax3.set_xticks(np.arange(0, n, step))
    
    # Subplot 1: Membrane Potential
    ax1.plot(x, kappa_values, 'b-', linewidth=2, label=r'$\kappa_f$ (Membrane Potential)')
    ax1.axhline(y=0.8, color='g', linestyle='--', linewidth=2, label='Threshold $T_s$')
    ax1.set_xticks([])              
    ax1.set_ylim(0.4, 1.0)          
    ax1.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) 
    ax1.set_ylabel('Membrane Pot.', fontsize=10)
    ax1.set_title('', fontsize=14)

    # Subplot 2: Spike Bar Chart
    ax2.bar(x, spike_values, width=0.5, color='red', alpha=0.7, label='Spike')
    ax2.set_xticks([]) 
    ax2.set_ylim(0, 2)                      
    ax2.set_yticks([0, 1])                   
    ax2.set_ylabel('Spike', fontsize=10, labelpad=14)
    
    # Subplot 3: Collision Detection Flags
    ax3.bar(x, collision_flag, width=0.5, color='orange', alpha=0.7, label='Collision Detected')
    ax3.set_ylim(0, 2)
    ax3.set_yticks([1])
    ax3.set_xlabel('Frame Index', fontsize=12)
    ax3.set_ylabel('Collision', fontsize=10, labelpad=14)
    
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.02)          
    
    # Save the result figure
    if current_video_name is not None:
        output_dir = r".\Result of Verification of The Role of Probability\clean" #Save path
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, current_video_name + ".png")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Result plot saved to: {save_path}")
    else:
        print("Warning: Video name not found, plot not saved.")
    
    plt.show()

class LGMD2006:
    """Replication of IEEE TNN 2006 Classic LGMD Model (FFM removed, Ts=0.8)"""
    def __init__(self, first_frame, dt):
        self.dt = dt
        self.rows, self.cols = first_frame.shape
        self.n_cell = self.rows * self.cols

        # Model Parameters
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

        # Kernels for Excitation and Inhibition
        self.w_I = np.array([[0.125, 0.25, 0.125],
                             [0.25, 0.0, 0.25],
                             [0.125, 0.25, 0.125]])
        self.w_e = np.ones((3, 3)) / 9.0

        # Layer Initializations
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
        return self.kappa_f

    def compute_spike(self):
        return 1 if self.kappa_f >= self.T_s else 0

    def update_collide(self, spike):
        self.spikes = np.roll(self.spikes, -1)
        self.spikes[-1] = spike
        self.collide = 1 if np.sum(self.spikes) >= self.nsp else 0
        return self.collide

    def forward(self, gray_frame):
        self.photoreceptor_layer(gray_frame)
        self.ei_layer()
        self.sum_layer()
        self.grouping_layer()
        return self.lgmd_cell()

def main():
    global current_video_name

    # ===================== Tamper Simulation Rules =====================
    # Input Tampering: {Target Frame Index: Source Frame Index to Swap}
    tamper_rules = {98:86,99:87,100:88,101:89,102:90,103:91,104:92,105:93,106:94,107:95,108:96,109:97}
    
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

    # ========== Pre-load all frames into memory ==========
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

    max_frame_idx = len(all_frames_uint8)

    # Initialize LGMD with the first frame
    lgmd = LGMD2006(all_frames_float[0], dt)
    kappa = lgmd.forward(all_frames_float[0])
    spike = lgmd.compute_spike()
    collide = lgmd.update_collide(spike)
    
    kappa_values.append(kappa)
    spike_values.append(spike)
    collision_flag.append(collide)

    frame_count = 1
    start_time = time.time()

    # Main Processing Loop (from the second frame onwards)
    for idx in range(1, len(all_frames_uint8)):
        current_frame_idx = idx + 1
        
        frame_bgr = all_frames_bgr[idx].copy()
        frame_uint8 = all_frames_uint8[idx].copy()
        frame_float = all_frames_float[idx].copy()

        # ====== Simulate Input Tampering: Swap frame content ======
        if current_frame_idx in tamper_rules:
            src_idx = tamper_rules[current_frame_idx]
            if 1 <= src_idx <= max_frame_idx:
                frame_bgr = all_frames_bgr[src_idx - 1].copy()
                frame_uint8 = all_frames_uint8[src_idx - 1].copy()
                frame_float = all_frames_float[src_idx - 1].copy()
                print(f"Frame {current_frame_idx} TAMPERED with content of Frame {src_idx}")
            else:
                print(f"Warning: Source frame {src_idx} in tamper rules is invalid, skipping.")

        # Run LGMD model to obtain membrane potential
        kappa = lgmd.forward(frame_float)
        # Calculate raw spike
        spike = lgmd.compute_spike()

        # ====== Simulate Output Tampering: Flip spike signal ======
        if current_frame_idx in output_tamper_frames:
            spike = 1 - spike
            print(f"Frame {current_frame_idx} Output Spike TAMPERED (Flipped)")
        # ==========================================================

        # Update history based on (possibly tampered) spike and get collision flag
        collide = lgmd.update_collide(spike)

        # Record final results
        kappa_values.append(kappa)
        spike_values.append(spike)
        collision_flag.append(collide)

        frame_count += 1
        print(f"Frame {frame_count:3d} | Kappa: {kappa:.4f} | Spike: {spike} | Collide: {collide}")

        cv2.imshow("Input Frame", frame_bgr)
        if cv2.waitKey(10) & 0xFF == 27: # Press 'ESC' to exit
            break

    cv2.destroyAllWindows()
    total_time = time.time() - start_time
    print(f"\nProcessing Complete: {frame_count} frames | Total Time: {total_time:.2f}s | Avg FPS: {frame_count / total_time:.1f}")
    plot_result()

if __name__ == "__main__":
    main()
