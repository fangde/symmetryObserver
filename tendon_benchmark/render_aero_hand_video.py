import os
import mujoco
import numpy as np
import imageio
import cv2

def generate_aero_hand_comparison_video():
    xml_path = "aero_open_sim/mujoco/scene_right.xml"
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Cannot find Aero Hand MuJoCo scene at {xml_path}")
        
    os.makedirs("tendon_benchmark/results", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    model = mujoco.MjModel.from_xml_path(xml_path)
    dt = model.opt.timestep # 0.01s (100 Hz)
    
    fps = 25
    subsample = int(1.0 / (fps * dt)) # step 4 sim steps per video frame
    T_total = 6.0
    total_sim_steps = int(T_total / dt)
    time_arr = np.linspace(0, T_total, total_sim_steps)
    
    # -------------------------------------------------------------
    # 1. Trajectory: Grasp + Wave Motion
    # -------------------------------------------------------------
    finger_acts = [0, 1, 2, 3] # Index, Middle, Ring, Pinky
    finger_jnts = [0, 3, 6, 9] # MCP joints
    finger_names = ["Index", "Middle", "Ring", "Pinky"]
    
    # Target trajectories for 4 fingers
    targets = np.zeros((4, total_sim_steps))
    for i, t in enumerate(time_arr):
        if t < 3.0:
            # Phase 1: Coordinated Grasp (tight fist and reopen)
            th_base = 0.65 + 0.50 * np.sin(2 * np.pi * 0.35 * t - np.pi/2)
            for f in range(4):
                targets[f, i] = np.clip(th_base, 0.05, 1.25)
        else:
            # Phase 2: Sequential Finger Wave (piano motion)
            t_rel = t - 3.0
            for f in range(4):
                phase_shift = f * 0.4
                targets[f, i] = np.clip(0.60 + 0.50 * np.sin(2 * np.pi * 0.8 * t_rel - phase_shift), 0.05, 1.25)
                
    # -------------------------------------------------------------
    # 2. Calibration of Diffeomorphism: theta <-> u
    # -------------------------------------------------------------
    ctrl_min, ctrl_max = model.actuator_ctrlrange[0]
    ctrl_grid = np.linspace(ctrl_max, ctrl_min, 120)
    theta_grid = []
    for c in ctrl_grid:
        d = mujoco.MjData(model)
        d.ctrl[0] = c
        for _ in range(250):
            mujoco.mj_step(model, d)
        theta_grid.append(d.qpos[0])
    theta_grid = np.array(theta_grid)
    s_idx = np.argsort(theta_grid)
    theta_clean = theta_grid[s_idx]
    ctrl_clean = ctrl_grid[s_idx]
    
    def diffeo_map(th):
        return np.interp(th, theta_clean, ctrl_clean)
        
    # -------------------------------------------------------------
    # 3. Two Simulators for Side-by-Side Video
    # -------------------------------------------------------------
    data_naive = mujoco.MjData(model)
    data_prop = mujoco.MjData(model)
    
    # Camera setup (focused 3D perspective)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = 142
    cam.elevation = -22
    cam.distance = 0.39
    cam.lookat = [0.12, 0.0, -0.06]
    
    H, W = 400, 520
    renderer_naive = mujoco.Renderer(model, height=H, width=W)
    renderer_prop = mujoco.Renderer(model, height=H, width=W)
    
    frames = []
    Kp_naive, Kd_naive = 0.035, 0.002
    u_mid = 0.092
    
    print(f"Rendering {total_sim_steps // subsample} frames at {fps} FPS...")
    
    for step in range(total_sim_steps):
        t = time_arr[step]
        
        # --- Controller 1: Naive Linear PD ---
        for f in range(4):
            act = finger_acts[f]
            jnt = finger_jnts[f]
            th = data_naive.qpos[jnt]
            dth = data_naive.qvel[jnt]
            des = targets[f, step]
            des_d = (targets[f, min(step+1, total_sim_steps-1)] - targets[f, max(step-1, 0)]) / (2*dt)
            e = des - th
            de = des_d - dth
            u = u_mid - Kp_naive * e - Kd_naive * de
            data_naive.ctrl[act] = np.clip(u, ctrl_min, ctrl_max)
        mujoco.mj_step(model, data_naive)
        
        # --- Controller 2: Proposed Controlled Symmetry Diffeomorphic Alignment ---
        for f in range(4):
            act = finger_acts[f]
            jnt = finger_jnts[f]
            th = data_prop.qpos[jnt]
            dth = data_prop.qvel[jnt]
            des = targets[f, step]
            des_d = (targets[f, min(step+1, total_sim_steps-1)] - targets[f, max(step-1, 0)]) / (2*dt)
            e = des - th
            de = des_d - dth
            th_virt = des + 0.35 * e + 0.03 * de
            u_base = diffeo_map(th_virt)
            u_lead = -0.0018 * np.tanh(3.0 * des_d)
            data_prop.ctrl[act] = np.clip(u_base + u_lead, ctrl_min, ctrl_max)
        mujoco.mj_step(model, data_prop)
        
        # Record video frame
        if step % subsample == 0:
            renderer_naive.update_scene(data_naive, camera=cam)
            img_n = renderer_naive.render()
            
            renderer_prop.update_scene(data_prop, camera=cam)
            img_p = renderer_prop.update_scene(data_prop, camera=cam)
            img_p = renderer_prop.render()
            
            # Combine side by side: [Left: Naive, Right: Proposed]
            combined = np.zeros((H, W * 2, 3), dtype=np.uint8)
            combined[:, :W] = img_n
            combined[:, W:] = img_p
            
            # Add split line
            combined[:, W-2:W+2] = [40, 50, 70]
            
            # HUD Overlays using OpenCV
            # Top Banner
            cv2.rectangle(combined, (0, 0), (W * 2, 45), (15, 23, 42), -1)
            cv2.putText(combined, "Aero Hand Open (MuJoCo) - Controlled Symmetry Benchmark", 
                        (W - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Left Tag: Naive PD
            cv2.rectangle(combined, (15, 55), (280, 85), (20, 20, 35), -1)
            cv2.rectangle(combined, (15, 55), (280, 85), (80, 60, 220), 1)
            err_n = targets[0, step] - data_naive.qpos[0]
            cv2.putText(combined, f"Naive PD: Index Err = {err_n:+.2f} rad", 
                        (25, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 150, 255), 1, cv2.LINE_AA)
            
            # Right Tag: Proposed Diffeomorphic Alignment
            cv2.rectangle(combined, (W + 15, 55), (W + 330, 85), (15, 35, 25), -1)
            cv2.rectangle(combined, (W + 15, 55), (W + 330, 85), (40, 200, 120), 1)
            err_p = targets[0, step] - data_prop.qpos[0]
            cv2.putText(combined, f"Controlled Symmetry: Err = {err_p:+.2f} rad", 
                        (W + 25, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 240, 160), 1, cv2.LINE_AA)
            
            # Bottom Motion Status
            motion_desc = "Phase 1: Coordinated Grasping" if t < 3.0 else "Phase 2: Sequential Piano Wave"
            cv2.rectangle(combined, (0, H - 35), (W * 2, H), (15, 23, 42), -1)
            cv2.putText(combined, f"t = {t:.2f}s | {motion_desc} | Desired Angle = {targets[0, step]:.2f} rad", 
                        (30, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 210, 230), 1, cv2.LINE_AA)
            
            frames.append(combined)
            
    # Save MP4 video
    mp4_path = "tendon_benchmark/results/aero_hand_control_demo.mp4"
    imageio.mimwrite(mp4_path, frames, fps=fps, quality=8)
    print(f"Saved MP4 video to {mp4_path} ({len(frames)} frames)")
    
    # Save optimized animated GIF for web embedding (subsample to 15 fps)
    gif_path = "tendon_benchmark/results/aero_hand_control_demo.gif"
    gif_frames = frames[::2] # 12.5 fps for fast web loading
    imageio.mimsave(gif_path, gif_frames, duration=1.0/12.5, loop=0)
    print(f"Saved GIF animation to {gif_path}")
    
    # Copy to results/ and brain
    os.system(f"cp {mp4_path} results/aero_hand_control_demo.mp4")
    os.system(f"cp {gif_path} results/aero_hand_control_demo.gif")
    brain_dir = "/Users/fangdeliu/.gemini/antigravity/brain/8988dc4f-935c-4c53-819e-72e1f2892381"
    os.system(f"cp {mp4_path} {brain_dir}/aero_hand_control_demo.mp4")
    os.system(f"cp {gif_path} {brain_dir}/aero_hand_control_demo.gif")
    print("Copied video files to results/ and brain directory successfully.")

if __name__ == "__main__":
    generate_aero_hand_comparison_video()
