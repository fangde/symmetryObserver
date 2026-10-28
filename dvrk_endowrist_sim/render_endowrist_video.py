"""
Generate Side-by-Side Comparison Video for da Vinci EndoWrist
Cable-Driven Precision & Force Control Benchmark
(Naive Linear PD vs Proposed Controlled Symmetry Diffeomorphic Alignment)
"""

import os
import mujoco
import numpy as np
import cv2
import imageio

def generate_endowrist_comparison_video():
    xml_path = "dvrk_endowrist_sim/endowrist.xml"
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Missing {xml_path}")
        
    os.makedirs("dvrk_endowrist_sim/results", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    model = mujoco.MjModel.from_xml_path(xml_path)
    dt = model.opt.timestep # 0.001s (1000 Hz)
    
    fps = 25
    subsample = int(1.0 / (fps * dt)) # record every 40 simulation steps (25 fps)
    T_total = 6.0
    total_steps = int(T_total / dt)
    time_arr = np.linspace(0, T_total, total_steps)
    
    # -----------------------------------------------------------------
    # Trajectory Profiles
    # Phase 1 [0.0 ~ 2.5s]: 3D Spatial Wrist Articulation (Pitch & Yaw)
    # Phase 2 [2.5 ~ 3.5s]: Jaw Closing Ramp onto Tissue Specimen
    # Phase 3 [3.5 ~ 6.0s]: Gentle Regulated Tissue Clamping (Target: 1.8 N)
    # -----------------------------------------------------------------
    pitch_des = 0.25 * np.sin(2 * np.pi * 0.8 * time_arr)
    dpitch_des = np.gradient(pitch_des, dt)
    
    yaw_des = 0.20 * np.sin(2 * np.pi * 0.6 * time_arr) + 0.06 * np.cos(2 * np.pi * 1.2 * time_arr)
    dyaw_des = np.gradient(yaw_des, dt)
    
    grasp_des = np.zeros(total_steps)
    for i, t in enumerate(time_arr):
        if t < 2.5:
            grasp_des[i] = -0.30 # wide open jaws (clears tissue)
        elif t < 3.5:
            alpha = (t - 2.5) / 1.0
            grasp_des[i] = -0.30 * (1.0 - alpha) + 0.03 * alpha
        else:
            grasp_des[i] = 0.03 # target closed state
    dgrasp_des = np.gradient(grasp_des, dt)
    
    F_target = 1.80 # 1.80 N safe clamping limit

    # Physical parameters
    r_pitch = 0.0035
    r_jaw   = 0.0028
    r_idler = 0.0025
    T_pre   = 1.5
    mu_alpha = 0.24
    
    tissue_geom_id = model.geom("tissue_geom").id

    def measure_clamping_force(d):
        fn_j1, fn_j2 = 0.0, 0.0
        for k in range(d.ncon):
            c = d.contact[k]
            if c.geom1 == tissue_geom_id or c.geom2 == tissue_geom_id:
                forces = np.zeros(6)
                mujoco.mj_contactForce(model, d, k, forces)
                fn_val = np.abs(forces[0])
                other_geom = c.geom2 if c.geom1 == tissue_geom_id else c.geom1
                geom_name = model.geom(other_geom).name
                if "jaw1" in geom_name:
                    fn_j1 += fn_val
                elif "jaw2" in geom_name:
                    fn_j2 += fn_val
        return 0.5 * (fn_j1 + fn_j2)

    # Simulators for side-by-side rendering
    data_naive = mujoco.MjData(model)
    data_prop  = mujoco.MjData(model)
    
    # Camera setup (focused close-up on wrist articulation and jaws)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = 135
    cam.elevation = -18
    cam.distance = 0.115
    cam.lookat = [0.0, 0.0, 0.120]
    
    vopt = mujoco.MjvOption()
    vopt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = 1 # render active cables

    H, W = 460, 520
    renderer_naive = mujoco.Renderer(model, height=H, width=W)
    renderer_prop  = mujoco.Renderer(model, height=H, width=W)
    
    # Controller states
    Kp_p1, Kd_p1 = 8.0, 0.15
    Kp_y1, Kd_y1 = 7.0, 0.12
    Kp_g1, Kd_g1 = 10.0, 0.15
    
    Kp_canon, Kd_canon = 18.0, 0.35
    f_integral = 0.0
    
    frames = []
    print(f"Rendering {total_steps // subsample} frames at {fps} FPS...")
    
    for step in range(total_steps):
        t = time_arr[step]
        
        # -------------------------------------------------------------
        # 1. Controller 1: Naive Decoupled Linear PD
        # -------------------------------------------------------------
        qp_n = data_naive.sensor("pos_pitch").data[0]
        dqp_n = data_naive.sensor("vel_pitch").data[0]
        qj1_n = data_naive.sensor("pos_jaw1").data[0]
        dqj1_n = data_naive.sensor("vel_jaw1").data[0]
        qj2_n = data_naive.sensor("pos_jaw2").data[0]
        dqj2_n = data_naive.sensor("vel_jaw2").data[0]
        
        qy_n = 0.5 * (qj1_n + qj2_n)
        dqy_n = 0.5 * (dqj1_n + dqj2_n)
        qg_n = (qj1_n - qj2_n)
        dqg_n = (dqj1_n - dqj2_n)
        
        tau_p_n = Kp_p1 * (pitch_des[step] - qp_n) + Kd_p1 * (dpitch_des[step] - dqp_n)
        tau_y_n = Kp_y1 * (yaw_des[step] - qy_n) + Kd_y1 * (dyaw_des[step] - dqy_n)
        tau_g_n = Kp_g1 * (grasp_des[step] - qg_n) + Kd_g1 * (dgrasp_des[step] - dqg_n)
        
        data_naive.ctrl[1] = np.clip(T_pre + max(0.0, -tau_p_n / r_pitch), 0, 20)
        data_naive.ctrl[2] = np.clip(T_pre + max(0.0, tau_p_n / r_pitch), 0, 20)
        
        tau_j1_n = tau_y_n + 0.5 * tau_g_n
        tau_j2_n = tau_y_n - 0.5 * tau_g_n
        data_naive.ctrl[3] = np.clip(T_pre + max(0.0, tau_j1_n / r_jaw), 0, 25)
        data_naive.ctrl[4] = np.clip(T_pre + max(0.0, -tau_j1_n / r_jaw), 0, 25)
        data_naive.ctrl[5] = np.clip(T_pre + max(0.0, -tau_j2_n / r_jaw), 0, 25)
        data_naive.ctrl[6] = np.clip(T_pre + max(0.0, tau_j2_n / r_jaw), 0, 25)
        
        mujoco.mj_step(model, data_naive)
        
        # -------------------------------------------------------------
        # 2. Controller 2: Proposed Controlled Symmetry Diffeomorphic Alignment
        # -------------------------------------------------------------
        qp_p = data_prop.sensor("pos_pitch").data[0]
        dqp_p = data_prop.sensor("vel_pitch").data[0]
        qj1_p = data_prop.sensor("pos_jaw1").data[0]
        dqj1_p = data_prop.sensor("vel_jaw1").data[0]
        qj2_p = data_prop.sensor("pos_jaw2").data[0]
        dqj2_p = data_prop.sensor("vel_jaw2").data[0]
        
        qy_p = 0.5 * (qj1_p + qj2_p)
        dqy_p = 0.5 * (dqj1_p + dqj2_p)
        qg_p = (qj1_p - qj2_p)
        dqg_p = (dqj1_p - dqj2_p)
        
        fn_p = measure_clamping_force(data_prop)
        
        # Canonical virtual controls
        ep = pitch_des[step] - qp_p
        dep = dpitch_des[step] - dqp_p
        tau_p_c = Kp_canon * ep + Kd_canon * dep
        
        ey = yaw_des[step] - qy_p
        dey = dyaw_des[step] - dqy_p
        tau_y_c = Kp_canon * ey + Kd_canon * dey
        
        # Admittance force regulation
        if t < 2.5:
            eg = grasp_des[step] - qg_p
            deg = dgrasp_des[step] - dqg_p
            tau_g_c = 15.0 * eg + 0.3 * deg
        else:
            ef = F_target - fn_p
            f_integral += ef * dt
            tau_contact_nom = 0.0125 * F_target
            tau_g_c = tau_contact_nom + 0.006 * ef + 0.003 * f_integral
            tau_g_c = np.clip(tau_g_c, 0.002, 0.045)
            
        # Cross-coupling decoupling
        tau_coupling = (r_idler / r_pitch) * tau_p_c * 0.15
        tau_j1_c = tau_y_c + 0.5 * tau_g_c - tau_coupling
        tau_j2_c = tau_y_c - 0.5 * tau_g_c + tau_coupling
        
        # Capstan friction diffeomorphic mapping
        fric_pitch = np.exp(mu_alpha * np.tanh(8.0 * dpitch_des[step]))
        fric_j1 = np.exp(mu_alpha * np.tanh(8.0 * dqj1_p))
        fric_j2 = np.exp(mu_alpha * np.tanh(8.0 * dqj2_p))
        
        u_lead_p = 0.05 * np.tanh(4.0 * dpitch_des[step])
        tau_p_total = tau_p_c + u_lead_p
        
        T_pitch_n = (T_pre + max(0.0, tau_p_total / r_pitch)) * fric_pitch
        T_pitch_p = (T_pre + max(0.0, -tau_p_total / r_pitch)) / max(fric_pitch, 0.1)
        
        T_j1_close = (T_pre + max(0.0, tau_j1_c / r_jaw)) * fric_j1
        T_j1_open  = (T_pre + max(0.0, -tau_j1_c / r_jaw)) / max(fric_j1, 0.1)
        
        T_j2_close = (T_pre + max(0.0, -tau_j2_c / r_jaw)) * fric_j2
        T_j2_open  = (T_pre + max(0.0, tau_j2_c / r_jaw)) / max(fric_j2, 0.1)
        
        data_prop.ctrl[1] = np.clip(T_pitch_p, 0, 20)
        data_prop.ctrl[2] = np.clip(T_pitch_n, 0, 20)
        data_prop.ctrl[3] = np.clip(T_j1_close, 0, 25)
        data_prop.ctrl[4] = np.clip(T_j1_open, 0, 25)
        data_prop.ctrl[5] = np.clip(T_j2_close, 0, 25)
        data_prop.ctrl[6] = np.clip(T_j2_open, 0, 25)
        
        mujoco.mj_step(model, data_prop)
        
        # -------------------------------------------------------------
        # Frame Recording & HUD Overlay
        # -------------------------------------------------------------
        if step % subsample == 0:
            renderer_naive.update_scene(data_naive, camera=cam, scene_option=vopt)
            img_n = renderer_naive.render()
            
            renderer_prop.update_scene(data_prop, camera=cam, scene_option=vopt)
            img_p = renderer_prop.render()
            
            # Combine side by side
            combined = np.zeros((H, W * 2, 3), dtype=np.uint8)
            combined[:, :W] = img_n
            combined[:, W:] = img_p
            
            # Vertical divider line
            combined[:, W-2:W+2] = [45, 55, 75]
            
            # 1. Top Header Banner
            cv2.rectangle(combined, (0, 0), (W * 2, 45), (15, 23, 42), -1)
            cv2.putText(combined, "da Vinci EndoWrist Cable Simulation - Controlled Symmetry Benchmark", 
                        (W - 310, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
            
            # 2. Left Tag: Naive PD
            fn_n = measure_clamping_force(data_naive)
            err_p_n = pitch_des[step] - qp_n
            cv2.rectangle(combined, (15, 55), (320, 105), (20, 20, 35), -1)
            cv2.rectangle(combined, (15, 55), (320, 105), (80, 60, 220), 1)
            cv2.putText(combined, "Baseline: Naive Decoupled PD", 
                        (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (120, 160, 255), 1, cv2.LINE_AA)
            cv2.putText(combined, f"Pitch Err: {err_p_n:+.3f} rad | Force: {fn_n:.2f} N", 
                        (25, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 230), 1, cv2.LINE_AA)
            
            # 3. Right Tag: Proposed Controller
            err_p_p = pitch_des[step] - qp_p
            cv2.rectangle(combined, (W + 15, 55), (W + 360, 105), (15, 35, 25), -1)
            cv2.rectangle(combined, (W + 15, 55), (W + 360, 105), (40, 200, 120), 1)
            cv2.putText(combined, "Ours: Controlled Symmetry Diffeomorphic", 
                        (W + 25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (80, 240, 160), 1, cv2.LINE_AA)
            cv2.putText(combined, f"Pitch Err: {err_p_p:+.3f} rad | Force: {fn_p:.2f} N (Target: 1.8N)", 
                        (W + 25, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 240, 220), 1, cv2.LINE_AA)
            
            # 4. Bottom Status Bar
            if t < 2.5:
                phase_text = "Phase 1: Free-Space 3D Articulation (Pitch & Yaw Tracking)"
            elif t < 3.5:
                phase_text = "Phase 2: Aperture Closing Ramp onto Elastic Tissue Specimen"
            else:
                phase_text = "Phase 3: Regulated Contact Clamping (Target Force = 1.80 N)"
                
            cv2.rectangle(combined, (0, H - 36), (W * 2, H), (15, 23, 42), -1)
            cv2.putText(combined, f"t = {t:.2f}s | {phase_text}", 
                        (25, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 220, 240), 1, cv2.LINE_AA)
            
            frames.append(combined)

    # Export MP4
    mp4_out = "dvrk_endowrist_sim/endowrist_control_demo.mp4"
    imageio.mimwrite(mp4_out, frames, fps=fps, quality=8)
    print(f"Saved MP4 video to {mp4_out} ({len(frames)} frames)")
    
    # Export animated GIF for documentation / markdown preview (subsampled to 12.5 fps)
    gif_out = "dvrk_endowrist_sim/endowrist_control_demo.gif"
    imageio.mimsave(gif_out, frames[::2], duration=1.0/12.5, loop=0)
    print(f"Saved GIF animation to {gif_out}")
    
    # Copy to results/ and current conversation brain directory
    os.system(f"cp {mp4_out} results/endowrist_control_demo.mp4")
    os.system(f"cp {gif_out} results/endowrist_control_demo.gif")
    
    brain_dir = "/Users/fangdeliu/.gemini/antigravity/brain/81435b2d-481f-4955-a6da-3d4399357ab4"
    os.makedirs(brain_dir, exist_ok=True)
    os.system(f"cp {mp4_out} {brain_dir}/endowrist_control_demo.mp4")
    os.system(f"cp {gif_out} {brain_dir}/endowrist_control_demo.gif")
    print(f"Copied demo video and GIF to {brain_dir} successfully.")

if __name__ == "__main__":
    generate_endowrist_comparison_video()
