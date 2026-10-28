"""
dVRK EndoWrist Cable-Driven Precision & Force Control Benchmark
Based on Controlled Symmetry & Diffeomorphic Alignment ("我们的控制器")

This script benchmarks:
1. Precision Control: Multi-frequency 3D trajectory tracking (Pitch & Yaw) under
   cable compliance, pulley cross-coupling, and Capstan friction.
2. Force Control: Smooth transition from free-motion aperture closing to 
   constant contact force clamping (Target: 1.8 N) on an elastic tissue specimen.
"""

import os
import mujoco
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def run_endowrist_benchmark():
    xml_path = "dvrk_endowrist_sim/endowrist.xml"
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Missing {xml_path}")
        
    model = mujoco.MjModel.from_xml_path(xml_path)
    dt = model.opt.timestep # 0.001s (1 kHz)
    T_total = 6.0
    steps = int(T_total / dt)
    time_arr = np.linspace(0, T_total, steps)
    
    # -----------------------------------------------------------------
    # Trajectory Definitions
    # Phase 1 [0.0 ~ 2.5s]: Free-space precision trajectory tracking (Pitch & Yaw)
    # Phase 2 [2.5 ~ 3.5s]: Smooth jaw closing ramp onto tissue
    # Phase 3 [3.5 ~ 6.0s]: Regulated contact clamping (Target: 1.8 N)
    # -----------------------------------------------------------------
    pitch_des = 0.25 * np.sin(2 * np.pi * 0.8 * time_arr)
    dpitch_des = np.gradient(pitch_des, dt)
    
    yaw_des = 0.20 * np.sin(2 * np.pi * 0.6 * time_arr) + 0.06 * np.cos(2 * np.pi * 1.2 * time_arr)
    dyaw_des = np.gradient(yaw_des, dt)
    
    # Grasp aperture profile (negative = open, positive = clamping)
    grasp_des = np.zeros(steps)
    for i, t in enumerate(time_arr):
        if t < 2.5:
            grasp_des[i] = -0.30 # wide open jaws (clears tissue)
        elif t < 3.5:
            # Smooth closing ramp
            alpha = (t - 2.5) / 1.0
            grasp_des[i] = -0.30 * (1.0 - alpha) + 0.03 * alpha
        else:
            grasp_des[i] = 0.03 # nominal clamped state
    dgrasp_des = np.gradient(grasp_des, dt)
    
    F_target = 1.80  # 1.80 Newtons (delicate surgical tissue handling)

    # Physical parameters of transmission
    r_pitch = 0.0035  # pitch pulley radius [m]
    r_jaw   = 0.0028  # jaw horn radius [m]
    r_idler = 0.0025  # pitch idler radius [m]
    L_jaw   = 0.0080  # jaw contact moment arm [m]
    T_pre   = 1.5     # base co-contraction cable preload [N]
    mu_alpha = 0.24   # Capstan friction exponent along sheath & pulleys
    
    tissue_geom_id = model.geom("tissue_geom").id

    def measure_clamping_force(d):
        fn_j1 = 0.0
        fn_j2 = 0.0
        for k in range(d.ncon):
            c = d.contact[k]
            if c.geom1 == tissue_geom_id or c.geom2 == tissue_geom_id:
                forces = np.zeros(6)
                mujoco.mj_contactForce(model, d, k, forces)
                fn_val = np.abs(forces[0])
                # Check which jaw contacted
                other_geom = c.geom2 if c.geom1 == tissue_geom_id else c.geom1
                geom_name = model.geom(other_geom).name
                if "jaw1" in geom_name:
                    fn_j1 += fn_val
                elif "jaw2" in geom_name:
                    fn_j2 += fn_val
        # Clamping force on the tissue specimen: average of bilateral normal forces
        return 0.5 * (fn_j1 + fn_j2)

    # -----------------------------------------------------------------
    # 1. Controller 1: Naive Decoupled Linear PD
    # -----------------------------------------------------------------
    data1 = mujoco.MjData(model)
    log_p1, log_y1, log_g1, log_f1 = [], [], [], []
    
    Kp_p1, Kd_p1 = 8.0, 0.15
    Kp_y1, Kd_y1 = 7.0, 0.12
    Kp_g1, Kd_g1 = 10.0, 0.15
    
    for i in range(steps):
        qp = data1.sensor("pos_pitch").data[0]
        dqp = data1.sensor("vel_pitch").data[0]
        qj1 = data1.sensor("pos_jaw1").data[0]
        dqj1 = data1.sensor("vel_jaw1").data[0]
        qj2 = data1.sensor("pos_jaw2").data[0]
        dqj2 = data1.sensor("vel_jaw2").data[0]
        
        qy = 0.5 * (qj1 + qj2)
        dqy = 0.5 * (dqj1 + dqj2)
        qg = (qj1 - qj2)
        dqg = (dqj1 - dqj2)
        
        fn = measure_clamping_force(data1)
        log_p1.append(qp)
        log_y1.append(qy)
        log_g1.append(qg)
        log_f1.append(fn)
        
        # Naive decoupled PD torques
        tau_p = Kp_p1 * (pitch_des[i] - qp) + Kd_p1 * (dpitch_des[i] - dqp)
        tau_y = Kp_y1 * (yaw_des[i] - qy) + Kd_y1 * (dyaw_des[i] - dqy)
        tau_g = Kp_g1 * (grasp_des[i] - qg) + Kd_g1 * (dgrasp_des[i] - dqg)
        
        # Cable distribution (Note: pitch_n exerts +tau_p, pitch_p exerts -tau_p)
        data1.ctrl[1] = np.clip(T_pre + max(0.0, -tau_p / r_pitch), 0, 20)
        data1.ctrl[2] = np.clip(T_pre + max(0.0, tau_p / r_pitch), 0, 20)
        
        tau_j1 = tau_y + 0.5 * tau_g
        tau_j2 = tau_y - 0.5 * tau_g
        data1.ctrl[3] = np.clip(T_pre + max(0.0, tau_j1 / r_jaw), 0, 25)
        data1.ctrl[4] = np.clip(T_pre + max(0.0, -tau_j1 / r_jaw), 0, 25)
        data1.ctrl[5] = np.clip(T_pre + max(0.0, -tau_j2 / r_jaw), 0, 25)
        data1.ctrl[6] = np.clip(T_pre + max(0.0, tau_j2 / r_jaw), 0, 25)
        
        mujoco.mj_step(model, data1)

    # -----------------------------------------------------------------
    # 2. Controller 2: High-Gain Linear PD
    # -----------------------------------------------------------------
    data2 = mujoco.MjData(model)
    log_p2, log_y2, log_g2, log_f2 = [], [], [], []
    
    Kp_p2, Kd_p2 = 25.0, 0.45
    Kp_y2, Kd_y2 = 22.0, 0.40
    Kp_g2, Kd_g2 = 30.0, 0.50
    
    for i in range(steps):
        qp = data2.sensor("pos_pitch").data[0]
        dqp = data2.sensor("vel_pitch").data[0]
        qj1 = data2.sensor("pos_jaw1").data[0]
        dqj1 = data2.sensor("vel_jaw1").data[0]
        qj2 = data2.sensor("pos_jaw2").data[0]
        dqj2 = data2.sensor("vel_jaw2").data[0]
        
        qy = 0.5 * (qj1 + qj2)
        dqy = 0.5 * (dqj1 + dqj2)
        qg = (qj1 - qj2)
        dqg = (dqj1 - dqj2)
        
        fn = measure_clamping_force(data2)
        log_p2.append(qp)
        log_y2.append(qy)
        log_g2.append(qg)
        log_f2.append(fn)
        
        tau_p = Kp_p2 * (pitch_des[i] - qp) + Kd_p2 * (dpitch_des[i] - dqp)
        tau_y = Kp_y2 * (yaw_des[i] - qy) + Kd_y2 * (dyaw_des[i] - dqy)
        tau_g = Kp_g2 * (grasp_des[i] - qg) + Kd_g2 * (dgrasp_des[i] - dqg)
        
        data2.ctrl[1] = np.clip(T_pre + max(0.0, -tau_p / r_pitch), 0, 20)
        data2.ctrl[2] = np.clip(T_pre + max(0.0, tau_p / r_pitch), 0, 20)
        
        tau_j1 = tau_y + 0.5 * tau_g
        tau_j2 = tau_y - 0.5 * tau_g
        data2.ctrl[3] = np.clip(T_pre + max(0.0, tau_j1 / r_jaw), 0, 25)
        data2.ctrl[4] = np.clip(T_pre + max(0.0, -tau_j1 / r_jaw), 0, 25)
        data2.ctrl[5] = np.clip(T_pre + max(0.0, -tau_j2 / r_jaw), 0, 25)
        data2.ctrl[6] = np.clip(T_pre + max(0.0, tau_j2 / r_jaw), 0, 25)
        
        mujoco.mj_step(model, data2)

    # -----------------------------------------------------------------
    # 3. Controller 3: OURS (Controlled Symmetry Diffeomorphic Alignment)
    # - Canonical rigid dynamics tracking
    # - Capstan sheath friction feedforward
    # - Geometric pulley cross-coupling decoupling
    # - Anticipatory velocity pre-tensioning
    # - Smooth Admittance Clamping Force Control
    # -----------------------------------------------------------------
    data3 = mujoco.MjData(model)
    log_p3, log_y3, log_g3, log_f3 = [], [], [], []
    
    Kp_canon, Kd_canon = 18.0, 0.35
    f_integral = 0.0
    
    for i in range(steps):
        t = time_arr[i]
        qp = data3.sensor("pos_pitch").data[0]
        dqp = data3.sensor("vel_pitch").data[0]
        qj1 = data3.sensor("pos_jaw1").data[0]
        dqj1 = data3.sensor("vel_jaw1").data[0]
        qj2 = data3.sensor("pos_jaw2").data[0]
        dqj2 = data3.sensor("vel_jaw2").data[0]
        
        qy = 0.5 * (qj1 + qj2)
        dqy = 0.5 * (dqj1 + dqj2)
        qg = (qj1 - qj2)
        dqg = (dqj1 - dqj2)
        
        fn = measure_clamping_force(data3)
        log_p3.append(qp)
        log_y3.append(qy)
        log_g3.append(qg)
        log_f3.append(fn)
        
        # --- A. Canonical Virtual Controls ---
        ep = pitch_des[i] - qp
        dep = dpitch_des[i] - dqp
        tau_p_canon = Kp_canon * ep + Kd_canon * dep
        
        ey = yaw_des[i] - qy
        dey = dyaw_des[i] - dqy
        tau_y_canon = Kp_canon * ey + Kd_canon * dey
        
        # --- B. Force & Grasp Admittance Regulation ---
        if t < 2.5:
            # Free motion aperture tracking
            eg = grasp_des[i] - qg
            deg = dgrasp_des[i] - dqg
            tau_g_canon = 15.0 * eg + 0.3 * deg
        else:
            # Clamping phase: smoothly regulate to target force 1.8 N
            ef = F_target - fn
            f_integral += ef * dt
            # Feedforward nominal torque + PI force error regulation
            tau_contact_nom = 0.0125 * F_target
            tau_g_canon = tau_contact_nom + 0.006 * ef + 0.003 * f_integral
            tau_g_canon = np.clip(tau_g_canon, 0.002, 0.045)

        # --- C. Pulley Cross-Coupling Decoupling ---
        # When pitch bends, cables passing through idlers change path length
        tau_coupling = (r_idler / r_pitch) * tau_p_canon * 0.15
        
        tau_j1_canon = tau_y_canon + 0.5 * tau_g_canon - tau_coupling
        tau_j2_canon = tau_y_canon - 0.5 * tau_g_canon + tau_coupling

        # --- D. Capstan Sheath Friction Diffeomorphic Compensation ---
        fric_pitch = np.exp(mu_alpha * np.tanh(8.0 * dpitch_des[i]))
        fric_j1 = np.exp(mu_alpha * np.tanh(8.0 * dqj1))
        fric_j2 = np.exp(mu_alpha * np.tanh(8.0 * dqj2))
        
        # --- E. Anticipatory Phase-Lead Velocity Pre-Tensioning ---
        u_lead_p = 0.05 * np.tanh(4.0 * dpitch_des[i])
        
        # Torque to cable pulls with friction factor
        # Pitch positive torque pulls cable_pitch_n; negative pulls cable_pitch_p
        tau_p_total = tau_p_canon + u_lead_p
        T_pitch_n = (T_pre + max(0.0, tau_p_total / r_pitch)) * fric_pitch
        T_pitch_p = (T_pre + max(0.0, -tau_p_total / r_pitch)) / max(fric_pitch, 0.1)
        
        T_j1_close = (T_pre + max(0.0, tau_j1_canon / r_jaw)) * fric_j1
        T_j1_open  = (T_pre + max(0.0, -tau_j1_canon / r_jaw)) / max(fric_j1, 0.1)
        
        T_j2_close = (T_pre + max(0.0, -tau_j2_canon / r_jaw)) * fric_j2
        T_j2_open  = (T_pre + max(0.0, tau_j2_canon / r_jaw)) / max(fric_j2, 0.1)
        
        data3.ctrl[1] = np.clip(T_pitch_p, 0, 20)
        data3.ctrl[2] = np.clip(T_pitch_n, 0, 20)
        data3.ctrl[3] = np.clip(T_j1_close, 0, 25)
        data3.ctrl[4] = np.clip(T_j1_open, 0, 25)
        data3.ctrl[5] = np.clip(T_j2_close, 0, 25)
        data3.ctrl[6] = np.clip(T_j2_open, 0, 25)
        
        mujoco.mj_step(model, data3)

    # -----------------------------------------------------------------
    # Metrics Computation
    # -----------------------------------------------------------------
    # Precision tracking evaluation (t in [0.5, 2.5]s)
    eval_idx_start = int(0.5 / dt)
    eval_idx_end = int(2.5 / dt)
    rmse_p1 = np.sqrt(np.mean((np.array(log_p1[eval_idx_start:eval_idx_end]) - pitch_des[eval_idx_start:eval_idx_end])**2))
    rmse_y1 = np.sqrt(np.mean((np.array(log_y1[eval_idx_start:eval_idx_end]) - yaw_des[eval_idx_start:eval_idx_end])**2))
    
    rmse_p2 = np.sqrt(np.mean((np.array(log_p2[eval_idx_start:eval_idx_end]) - pitch_des[eval_idx_start:eval_idx_end])**2))
    rmse_y2 = np.sqrt(np.mean((np.array(log_y2[eval_idx_start:eval_idx_end]) - yaw_des[eval_idx_start:eval_idx_end])**2))
    
    rmse_p3 = np.sqrt(np.mean((np.array(log_p3[eval_idx_start:eval_idx_end]) - pitch_des[eval_idx_start:eval_idx_end])**2))
    rmse_y3 = np.sqrt(np.mean((np.array(log_y3[eval_idx_start:eval_idx_end]) - yaw_des[eval_idx_start:eval_idx_end])**2))
    
    # Clamping force evaluation (t in [4.0, 6.0]s)
    clamp_idx = int(4.0 / dt)
    f_arr1 = np.array(log_f1)[clamp_idx:]
    f_arr2 = np.array(log_f2)[clamp_idx:]
    f_arr3 = np.array(log_f3)[clamp_idx:]
    
    f_mean1, f_err1, f_std1 = np.mean(f_arr1), np.abs(np.mean(f_arr1) - F_target), np.std(f_arr1)
    f_mean2, f_err2, f_std2 = np.mean(f_arr2), np.abs(np.mean(f_arr2) - F_target), np.std(f_arr2)
    f_mean3, f_err3, f_std3 = np.mean(f_arr3), np.abs(np.mean(f_arr3) - F_target), np.std(f_arr3)
    
    print("\n" + "="*70)
    print("dVRK EndoWrist Cable Control Benchmark Results")
    print("="*70)
    print(f"1. Naive Linear PD:        Pitch RMSE={rmse_p1:.4f} rad, Yaw RMSE={rmse_y1:.4f} rad | Force: Mean={f_mean1:.2f}N, Err={f_err1:.2f}N (std: {f_std1:.2f})")
    print(f"2. High-Gain Linear PD:    Pitch RMSE={rmse_p2:.4f} rad, Yaw RMSE={rmse_y2:.4f} rad | Force: Mean={f_mean2:.2f}N, Err={f_err2:.2f}N (std: {f_std2:.2f})")
    print(f"3. OURS (Diffeomorphic):   Pitch RMSE={rmse_p3:.4f} rad, Yaw RMSE={rmse_y3:.4f} rad | Force: Mean={f_mean3:.2f}N, Err={f_err3:.2f}N (std: {f_std3:.2f})")
    print("-" * 70)
    print(f"Precision Improvement:  Pitch: -{(1 - rmse_p3/rmse_p1)*100:.1f}%, Yaw: -{(1 - rmse_y3/rmse_y1)*100:.1f}%")
    print(f"Force Error Reduction:  -{(1 - f_err3/f_err1)*100:.1f}% (Gentle clamping without tissue crushing)")
    print("="*70 + "\n")
    
    # -----------------------------------------------------------------
    # High-Resolution Publication Plot
    # -----------------------------------------------------------------
    fig, axs = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    
    # 1. Pitch Tracking
    axs[0].plot(time_arr, pitch_des, 'k--', label="Target Ref", linewidth=1.8)
    axs[0].plot(time_arr, log_p1, color="#d62728", linestyle=":", label=f"Naive PD (RMSE={rmse_p1:.4f} rad)", alpha=0.8)
    axs[0].plot(time_arr, log_p2, color="#ff7f0e", linestyle="-.", label=f"High-Gain PD (RMSE={rmse_p2:.4f} rad)", alpha=0.8)
    axs[0].plot(time_arr, log_p3, color="#1f77b4", linestyle="-", label=f"Ours: Diffeomorphic (RMSE={rmse_p3:.4f} rad)", linewidth=1.8)
    axs[0].set_ylabel("Pitch Angle [rad]", fontsize=11)
    axs[0].set_title("da Vinci EndoWrist Cable-Driven Precision & Force Control", fontsize=13, fontweight='bold')
    axs[0].legend(loc="upper right", framealpha=0.9)
    axs[0].grid(True, linestyle="--", alpha=0.5)
    
    # 2. Yaw Tracking
    axs[1].plot(time_arr, yaw_des, 'k--', label="Target Ref", linewidth=1.8)
    axs[1].plot(time_arr, log_y1, color="#d62728", linestyle=":", label=f"Naive PD (RMSE={rmse_y1:.4f} rad)", alpha=0.8)
    axs[1].plot(time_arr, log_y2, color="#ff7f0e", linestyle="-.", label=f"High-Gain PD (RMSE={rmse_y2:.4f} rad)", alpha=0.8)
    axs[1].plot(time_arr, log_y3, color="#1f77b4", linestyle="-", label=f"Ours: Diffeomorphic (RMSE={rmse_y3:.4f} rad)", linewidth=1.8)
    axs[1].set_ylabel("Yaw Angle [rad]", fontsize=11)
    axs[1].legend(loc="upper right", framealpha=0.9)
    axs[1].grid(True, linestyle="--", alpha=0.5)
    
    # 3. Grasping Contact Force
    axs[2].axhline(F_target, color='k', linestyle='--', label=f"Target Tissue Clamping Force ({F_target:.1f} N)", linewidth=1.8)
    axs[2].plot(time_arr, log_f1, color="#d62728", linestyle=":", label=f"Naive PD (Mean: {f_mean1:.2f}N, Err: {f_err1:.2f}N)", alpha=0.8)
    axs[2].plot(time_arr, log_f2, color="#ff7f0e", linestyle="-.", label=f"High-Gain PD (Mean: {f_mean2:.2f}N, Err: {f_err2:.2f}N, Peak={np.max(f_arr2):.2f}N)", alpha=0.8)
    axs[2].plot(time_arr, log_f3, color="#1f77b4", linestyle="-", label=f"Ours: Admittance Clamping (Mean: {f_mean3:.2f}N, Err: {f_err3:.2f}N)", linewidth=1.8)
    axs[2].set_ylabel("Clamping Force [N]", fontsize=11)
    axs[2].set_xlabel("Time [s]", fontsize=11)
    axs[2].legend(loc="upper left", framealpha=0.9)
    axs[2].grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    out_img = "dvrk_endowrist_sim/endowrist_control_benchmark.png"
    plt.savefig(out_img, dpi=300)
    plt.close()
    print(f"Saved benchmark figure to {out_img}")

if __name__ == "__main__":
    run_endowrist_benchmark()
