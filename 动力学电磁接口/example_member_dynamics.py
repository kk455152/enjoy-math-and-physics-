from em_force_model import DEFAULT_DATA, ElectromagneticForceModel, TrajectoryState


model = ElectromagneticForceModel(
    DEFAULT_DATA / "GCr15_hardened_static_force_dense.csv",
    DEFAULT_DATA / "GCr15_hardened_centerline_BH.csv",
    material="GCr15_hardened",
    N=10,
    R_mm=80.0,
    rho_mm=85.0,
    g_mm=6.0,
    D_mm=6.0,
    plate_mm=1.0,
    tau_mag_s=0.0,
)

# Replace these states with the dynamics solver's states at each time step.
states = [
    TrajectoryState(
        t_s=0.0,
        x_mm=0.0,
        y_mm=85.0,
        vx_m_s=0.0,
        vy_m_s=0.0,
        Omega_rot_rad_s=10.0,
        omega_ball_z_rad_s=0.0,
    ),
    TrajectoryState(
        t_s=0.0005,
        x_mm=0.0,
        y_mm=85.0,
        vx_m_s=0.0,
        vy_m_s=0.0,
        Omega_rot_rad_s=10.0,
        omega_ball_z_rad_s=0.0,
    ),
]

forces = model.evaluate_trajectory(states, rotor_phase0_rad=0.0)
for row in forces:
    print(row["t_s"], row["Fmagnetic_x_N"], row["Fmagnetic_y_N"], row["Fmagnetic_z_N"])
