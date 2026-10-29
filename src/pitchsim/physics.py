"""
Pitch flight physics: gravity + drag + Magnus (spin) force.

Coordinate system (same as Statcast), units in feet and seconds:
    x: horizontal, positive toward the catcher's right (first-base side)
    y: from the back tip of home plate toward the pitcher (pitch travels in -y)
    z: up from the ground

Model follows Alan Nathan's standard baseball trajectory model:
    a = g + a_drag + a_magnus
    a_drag   = -K * Cd * |v| * v
    a_magnus =  K * CL * |v|^2 * (w_hat x v_hat)
    K        = rho * A / (2 m)
    CL       = S / (0.4 + 2.32 S),  S = r * w_active / |v|   (Nathan / Sawicki fit)

What this model deliberately does NOT include yet (good next steps):
    - seam-shifted wake (movement not explained by spin)
    - spin decay during flight (small over ~0.4 s)
    - Cd depending on speed/spin
"""

from dataclasses import dataclass

import numpy as np

# ---- Ball and environment constants -------------------------------------
G = 32.174                      # ft/s^2
BALL_MASS_LB = 5.125 / 16       # 5.125 oz
BALL_CIRC_IN = 9.125            # inches
BALL_RADIUS_FT = BALL_CIRC_IN / (2 * np.pi) / 12
BALL_AREA_FT2 = np.pi * BALL_RADIUS_FT ** 2

FRONT_OF_PLATE_Y = 17 / 12      # Statcast reports plate_x/plate_z here
STATCAST_Y0 = 50.0              # Statcast's fitted initial state is at y = 50 ft


def air_density(temp_f: float = 70.0, elevation_ft: float = 0.0,
                rel_humidity: float = 0.5) -> float:
    """Approximate air density in lb/ft^3.

    Sea level, 70F -> ~0.0748. Coors Field (5,200 ft) is ~17% thinner,
    which is why breaking balls break less in Denver.

    Pressure from the standard-atmosphere barometric formula (ICAO / US Standard
    Atmosphere 1976, troposphere): p = 101325 (1 - 2.25577e-5 h)^5.25588, h in m.
    """
    temp_c = (temp_f - 32) * 5 / 9
    temp_k = temp_c + 273.15
    pressure_pa = 101325 * (1 - 2.25577e-5 * elevation_ft * 0.3048) ** 5.25588
    # saturation vapour pressure (Tetens), partial pressure of water vapour
    p_sat = 610.78 * 10 ** (7.5 * temp_c / (temp_c + 237.3))
    p_v = rel_humidity * p_sat
    p_d = pressure_pa - p_v
    rho_kg_m3 = p_d / (287.058 * temp_k) + p_v / (461.495 * temp_k)
    return rho_kg_m3 * 0.0624280   # kg/m^3 -> lb/ft^3


@dataclass
class PhysicsParams:
    cd: float = 0.33               # drag coefficient (typical 0.30-0.40 at pitch speeds)
    spin_efficiency: float = 0.9   # fraction of total spin that is "active" (transverse)
    rho: float = air_density()     # lb/ft^3

    @property
    def k(self) -> float:
        return self.rho * BALL_AREA_FT2 / (2 * BALL_MASS_LB)


def lift_coefficient(spin_factor: np.ndarray) -> np.ndarray:
    return spin_factor / (0.4 + 2.32 * spin_factor)


def spin_unit_vector(spin_axis_deg: float) -> np.ndarray:
    """Unit spin vector from Statcast's spin_axis.

    Statcast's spin_axis is the 2-D direction of spin-induced movement as seen
    from behind home plate: 180 deg = pure backspin (movement straight up),
    0 deg = pure topspin (straight down). The movement direction is
    m = (sin(theta), -cos(theta)) in (x, z). For a ball moving in -y, the
    Magnus force w x v points along m when w = (-m_z, 0, m_x).
    """
    th = np.radians(spin_axis_deg)
    mx, mz = np.sin(th), -np.cos(th)
    return np.array([-mz, 0.0, mx])


def acceleration(vel: np.ndarray, spin_rpm: float, w_hat: np.ndarray,
                 p: PhysicsParams) -> np.ndarray:
    speed = np.linalg.norm(vel)
    v_hat = vel / speed
    w_active = spin_rpm * p.spin_efficiency * 2 * np.pi / 60   # rad/s
    s = BALL_RADIUS_FT * w_active / speed
    cl = lift_coefficient(s)

    a_drag = -p.k * p.cd * speed * vel
    magnus_dir = np.cross(w_hat, v_hat)
    a_magnus = p.k * cl * speed ** 2 * magnus_dir
    return np.array([0.0, 0.0, -G]) + a_drag + a_magnus


def simulate(pos0, vel0, spin_rpm: float, spin_axis_deg: float,
             p: PhysicsParams | None = None, y_stop: float = FRONT_OF_PLATE_Y,
             dt: float = 1e-3) -> dict:
    """Integrate a pitch with RK4 until it reaches y_stop.

    Returns dict with arrays t, pos (N x 3), vel (N x 3); the final point is
    interpolated to land exactly on y_stop.
    """
    p = p or PhysicsParams()
    w_hat = spin_unit_vector(spin_axis_deg)
    pos = np.asarray(pos0, dtype=float)
    vel = np.asarray(vel0, dtype=float)
    t = 0.0
    ts, ps, vs = [t], [pos.copy()], [vel.copy()]

    def f(v):
        return acceleration(v, spin_rpm, w_hat, p)

    while pos[1] > y_stop:
        # RK4 on (pos, vel); acceleration depends only on velocity
        k1v = f(vel);              k1x = vel
        k2v = f(vel + 0.5*dt*k1v); k2x = vel + 0.5*dt*k1v
        k3v = f(vel + 0.5*dt*k2v); k3x = vel + 0.5*dt*k2v
        k4v = f(vel + dt*k3v);     k4x = vel + dt*k3v
        new_pos = pos + dt/6 * (k1x + 2*k2x + 2*k3x + k4x)
        new_vel = vel + dt/6 * (k1v + 2*k2v + 2*k3v + k4v)
        t += dt

        if new_pos[1] <= y_stop:
            frac = (pos[1] - y_stop) / (pos[1] - new_pos[1])
            new_pos = pos + frac * (new_pos - pos)
            new_vel = vel + frac * (new_vel - vel)
            t = t - dt + frac * dt
        pos, vel = new_pos, new_vel
        ts.append(t); ps.append(pos.copy()); vs.append(vel.copy())

        if t > 2.0:   # safety: something is wrong (ball not reaching plate)
            raise RuntimeError("Pitch did not reach the plate within 2 s")

    return {"t": np.array(ts), "pos": np.array(ps), "vel": np.array(vs)}


def simulate_batch(pos0, vel0, spin_rpm, spin_axis_deg, cd, spin_efficiency,
                   rho: float | np.ndarray | None = None,
                   y_stop: float = FRONT_OF_PLATE_Y, dt: float = 1e-3):
    """Vectorized version of `simulate` for many pitches at once.

    All array arguments have length N (cd / spin_efficiency / rho may be scalars).
    rho can vary per pitch: each game's ballpark and temperature set its own air
    density (Coors Field air is ~17% thinner than sea level).
    Returns (pos_at_plate (N x 3), vel_at_plate (N x 3), flight_time (N,)).
    Used for validation and for fitting Cd / spin efficiency, where we need
    thousands of simulations per optimizer step.
    """
    pos = np.array(pos0, dtype=float)
    vel = np.array(vel0, dtype=float)
    n = len(pos)
    spin = np.asarray(spin_rpm, dtype=float)
    axis = np.radians(np.asarray(spin_axis_deg, dtype=float))
    cd = np.broadcast_to(np.asarray(cd, dtype=float), (n,))
    eff = np.broadcast_to(np.asarray(spin_efficiency, dtype=float), (n,))
    rho = air_density() if rho is None else rho
    k = np.broadcast_to(np.asarray(rho, dtype=float) * BALL_AREA_FT2 / (2 * BALL_MASS_LB), (n,))

    mx, mz = np.sin(axis), -np.cos(axis)
    w_hat = np.stack([-mz, np.zeros(n), mx], axis=1)
    w_active = spin * eff * 2 * np.pi / 60
    gvec = np.array([0.0, 0.0, -G])

    def f(v, m):
        # m: boolean mask of pitches still in flight
        speed = np.linalg.norm(v, axis=1, keepdims=True)
        s = BALL_RADIUS_FT * w_active[m][:, None] / speed
        cl = lift_coefficient(s)
        v_hat = v / speed
        return (gvec - k[m][:, None] * cd[m][:, None] * speed * v
                + k[m][:, None] * cl * speed ** 2 * np.cross(w_hat[m], v_hat))

    out_pos = np.full((n, 3), np.nan)
    out_vel = np.full((n, 3), np.nan)
    out_t = np.full(n, np.nan)
    active = np.ones(n, dtype=bool)
    t = 0.0
    while active.any() and t < 2.0:
        p_, v_ = pos[active], vel[active]
        k1v = f(v_, active)
        k2v = f(v_ + 0.5*dt*k1v, active)
        k3v = f(v_ + 0.5*dt*k2v, active)
        k4v = f(v_ + dt*k3v, active)
        k1x, k2x, k3x, k4x = v_, v_ + 0.5*dt*k1v, v_ + 0.5*dt*k2v, v_ + dt*k3v
        new_p = p_ + dt/6 * (k1x + 2*k2x + 2*k3x + k4x)
        new_v = v_ + dt/6 * (k1v + 2*k2v + 2*k3v + k4v)

        crossed = new_p[:, 1] <= y_stop
        if crossed.any():
            idx = np.flatnonzero(active)[crossed]
            frac = ((p_[crossed, 1] - y_stop)
                    / (p_[crossed, 1] - new_p[crossed, 1]))[:, None]
            out_pos[idx] = p_[crossed] + frac * (new_p[crossed] - p_[crossed])
            out_vel[idx] = v_[crossed] + frac * (new_v[crossed] - v_[crossed])
            out_t[idx] = t + frac[:, 0] * dt
        pos[active], vel[active] = new_p, new_v
        active[np.flatnonzero(active)[crossed]] = False
        t += dt
    return out_pos, out_vel, out_t
