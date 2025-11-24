# jaxgdsp_lexicon_plate
```python
# jaxgdsp_lexicon_plate.py
#
# Lexicon-style plate reverb in pure functional JAX / GDSP style.
#
# Structure:
#   Input -> Pre-delay -> 4x Allpass Diffusion -> 4+4 Parallel Comb Tank
#         -> Damping, Modulation, Stereo Width, Wet/Dry -> Stereo Output
#
# - Pure functional JAX
# - State is a tuple of arrays/scalars
# - No classes, no dataclasses, no dicts
# - Fully differentiable
# - All shapes are fixed in init (no dynamic allocation in jit)
# - tick(): (y, new_state)
# - process(): lax.scan over tick()
#
# This module is meant as a conceptual / reference implementation and
# can be adapted into the rest of the GDSP / GammaJAX codebase.

import math
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import lax


# -------------------------------------------------------------------
# Helpers: fractional delay read/write
# -------------------------------------------------------------------

def _delay_read_linear(buf: jnp.ndarray,
                       write_idx: jnp.int32,
                       delay_samples: jnp.float32) -> jnp.float32:
    """
    Fractional delay read with linear interpolation from a 1D ring buffer.

    buf: [N]
    write_idx: scalar int index of "current" write position
    delay_samples: scalar float (may be non-integer, >= 0)
    """
    n = buf.shape[0]
    n_f = jnp.float32(n)

    # Compute read position (float), wrap into [0, N)
    read_pos = jnp.float32(write_idx) - delay_samples
    read_pos_wrapped = jnp.mod(read_pos, n_f)

    i0 = jnp.floor(read_pos_wrapped).astype(jnp.int32)
    i1 = jnp.mod(i0 + 1, n)
    frac = read_pos_wrapped - jnp.float32(i0)

    x0 = buf[i0]
    x1 = buf[i1]
    return x0 + (x1 - x0) * frac


def _delay_write(buf: jnp.ndarray,
                 write_idx: jnp.int32,
                 x: jnp.float32) -> Tuple[jnp.ndarray, jnp.int32]:
    """
    Write a single sample x into ring buffer at write_idx.
    """
    update_slice = jnp.array([x], dtype=buf.dtype)
    new_buf = lax.dynamic_update_slice(buf, update_slice, (write_idx,))
    n = buf.shape[0]
    new_idx = jnp.mod(write_idx + 1, n)
    return new_buf, new_idx


# -------------------------------------------------------------------
# Allpass filter (single stage)
# -------------------------------------------------------------------

def _allpass_tick(x: jnp.float32,
                  buf: jnp.ndarray,
                  idx: jnp.int32,
                  delay_samples: jnp.float32,
                  g: jnp.float32) -> Tuple[jnp.float32, jnp.ndarray, jnp.int32]:
    """
    Single allpass stage with fractional delay and gain g.

    Equations:
      d[n] = delay line
      d_out = d[n - M]
      s = x + g * d_out
      y = -g * s + d_out
      d[n] = s
    """
    d_out = _delay_read_linear(buf, idx, delay_samples)
    s = x + g * d_out
    y = -g * s + d_out
    new_buf, new_idx = _delay_write(buf, idx, s)
    return y, new_buf, new_idx


# -------------------------------------------------------------------
# Comb filter with in-loop damping and fractional delay
# -------------------------------------------------------------------

def _comb_damped_tick(x: jnp.float32,
                      buf: jnp.ndarray,
                      idx: jnp.int32,
                      z_lp: jnp.float32,
                      delay_samples: jnp.float32,
                      g: jnp.float32,
                      a_lp: jnp.float32) -> Tuple[jnp.float32, jnp.ndarray, jnp.int32, jnp.float32]:
    """
    Schroeder/Moorer comb with in-loop one-pole lowpass.

    d_out = delay(buf, idx, D_eff)
    z_lp_new = (1 - a_lp) * d_out + a_lp * z_lp
    fb = x + g * z_lp_new
    write fb into delay
    output y = d_out
    """
    d_out = _delay_read_linear(buf, idx, delay_samples)
    z_new = (1.0 - a_lp) * d_out + a_lp * z_lp
    fb = x + g * z_new
    new_buf, new_idx = _delay_write(buf, idx, fb)
    return d_out, new_buf, new_idx, z_new


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------

def gdsp_lexicon_plate_init(sample_rate: float,
                            max_pre_delay_ms: float = 80.0,
                            max_t60_s: float = 30.0,
                            pre_delay_ms: float = 20.0,
                            t60_s: float = 2.0,
                            damp_hz: float = 6000.0,
                            width: float = 1.0,
                            mix: float = 0.5,
                            mod_depth_ms: float = 2.0,
                            mod_rate_hz: float = 0.3,
                            param_smooth_time_s: float = 0.05) -> Tuple[tuple, jnp.ndarray]:
    """
    Initialize the Lexicon-style plate reverb.

    Returns:
      state: tuple of arrays/scalars
      params: 1D array of control parameters in the order:
          [pre_delay_ms, t60_s, damp_hz, width, mix, mod_depth_ms, mod_rate_hz]
    """
    fs = float(sample_rate)

    # Fixed structure: 4 allpasses, 4 combs per channel
    # Delay times chosen in milliseconds, then converted to samples.
    # (These are not exact Lexicon values, just musically reasonable.)
    ap_delays_ms = [12.0, 17.0, 7.0, 11.0]      # early diffusion
    ap_gains = [0.7, 0.7, 0.7, 0.7]

    combL_delays_ms = [40.0, 53.0, 61.0, 73.0]
    combR_delays_ms = [43.0, 47.0, 59.0, 71.0]

    def _ms_to_samples(ms: float) -> int:
        return int(round(ms * 0.001 * fs))

    # Pre-delay buffer length
    max_pre_delay_samples = _ms_to_samples(max_pre_delay_ms)
    if max_pre_delay_samples < 1:
        max_pre_delay_samples = 1

    # Allpass buffer lengths (slightly longer than needed)
    ap_delays_samples = jnp.array([_ms_to_samples(d) for d in ap_delays_ms], dtype=jnp.float32)
    ap_buf_lengths = [max(2, int(ds) + 4) for ds in ap_delays_samples.tolist()]

    # Comb buffer lengths (slightly longer than needed)
    combL_delays_samples = jnp.array([_ms_to_samples(d) for d in combL_delays_ms], dtype=jnp.float32)
    combR_delays_samples = jnp.array([_ms_to_samples(d) for d in combR_delays_ms], dtype=jnp.float32)
    combL_buf_lengths = [max(2, int(ds) + 16) for ds in combL_delays_samples.tolist()]
    combR_buf_lengths = [max(2, int(ds) + 16) for ds in combR_delays_samples.tolist()]

    # Allocate buffers as JAX arrays (created outside jit)
    pre_buf = jnp.zeros(max_pre_delay_samples, dtype=jnp.float32)
    pre_idx = jnp.int32(0)

    ap1_buf = jnp.zeros(ap_buf_lengths[0], dtype=jnp.float32)
    ap2_buf = jnp.zeros(ap_buf_lengths[1], dtype=jnp.float32)
    ap3_buf = jnp.zeros(ap_buf_lengths[2], dtype=jnp.float32)
    ap4_buf = jnp.zeros(ap_buf_lengths[3], dtype=jnp.float32)
    ap1_idx = jnp.int32(0)
    ap2_idx = jnp.int32(0)
    ap3_idx = jnp.int32(0)
    ap4_idx = jnp.int32(0)

    def _init_comb(buf_len: int):
        return jnp.zeros(buf_len, dtype=jnp.float32), jnp.int32(0), jnp.float32(0.0)

    combL1_buf, combL1_idx, combL1_lp = _init_comb(combL_buf_lengths[0])
    combL2_buf, combL2_idx, combL2_lp = _init_comb(combL_buf_lengths[1])
    combL3_buf, combL3_idx, combL3_lp = _init_comb(combL_buf_lengths[2])
    combL4_buf, combL4_idx, combL4_lp = _init_comb(combL_buf_lengths[3])

    combR1_buf, combR1_idx, combR1_lp = _init_comb(combR_buf_lengths[0])
    combR2_buf, combR2_idx, combR2_lp = _init_comb(combR_buf_lengths[1])
    combR3_buf, combR3_idx, combR3_lp = _init_comb(combR_buf_lengths[2])
    combR4_buf, combR4_idx, combR4_lp = _init_comb(combR_buf_lengths[3])

    # Modulation phases
    mod_phase_L = jnp.float32(0.0)
    mod_phase_R = jnp.float32(math.pi * 0.5)  # offset for decorrelation

    # Parameter smoothing coefficient
    # p_s[n+1] = s * p_s[n] + (1 - s) * p[n]
    # s ~ exp(-1 / (tau * fs))
    tau = float(param_smooth_time_s)
    if tau <= 0.0:
        smooth_coeff = jnp.float32(0.0)
    else:
        smooth_coeff = jnp.float32(math.exp(-1.0 / (tau * fs)))

    # Store base values for delays & gains as arrays
    ap_delays_samples_arr = ap_delays_samples  # (4,)
    ap_gains_arr = jnp.array(ap_gains, dtype=jnp.float32)
    combL_delays_samples_arr = combL_delays_samples
    combR_delays_samples_arr = combR_delays_samples

    # Parameter vector, order:
    # [pre_delay_ms, t60_s, damp_hz, width, mix, mod_depth_ms, mod_rate_hz]
    params = jnp.array([
        pre_delay_ms,
        t60_s,
        damp_hz,
        width,
        mix,
        mod_depth_ms,
        mod_rate_hz,
    ], dtype=jnp.float32)

    # Initial smoothed params = params
    param_smooth = params

    state = (
        jnp.float32(fs),          # state_fs
        param_smooth,             # state_param_smooth

        pre_buf, pre_idx,         # pre-delay

        ap1_buf, ap1_idx,         # AP1
        ap2_buf, ap2_idx,         # AP2
        ap3_buf, ap3_idx,         # AP3
        ap4_buf, ap4_idx,         # AP4

        combL1_buf, combL1_idx, combL1_lp,
        combL2_buf, combL2_idx, combL2_lp,
        combL3_buf, combL3_idx, combL3_lp,
        combL4_buf, combL4_idx, combL4_lp,

        combR1_buf, combR1_idx, combR1_lp,
        combR2_buf, combR2_idx, combR2_lp,
        combR3_buf, combR3_idx, combR3_lp,
        combR4_buf, combR4_idx, combR4_lp,

        mod_phase_L,
        mod_phase_R,

        ap_delays_samples_arr,
        ap_gains_arr,
        combL_delays_samples_arr,
        combR_delays_samples_arr,
        smooth_coeff,
    )

    return state, params


# -------------------------------------------------------------------
# Update state / derive audio coefficients from control params
# -------------------------------------------------------------------

def gdsp_lexicon_plate_update_state(state: tuple,
                                    params: jnp.ndarray) -> Tuple[tuple, tuple]:
    """
    Update smoothed control parameters and derive audio-rate coefficients.

    Inputs:
      state: current state tuple
      params: control parameter vector [pre_delay_ms, t60_s, damp_hz,
                                        width, mix, mod_depth_ms, mod_rate_hz]

    Returns:
      new_state: updated state with smoothed params
      audio_params: tuple of derived parameters for tick(), including:
        (pre_delay_samples,
         comb_gains_L, comb_gains_R,
         damping_coeff,
         width,
         mix,
         mod_depth_samples,
         mod_rate_rad_per_sample_L,
         mod_rate_rad_per_sample_R,
         ap_delays_samples,
         ap_gains,
         combL_delays_samples,
         combR_delays_samples)
    """
    (
        state_fs,
        state_param_smooth,

        pre_buf, pre_idx,

        ap1_buf, ap1_idx,
        ap2_buf, ap2_idx,
        ap3_buf, ap3_idx,
        ap4_buf, ap4_idx,

        combL1_buf, combL1_idx, combL1_lp,
        combL2_buf, combL2_idx, combL2_lp,
        combL3_buf, combL3_idx, combL3_lp,
        combL4_buf, combL4_idx, combL4_lp,

        combR1_buf, combR1_idx, combR1_lp,
        combR2_buf, combR2_idx, combR2_lp,
        combR3_buf, combR3_idx, combR3_lp,
        combR4_buf, combR4_idx, combR4_lp,

        mod_phase_L,
        mod_phase_R,

        ap_delays_samples_arr,
        ap_gains_arr,
        combL_delays_samples_arr,
        combR_delays_samples_arr,
        smooth_coeff,
    ) = state

    fs = state_fs

    # Parameter smoothing
    s = smooth_coeff
    params_smooth = s * state_param_smooth + (1.0 - s) * params

    pre_delay_ms = params_smooth[0]
    t60_s = params_smooth[1]
    damp_hz = params_smooth[2]
    width = params_smooth[3]
    mix = params_smooth[4]
    mod_depth_ms = params_smooth[5]
    mod_rate_hz = params_smooth[6]

    # Derived sample-based values
    pre_delay_samples = (pre_delay_ms * 0.001 * fs)

    # Comb gains from T60 and comb delays
    # g = 10^(-3 * D / (T60 * fs))
    eps_t60 = jnp.float32(1e-4)
    t60_safe = jnp.maximum(t60_s, eps_t60)
    D_L = combL_delays_samples_arr
    D_R = combR_delays_samples_arr
    comb_gains_L = jnp.power(10.0, -3.0 * D_L / (t60_safe * fs))
    comb_gains_R = jnp.power(10.0, -3.0 * D_R / (t60_safe * fs))

    # Damping coefficient (1-pole lowpass)
    # a = exp(-2*pi * f_c / fs)
    damp_hz_clamped = jnp.clip(damp_hz, 20.0, fs * 0.49)
    damping_coeff = jnp.exp(-2.0 * jnp.pi * damp_hz_clamped / fs)

    # Modulation
    mod_depth_samples = mod_depth_ms * 0.001 * fs
    mod_rate_rad = 2.0 * jnp.pi * mod_rate_hz / fs

    # Two independent rates (slightly detuned) for L and R for extra motion
    mod_rate_rad_L = mod_rate_rad
    mod_rate_rad_R = mod_rate_rad * 1.1111

    audio_params = (
        pre_delay_samples,
        comb_gains_L,
        comb_gains_R,
        damping_coeff,
        width,
        mix,
        mod_depth_samples,
        mod_rate_rad_L,
        mod_rate_rad_R,
        ap_delays_samples_arr,
        ap_gains_arr,
        D_L,
        D_R,
    )

    new_state = (
        state_fs,
        params_smooth,

        pre_buf, pre_idx,

        ap1_buf, ap1_idx,
        ap2_buf, ap2_idx,
        ap3_buf, ap3_idx,
        ap4_buf, ap4_idx,

        combL1_buf, combL1_idx, combL1_lp,
        combL2_buf, combL2_idx, combL2_lp,
        combL3_buf, combL3_idx, combL3_lp,
        combL4_buf, combL4_idx, combL4_lp,

        combR1_buf, combR1_idx, combR1_lp,
        combR2_buf, combR2_idx, combR2_lp,
        combR3_buf, combR3_idx, combR3_lp,
        combR4_buf, combR4_idx, combR4_lp,

        mod_phase_L,
        mod_phase_R,

        ap_delays_samples_arr,
        ap_gains_arr,
        combL_delays_samples_arr,
        combR_delays_samples_arr,
        smooth_coeff,
    )

    return new_state, audio_params


# -------------------------------------------------------------------
# Single-sample tick
# -------------------------------------------------------------------

def gdsp_lexicon_plate_tick(x: jnp.float32,
                            state: tuple,
                            audio_params: tuple) -> Tuple[jnp.ndarray, tuple]:
    """
    One-sample tick of the Lexicon-style plate.

    x: scalar mono input
    state: current state
    audio_params: derived parameters from update_state()
    """
    (
        state_fs,
        state_param_smooth,

        pre_buf, pre_idx,

        ap1_buf, ap1_idx,
        ap2_buf, ap2_idx,
        ap3_buf, ap3_idx,
        ap4_buf, ap4_idx,

        combL1_buf, combL1_idx, combL1_lp,
        combL2_buf, combL2_idx, combL2_lp,
        combL3_buf, combL3_idx, combL3_lp,
        combL4_buf, combL4_idx, combL4_lp,

        combR1_buf, combR1_idx, combR1_lp,
        combR2_buf, combR2_idx, combR2_lp,
        combR3_buf, combR3_idx, combR3_lp,
        combR4_buf, combR4_idx, combR4_lp,

        mod_phase_L,
        mod_phase_R,

        ap_delays_samples_arr,
        ap_gains_arr,
        combL_delays_samples_arr,
        combR_delays_samples_arr,
        smooth_coeff,
    ) = state

    (
        pre_delay_samples,
        comb_gains_L,
        comb_gains_R,
        damping_coeff,
        width,
        mix,
        mod_depth_samples,
        mod_rate_rad_L,
        mod_rate_rad_R,
        ap_delays_samples_arr_in,
        ap_gains_arr_in,
        D_L,
        D_R,
    ) = audio_params

    # Ensure local consistency with stored arrays (they should be identical)
    ap_delays = ap_delays_samples_arr_in
    ap_gains = ap_gains_arr_in

    # ----------------------------------------------------------------
    # Pre-delay
    # ----------------------------------------------------------------
    x_pd = _delay_read_linear(pre_buf, pre_idx, pre_delay_samples)
    pre_buf, pre_idx = _delay_write(pre_buf, pre_idx, x)

    # ----------------------------------------------------------------
    # Early diffusion: 4 allpass chain
    # ----------------------------------------------------------------
    u = x_pd

    # AP1
    u, ap1_buf, ap1_idx = _allpass_tick(
        u, ap1_buf, ap1_idx, ap_delays[0], ap_gains[0]
    )
    # AP2
    u, ap2_buf, ap2_idx = _allpass_tick(
        u, ap2_buf, ap2_idx, ap_delays[1], ap_gains[1]
    )
    # AP3
    u, ap3_buf, ap3_idx = _allpass_tick(
        u, ap3_buf, ap3_idx, ap_delays[2], ap_gains[2]
    )
    # AP4
    u, ap4_buf, ap4_idx = _allpass_tick(
        u, ap4_buf, ap4_idx, ap_delays[3], ap_gains[3]
    )

    u_diff = u

    # ----------------------------------------------------------------
    # Modulation LFO phases for L & R
    # ----------------------------------------------------------------
    # LFO step
    two_pi = jnp.float32(2.0 * math.pi)
    phase_L = jnp.mod(mod_phase_L + mod_rate_rad_L, two_pi)
    phase_R = jnp.mod(mod_phase_R + mod_rate_rad_R, two_pi)

    # LFO values
    m_L = mod_depth_samples * jnp.sin(phase_L)
    m_R = mod_depth_samples * jnp.sin(phase_R)

    # ----------------------------------------------------------------
    # Comb filters: 4 per channel in parallel
    # ----------------------------------------------------------------
    a_lp = damping_coeff

    # Left combs
    yL1, combL1_buf, combL1_idx, combL1_lp = _comb_damped_tick(
        u_diff,
        combL1_buf,
        combL1_idx,
        combL1_lp,
        D_L[0] + m_L,
        comb_gains_L[0],
        a_lp,
    )
    yL2, combL2_buf, combL2_idx, combL2_lp = _comb_damped_tick(
        u_diff,
        combL2_buf,
        combL2_idx,
        combL2_lp,
        D_L[1] + m_L * 0.7,
        comb_gains_L[1],
        a_lp,
    )
    yL3, combL3_buf, combL3_idx, combL3_lp = _comb_damped_tick(
        u_diff,
        combL3_buf,
        combL3_idx,
        combL3_lp,
        D_L[2] - m_L * 0.5,
        comb_gains_L[2],
        a_lp,
    )
    yL4, combL4_buf, combL4_idx, combL4_lp = _comb_damped_tick(
        u_diff,
        combL4_buf,
        combL4_idx,
        combL4_lp,
        D_L[3] + m_L * 0.3,
        comb_gains_L[3],
        a_lp,
    )

    yL = (yL1 + yL2 + yL3 + yL4) * 0.25

    # Right combs
    yR1, combR1_buf, combR1_idx, combR1_lp = _comb_damped_tick(
        u_diff,
        combR1_buf,
        combR1_idx,
        combR1_lp,
        D_R[0] + m_R,
        comb_gains_R[0],
        a_lp,
    )
    yR2, combR2_buf, combR2_idx, combR2_lp = _comb_damped_tick(
        u_diff,
        combR2_buf,
        combR2_idx,
        combR2_lp,
        D_R[1] + m_R * 0.6,
        comb_gains_R[1],
        a_lp,
    )
    yR3, combR3_buf, combR3_idx, combR3_lp = _comb_damped_tick(
        u_diff,
        combR3_buf,
        combR3_idx,
        combR3_lp,
        D_R[2] - m_R * 0.4,
        comb_gains_R[2],
        a_lp,
    )
    yR4, combR4_buf, combR4_idx, combR4_lp = _comb_damped_tick(
        u_diff,
        combR4_buf,
        combR4_idx,
        combR4_lp,
        D_R[3] + m_R * 0.2,
        comb_gains_R[3],
        a_lp,
    )

    yR = (yR1 + yR2 + yR3 + yR4) * 0.25

    # ----------------------------------------------------------------
    # Stereo width (mid/side)
    # ----------------------------------------------------------------
    M = 0.5 * (yL + yR)
    S = 0.5 * (yL - yR)
    Sw = width * S

    yL_w = M + Sw
    yR_w = M - Sw

    # ----------------------------------------------------------------
    # Wet/dry mix
    # ----------------------------------------------------------------
    yL_out = mix * yL_w + (1.0 - mix) * x
    yR_out = mix * yR_w + (1.0 - mix) * x

    y = jnp.stack([yL_out, yR_out], axis=0)

    # ----------------------------------------------------------------
    # Pack new state
    # ----------------------------------------------------------------
    new_state = (
        state_fs,
        state_param_smooth,

        pre_buf, pre_idx,

        ap1_buf, ap1_idx,
        ap2_buf, ap2_idx,
        ap3_buf, ap3_idx,
        ap4_buf, ap4_idx,

        combL1_buf, combL1_idx, combL1_lp,
        combL2_buf, combL2_idx, combL2_lp,
        combL3_buf, combL3_idx, combL3_lp,
        combL4_buf, combL4_idx, combL4_lp,

        combR1_buf, combR1_idx, combR1_lp,
        combR2_buf, combR2_idx, combR2_lp,
        combR3_buf, combR3_idx, combR3_lp,
        combR4_buf, combR4_idx, combR4_lp,

        phase_L,
        phase_R,

        ap_delays_samples_arr,
        ap_gains_arr,
        combL_delays_samples_arr,
        combR_delays_samples_arr,
        smooth_coeff,
    )

    return y, new_state


# -------------------------------------------------------------------
# Block processing via lax.scan
# -------------------------------------------------------------------

def gdsp_lexicon_plate_process(x: jnp.ndarray,
                               state: tuple,
                               audio_params: tuple) -> Tuple[jnp.ndarray, tuple]:
    """
    Process a block of samples.

    x: [T] mono input
    state: state tuple
    audio_params: derived parameters from update_state()

    Returns:
      y: [T, 2] stereo output
      new_state: updated state
    """

    def _scan_fn(carry, x_t):
        st = carry
        y_t, st_new = gdsp_lexicon_plate_tick(x_t, st, audio_params)
        return st_new, y_t

    new_state, y = lax.scan(_scan_fn, state, x)
    return y, new_state


if __name__ == "__main__":
    import numpy as np
    import sounddevice as sd
    import soundfile as sf
    import matplotlib.pyplot as plt

    
    # -----------------------------
    # Load input.wav (any channels)
    # -----------------------------
    input_audio, fs = sf.read("input.wav")

    
    # Convert to float32
    # Convert input to float32
    input_audio = input_audio.astype(np.float32)

    # ----------------------------------------
    # Add extra duration to allow full reverb tail
    # ----------------------------------------
    extra_tail_seconds = 1.0      # <-- change this number
    extra_samples = int(extra_tail_seconds * fs)

    # Append silence
    input_audio = np.concatenate(
        [input_audio, np.zeros(extra_samples, dtype=np.float32)],
        axis=0
    )


    # Force mono
    if input_audio.ndim == 2:
        x_mono = np.mean(input_audio, axis=1)
    else:
        x_mono = input_audio

    x = jnp.array(x_mono)

    # -----------------------------
    # Init reverb
    # -----------------------------
    state, params = gdsp_lexicon_plate_init(
        sample_rate=fs,
        pre_delay_ms=25.0,
        t60_s=2.2,
        damp_hz=5000.0,
        width=1.0,
        mix=0.6,
        mod_depth_ms=3.0,
        mod_rate_hz=0.35,
    )

    # Derive audio params
    state, audio_params = gdsp_lexicon_plate_update_state(state, params)

    # JIT process
    process_jit = jax.jit(gdsp_lexicon_plate_process)

    # -----------------------------
    # Process audio
    # -----------------------------
    print("Processing input.wav through Lexicon plate...")
    y, state = process_jit(x, state, audio_params)
    y_np = np.array(y)  # [T, 2]

    print("Done. Output shape:", y_np.shape)

    # -----------------------------
    # Listen
    # -----------------------------
    print("Playing output...")
    sd.play(y_np, int(fs))
    sd.wait()
    print("Playback finished.")

    # -----------------------------
    # Write output.wav (optional)
    # -----------------------------
    sf.write("output.wav", y_np, int(fs))
    print("Wrote output.wav")

    # -----------------------------
    # Plot L/R channels (optional)
    # -----------------------------
    t = np.arange(y_np.shape[0]) / fs
    plt.figure(figsize=(10,5))
    plt.plot(t, y_np[:,0], label="Left")
    plt.plot(t, y_np[:,1], label="Right", alpha=0.7)
    plt.legend()
    plt.grid(True)
    plt.title("Reverb Output Waveform")
    plt.xlabel("Time (s)")
    plt.show()
    exit()

# -------------------------------------------------------------------
# Smoke test / simple example
# -------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    try:
        import sounddevice as sd
    except ImportError:
        sd = None

    fs = 48000.0
    dur_s = 2.0
    num_samples = int(dur_s * fs)

    # Impulse input
    x_np = np.zeros(num_samples, dtype=np.float32)
    x_np[0] = 1.0
    x = jnp.array(x_np)

    # Init
    state, params = gdsp_lexicon_plate_init(
        sample_rate=fs,
        pre_delay_ms=25.0,
        t60_s=2.5,
        damp_hz=5000.0,
        width=1.0,
        mix=1.0,
        mod_depth_ms=3.0,
        mod_rate_hz=0.4,
    )

    # Update state / derive audio params
    state, audio_params = gdsp_lexicon_plate_update_state(state, params)

    # JIT the process function for speed
    process_jit = jax.jit(gdsp_lexicon_plate_process)

    # Process
    y, state = process_jit(x, state, audio_params)

    y_np = np.array(y)  # [T, 2]

    print("Output shape:", y_np.shape)
    print("L tail max:", y_np[:, 0].max(), "R tail max:", y_np[:, 1].max())

    # Plot the stereo impulse responses
    t = np.arange(num_samples) / fs
    plt.figure(figsize=(10, 6))
    plt.plot(t, y_np[:, 0], label="Left")
    plt.plot(t, y_np[:, 1], label="Right", alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("gdsp_lexicon_plate – Stereo Impulse Response")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Listen if sounddevice is available
    if sd is not None:
        print("Playing stereo reverb tail...")
        sd.play(y_np, int(fs))
        sd.wait()
        print("Done.")


```
