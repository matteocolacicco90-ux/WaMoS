# ============================================
# Lock-in simulation with optional line definition
# ============================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import wofz
from scipy.signal import butter, filtfilt, find_peaks
from scipy.optimize import curve_fit

# -----------------------------
# Simulation Parameters
# -----------------------------
fs = 50000          # sampling frequency [Hz]
duration = 10.0     # duration [s]
t = np.linspace(0, duration, int(fs*duration))
n = len(t)

# -----------------------------
# Signal Source Selection
# -----------------------------
# -----------------------------
# Signal Source Selection
# -----------------------------
# Set this to True to use your file
use_csv_signal = True   
# Replace with your actual file name 
csv_file = "M1_2025-11-04_Qin10l_lmin_phi1_ch4air_Tld51stC_y10mm00001.txt" 

if use_csv_signal:
    print(f"Loading data from {csv_file}...")
    
    # 1. Load the file
    # skiprows=4: Ignora le prime 4 righe di metadati LeCroy / Ignores the first 4 metadata lines
    # sep=';': Il tuo file usa il punto e virgola come separatore / Your file uses semicolon separator
    data = pd.read_csv(csv_file, sep=',', decimal='.', skiprows=4) #for Mx files sep=',' while for Cx sep=';'

    # 2. Assign columns

    try:
        t = data["Time"].values
        signal_raw = data["Ampl"].values
    except KeyError:
        # Fallback just in case columns have spaces, e.g., " Time"
        print("Warning: Exact column names 'Time'/'Ampl' not found. Trying by index.")
        t = data.iloc[:, 0].values
        signal_raw = data.iloc[:, 1].values

    # 3. Time correction (Shift to start at 0)
    # Your data starts at -20s. This shifts it so the first sample is at t=0s.
    t = t - t[0]

    # 4. Recalculate Sampling Frequency (fs) from real data
    # Calculate the average time step (dt) to find the real sampling rate
    dt = np.mean(np.diff(t))
    fs = 1 / dt
    n = len(t)
    
    print(f"Detected sampling frequency (fs): {fs:.2f} Hz")
    print(f"Number of samples: {n}")
    print(f"Signal duration: {t[-1]:.2f} s")

    # 5. Set the base signal
    # We use the raw experimental data directly
    S_t = signal_raw

else:
    # -----------------------------
    # Synthetic Model Generation (Original code)
    # -----------------------------
    # Původní syntetická varianta
    S_t = 0.25 * t - 0.005 * t*t

# Modulation parameters
f_mod = 2345        # modulation frequency [Hz]
mod_depth = 0.05
mod_signal = mod_depth * np.sin(2*np.pi*f_mod*t)
scan = t + mod_signal

# -----------------------------
# Line Definitions
# -----------------------------
# Variant 1: Manual input
# Each item: [position, intensity, sigma, gamma]
lines_manual = [
    [3.0, 1.5e-2, 0.025, 0.03],
    [3.33, 0.5e-2, 0.0275, 0.03],
    [4.5, 1.0e-2, 0.025, 0.03],
    [4.75, 1.0e-2, 0.0275, 0.03],
    [6.0, 0.67e-2, 0.025, 0.03],
    [7.25, 0.5e-2, 0.0275, 0.03],
    [9.0, 0.33e-2, 0.025, 0.03]
]

# Variant 2: Load from HITRAN-like CSV
# CSV must have columns: position, intensity, sigma, gamma
use_hitran = False   # <- switch: True = use CSV, False = manual input
hitran_file = "hitran_lines.csv"

if use_hitran:
    hitran_data = pd.read_csv(hitran_file)
    lines = hitran_data[["position", "intensity", "sigma", "gamma"]].values.tolist()
else:
    lines = lines_manual

# -----------------------------
# Voigt Profile and Absorption
# -----------------------------
def voigt(x, sigma, gamma):
    z = (x + 1j*gamma) / (sigma*np.sqrt(2))
    return np.real(wofz(z)) / (sigma*np.sqrt(2*np.pi))

def absorption_profile(scan, lines):
    absorption = np.zeros_like(scan)
    for xc, strength, sigma, gamma in lines:
        absorption += strength * voigt(scan - xc, sigma, gamma)
    return absorption

absorption = absorption_profile(scan, lines)

# -----------------------------
# Detector Signal with Noise
# -----------------------------
noise_std = 0.001
signal = S_t * (1 - absorption) + np.random.normal(0, noise_std, n)

# -----------------------------
# Time Constant Adaptation τ
# -----------------------------
tau = 0.001  # lock-in time constant [s]
f_cut = 1/(2*np.pi*tau)   # cutoff frequency -3 dB
print(f"Time constant used τ = {tau:.3f} s → cutoff = {f_cut:.2f} Hz")

def lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, data)

# -----------------------------
# Lock-in Demodulation
# -----------------------------
# 2f
fref2f = 2*f_mod
refcos2f = np.cos(2*np.pi*fref2f*t)
refsin2f = np.sin(2*np.pi*fref2f*t)
xraw2f = signal * refcos2f
yraw2f = signal * refsin2f
xfiltered = lowpass_filter(xraw2f, f_cut, fs)
yfiltered2f = lowpass_filter(yraw2f, f_cut, fs)

# 4f
fref4f = 4*f_mod
refcos4f = np.cos(2*np.pi*fref4f*t)
refsin4f = np.sin(2*np.pi*fref4f*t)
xraw4f = signal * refcos4f
yraw4f = signal * refsin4f
xfiltered4f = lowpass_filter(xraw4f, f_cut, fs)
yfiltered4f = lowpass_filter(yraw4f, f_cut, fs)

# -----------------------------
# Normalization and Ratios
# -----------------------------
def safe_divide(num, denom, eps=1e-12):
    return num / (denom + eps)

k = 0.1
ratio = safe_divide(xfiltered, np.sqrt(xfiltered4f + k))
ratio_normalized = safe_divide(ratio, S_t)

# -----------------------------
# Evaluation Criteria
# -----------------------------
mask = (t >= 0.5) & (t <= 9.5)
t_sel = t[mask]
ratio_seln = ratio_normalized[mask]

# Set threshold (e.g., 0.002 = only significant peaks)
threshold = 0.001
min_time = 0.1  # seconds
distance_int = int(min_time * fs)  # converted to samples

# Find local maxima above threshold
peaks_max, props_max = find_peaks(ratio_seln, distance=distance_int, height=threshold)
peak_vals_max = ratio_seln[peaks_max]

# Find local minima below -threshold (i.e., deep enough negative values)
peaks_min, props_min = find_peaks(-ratio_seln, distance=5000, height=threshold)
peak_vals_min = ratio_seln[peaks_min]

# Print values
print("Local Maxima:")
for i, idx in enumerate(peaks_max, 1):
    print(f"{i}: t = {t_sel[idx]:.3f}, value = {ratio_seln[idx]:.3f}")

print("Local Minima:")
for i, idx in enumerate(peaks_min, 1):
    print(f"{i}: t = {t_sel[idx]:.3f}, value = {ratio_seln[idx]:.3f}")

# Visualization
plt.figure(figsize=(12,6))
plt.plot(t_sel, ratio_seln, color='grey', label='Normalized Ratio')

# maxima in red
plt.plot(t_sel[peaks_max], ratio_seln[peaks_max], 'ro', label='Local Maxima')
for i, idx in enumerate(peaks_max, 1):
    plt.text(t_sel[idx], ratio_seln[idx], f'{i}', color='red', fontsize=10)

# minima in blue
plt.plot(t_sel[peaks_min], ratio_seln[peaks_min], 'bo', label='Local Minima')
for i, idx in enumerate(peaks_min, 1):
    plt.text(t_sel[idx], ratio_seln[idx], f'-{i}', color='blue', fontsize=10)

plt.xlabel("time [s]")
plt.ylabel("normalized signal")
plt.title("Normalized signal with local extrema")
plt.legend()
plt.show()

# ============================================
# Theoretical Absorption Line Profiles (constant cell length)
# ============================================
scan_theory = t  # monotonic axis (no modulation)
cell_length = 1.0   # constant cell length
scale = 0.005       # absorption scaling factor

def single_line_absorption(scan_axis, xc, strength, sigma, gamma, cell_length=1.0, scale=0.005):
    return (strength * cell_length) * voigt(scan_axis - xc, sigma, gamma) * scale

fig, ax = plt.subplots(len(lines), 1, figsize=(12, 2.6*len(lines)), sharex=True)
peak_summary = []

for i, (xc, strength, sigma, gamma) in enumerate(lines):
    prof = single_line_absorption(scan_theory, xc, strength, sigma, gamma,
                                  cell_length=cell_length, scale=scale)
    ax[i].plot(scan_theory, prof, color='tab:blue', lw=1.8, label='Theoretical Profile')

    # Find profile maximum around xc
    win = 0.8
    m = (scan_theory >= xc - win/2) & (scan_theory <= xc + win/2)
    prof_win = prof[m]
    scan_win = scan_theory[m]

    if prof_win.size > 0:
        idx_peak = np.argmax(prof_win)
        peak_pos = scan_win[idx_peak]
        peak_val = prof_win[idx_peak]
        ax[i].plot(peak_pos, peak_val, 'ro')
        peak_summary.append({
            "line_center": xc,
            "peak_pos": peak_pos,
            "peak_value": peak_val
        })

    ax[i].axvline(x=xc, color='black', linestyle='--', alpha=0.5)
    ax[i].set_ylabel("Absorption")
    ax[i].set_title(f"Line at {xc:.3f} s (S={strength:.3f})")
    ax[i].legend()

ax[-1].set_xlabel("time/scan axis")
plt.tight_layout()
plt.show()

# Print relative intensities (max values)
print("Relative peak intensities (max values):")
for rec in peak_summary:
    print(f"Line at {rec['line_center']:.3f}s → maximum {rec['peak_value']:.5f}")

# --- Calculation of experimental line amplitudes (2 maxima + 1 minimum) ---
amplitudes = []

# Sort indices by time
all_max = sorted(peaks_max)
all_min = sorted(peaks_min)

for i in range(len(all_max)-1):
    left_max = all_max[i]
    right_max = all_max[i+1]

    # Find minima between these two maxima
    local_minima = [m for m in all_min if left_max < m < right_max]
    if len(local_minima) == 0:
        continue  # no minimum between max → skip

    # Take the deepest minimum between maxima
    min_idx = local_minima[np.argmin(ratio_seln[local_minima])]

    # Values
    val_left = ratio_seln[left_max]
    val_right = ratio_seln[right_max]
    val_min = ratio_seln[min_idx]

    # Amplitude = average of both maxima - minimum
    amplitude = ( (val_left + val_right)/2 ) - val_min

    amplitudes.append({
        "t_left_max": t_sel[left_max],
        "t_right_max": t_sel[right_max],
        "t_min": t_sel[min_idx],
        "amplitude": amplitude
    })

# Function: maximum Voigt profile value for a given line
def voigt_peak_value(xc, sigma, gamma, strength=1.0, scale=0.005):
    # Calculate profile around xc
    x_local = np.linspace(xc - 0.5, xc + 0.5, 2000)
    profile = strength * voigt(x_local - xc, sigma, gamma) * scale
    return np.max(profile)

# -----------------------------
# Result Output – Amplitudes vs. Voigt Peak Values
# -----------------------------
results = []
print("Comparison of amplitudes and FWHM (referenced to Voigt peak):")

for amp in amplitudes:
    t_min = amp["t_min"]
    nearest_line = min(lines, key=lambda l: abs(l[0] - t_min))
    line_pos, line_strength, sigma, gamma = nearest_line

    # Reference value = maximum of Voigt profile
    voigt_peak = voigt_peak_value(line_pos, sigma, gamma, strength=line_strength)
    
    # Experimental and Theoretical FWHM
    fwhm_exp = amp["t_right_max"] - amp["t_left_max"]
    fwhm_theory = 0.5346*(2*gamma) + np.sqrt(0.2166*(2*gamma)**2 + (2.355*sigma)**2)
    fwhm_ratio = fwhm_exp / fwhm_theory

    # Ratio of amplitude to Voigt peak
    amp_ratio = (amp["amplitude"]) / (voigt_peak/fwhm_theory)

    results.append({
        "line_pos": line_pos,
        "voigt_peak": voigt_peak,
        "amplitude": amp["amplitude"],
        "amp_ratio": amp_ratio,
        "fwhm_exp": fwhm_exp,
        "fwhm_theory": fwhm_theory,
        "fwhm_ratio": fwhm_ratio
    })

    print(f"Line at {line_pos:.3f}s: "
          f"A = {amp['amplitude']:.4f}, "
          f"Voigt_peak = {voigt_peak:.5f}, "
          f"A_exp/Voigt/FWHM_theory = {amp_ratio:.2f}, "
          f"FWHM_exp = {fwhm_exp:.3f}, "
          f"FWHM_theory = {fwhm_theory:.3f}, "
          f"FWHM_ratio = {fwhm_ratio:.2f}")

# Print amplitude results
print("Line Amplitudes (2 maxima + 1 minimum):")
for i, amp in enumerate(amplitudes, 1):
    print(f"{i}: between {amp['t_left_max']:.3f}s and {amp['t_right_max']:.3f}s "
          f"(min at {amp['t_min']:.3f}s) → A = {amp['amplitude']:.4f}")

results = []
for amp in amplitudes:
    # Find nearest line by minimum position
    t_min = amp["t_min"]
    nearest_line = min(lines, key=lambda l: abs(l[0] - t_min))
    line_pos, line_intensity, sigma, gamma = nearest_line

    # Ratio of amplitude to intensity
    amp_ratio = amp["amplitude"] / line_intensity

    # Experimental FWHM = distance between two maxima
    fwhm_exp = amp["t_right_max"] - amp["t_left_max"]

    # Theoretical FWHM (Voigt profile approximation)
    fwhm_theory = 0.5346*(2*gamma) + np.sqrt(0.2166*(2*gamma)**2 + (2.355*sigma)**2)

    fwhm_ratio = fwhm_exp / fwhm_theory

    results.append({
        "line_pos": line_pos,
        "intensity": line_intensity,
        "amplitude": amp["amplitude"],
        "amp_ratio": amp_ratio,
        "fwhm_exp": fwhm_exp,
        "fwhm_theory": fwhm_theory,
        "fwhm_ratio": fwhm_ratio
    })

# Print Comparison Results
print("Comparison of Amplitudes and FWHM:")
for r in results:
    print(f"Line at {r['line_pos']:.3f}s: "
          f"A = {r['amplitude']:.4f}, I = {r['intensity']:.4f}, A/I = {r['amp_ratio']:.4f}, "
          f"FWHM_exp = {r['fwhm_exp']:.3f}, FWHM_theory = {r['fwhm_theory']:.3f}, "
          f"FWHM_ratio = {r['fwhm_ratio']:.2f}")

# -----------------------------
# Calculation of baseline mask dynamically based on lines
# -----------------------------
exclude_halfwidth = 0.5  # [s] – half-width of area to exclude around each line

# Start with a mask where everything is True
baseline_mask = np.ones_like(t_sel, dtype=bool)

for xc, strength, sigma, gamma in lines:
    # Exclude interval [xc - Δ, xc + Δ]
    baseline_mask &= ~((t_sel >= (xc - exclude_halfwidth)) & (t_sel <= (xc + exclude_halfwidth)))

# Calculate baseline standard deviation
baseline_std = np.std(ratio_seln[baseline_mask])
print("Baseline std:", baseline_std)

# -----------------------------
# Visualization
# -----------------------------
# --- Select interval 0.5–9.5 s ---
mask = (t >= 0.5) & (t <= 9.5)
t_sel = t[mask]
signal_sel = signal[mask]
x2f_sel = xfiltered[mask]
x4f_sel = xfiltered4f[mask]
x4f_sqrt_sel = np.sqrt(xfiltered4f[mask] + 0.1)
ratio_sel = ratio[mask]
ratio_seln = ratio_normalized[mask]

# --- Plotting ---
plt.style.use('seaborn-v0_8')
fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

# 1. Detector Signal
axs[0].plot(t_sel, signal_sel, color='gray', label='Detector Signal')
axs[0].set_ylabel("Intensity")
axs[0].set_title("Detector Signal")
axs[0].legend()

# 2. Demodulated Components
axs[1].plot(t_sel, x2f_sel, label='X2f (cos component)', color='tab:blue')
axs[1].plot(t_sel, x4f_sel, label='X4f (cos component)', color='tab:red')
axs[1].set_ylabel("Amplitude")
axs[1].set_title("Demodulated Components")
axs[1].legend()

# 3. Square Root Component X4f
axs[2].plot(t_sel, x4f_sqrt_sel, label='√X4f', color='tab:pink')
axs[2].set_ylabel("Amplitude")
axs[2].set_title("Square Root Component X4f")
axs[2].legend()

# 4. Ratios and Normalization
axs[3].plot(t_sel, ratio_sel, label='X2f / √X4f', color='tab:green')
axs[3].plot(t_sel, ratio_seln, label='X2f / √X4f (norm.)', color='tab:grey')

# If maxima/minima are calculated, plot them
if 'peaks_max' in locals():
    axs[3].plot(t_sel[peaks_max], ratio_seln[peaks_max], 'ro', label='Maxima')
if 'peaks_min' in locals():
    axs[3].plot(t_sel[peaks_min], ratio_seln[peaks_min], 'bo', label='Minima')

axs[3].set_xlabel("time [s]")
axs[3].set_ylabel("Normalized Ratio")
axs[3].set_title("Normalized Signal")
axs[3].legend()

# Vertical lines for line positions
for ax in axs:
    for xc, strength, sigma, gamma in lines:
        if 0.5 <= xc <= 9.5:
            ax.axvline(x=xc, color='black', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
