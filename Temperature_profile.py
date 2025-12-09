import cantera as ct
import numpy as np
import matplotlib.pyplot as plt 
import os 

# --- 1. DEFINE PHYSICAL INPUTS (SI Units and Practical Units) ---
# Burner/Domain Inputs
A_burner = 0.00113       # Burner surface area (m^2)
max_distance = 0.011     # Maximum output distance (11 mm in m)
p_lab = 101300           # Lab atmospheric pressure (1013 hPa = 101300 Pa)

# NEW BURNER CONDITION
T_burner_plate = 50.0 + 273.15 # Fixed Burner Plate Temperature (323.15 K)

# Mixture/Flow Inputs (Q_tot=10 L/min, using converted fuel flow)
fuel_name = 'CH4'        
T_in = T_burner_plate    # Inlet temperature (323.15 K)
phi = 1.0                # Equivalence Ratio (Stoichiometric)

# Input Unit: L/min (Normal Conditions: 0°C and 1 atm)
V_fuel_norm_Lpm = 0.95   # Fuel volumetric flow rate at NC (using 0.95 L/min from your run)

# --- CORRECTION: USE .h5 FORMAT ---
initial_guess_file = 'initial_stable_flame.h5' # Changed extension from .xml to .h5

# --- 2. INITIALIZE GAS AND MIXTURE AT NC (for mdot calculation) ---
gas_norm = ct.Solution('ffcm-1.yaml') 
air = "O2:0.21,N2:0.79"
gas_norm.set_equivalence_ratio(phi=phi, fuel=f"{fuel_name}:1", oxidizer=air)
T_cn = 273.15  # K
P_cn = ct.one_atm # 1 atm
gas_norm.TP = T_cn, P_cn

# --- 3. CALCULATE MASS FLOW RATE (mdot) ---

X_fuel = gas_norm[fuel_name].X[0]
V_fuel_norm_mps = V_fuel_norm_Lpm * (1e-3 / 60.0) 
V_tot_norm_mps = V_fuel_norm_mps / X_fuel

rho_cn = gas_norm.density_mass 
mdot_target = rho_cn * V_tot_norm_mps / A_burner 

# Calculate inlet velocity at preheat temperature (T_in)
gas_inlet = ct.Solution('ffcm-1.yaml')
gas_inlet.set_equivalence_ratio(phi=phi, fuel=f"{fuel_name}:1", oxidizer=air)
gas_inlet.TP = T_in, p_lab
vel_T_preheat = mdot_target / gas_inlet.density_mass

print(f"Target Fuel Flow Rate: {V_fuel_norm_Lpm} L/min (NC)")
print(f"Target mass flow rate (mdot): {mdot_target:.6e} kg/(m^2*s)")
print(f"Inlet velocity (at {T_in} K): {vel_T_preheat:.4f} m/s")

# --- 4. FLAME SETUP: INITIALIZE GAS AT INLET CONDITIONS ---
gas = ct.Solution('ffcm-1.yaml')
gas.set_equivalence_ratio(phi=phi, fuel=f"{fuel_name}:1", oxidizer=air)
gas.TP = T_in, p_lab 

# --- 5. COMPUTATIONAL DOMAIN AND FLAME SETUP ---

width = 0.015              # m
grid = np.linspace(0, width, 61, endpoint=True)
loglevel = 1               

# Use ct.BurnerFlame
f = ct.BurnerFlame(gas, grid=grid) 
f.burner.mdot = mdot_target
f.burner.T = T_burner_plate 

# Set common solver parameters
tol_ss = [1.0e-5, 1.0e-13]
tol_ts = [1.0e-4, 1.0e-10]
f.flame.set_steady_tolerances(default=tol_ss)
f.flame.set_transient_tolerances(default=tol_ts)
f.set_refine_criteria(ratio=5.0, slope=0.25, curve=0.25, prune=0.01)
f.set_max_jac_age(ss_age=10, ts_age=10)


# --- 6. ROBUST TWO-STEP SOLVING STRATEGY ---

try:
    # --- Step 6a: Solve a low-flow rate flame (10% of target) for an initial guess ---
    mdot_stable_guess = mdot_target * 0.10 
    f.burner.mdot = mdot_stable_guess
    f.energy_enabled = False # Start simple
    f.transport_model = "mixture-averaged"
    f.set_time_step(5.0e-3, [1, 2, 4, 8]) 
    
    print(f"\n--- Step 6a: Solving Low Flow Rate Guess (mdot={mdot_stable_guess:.2e}) ---")
    f.solve(loglevel=0, auto=True, refine_grid=False) 
    
    # --- SAVING AS .h5 ---
    f.save(initial_guess_file, 'initial_guess')
    print("Initial stable solution saved.")
    
    # --- Step 6b: Solve the Target Flow Rate using the stable solution as guess ---
    f_target = ct.BurnerFlame(gas, grid=grid)
    f_target.restore(initial_guess_file, 'initial_guess') # Restoring from .h5
    
    # Set the target mass flow rate and physics
    f_target.burner.mdot = mdot_target
    f_target.burner.T = T_burner_plate
    
    f_target.energy_enabled = True
    f_target.transport_model = "multicomponent"
    f_target.soret_enabled = True
    f_target.radiation_enabled = True 
    f_target.flame.set_steady_tolerances(default=tol_ss)
    f_target.flame.set_transient_tolerances(default=tol_ts)
    f_target.set_time_step(5.0e-3, [2, 4, 8, 16, 32])
    f_target.set_refine_criteria(ratio=2.0, slope=0.025, curve=0.01, prune=0.01)
    f_target.set_max_jac_age(ss_age=20, ts_age=20)
    
    print(f"\n--- Step 6b: Solving Target Flow Rate (mdot={mdot_target:.2e}) ---")
    f_target.solve(loglevel=loglevel, auto=True, refine_grid=True) 
    print("Calculation completed successfully.")
    success = True
    f = f_target
    
except Exception as e:
    print(f"Calculation failed: {e}")
    success = False
finally:
    # Clean up the temporary file
    if os.path.exists(initial_guess_file):
        os.remove(initial_guess_file)


# --- 7. EXTRACTION, OUTPUT, AND PLOTTING ---
if success:
    z = f.grid
    T = f.T
    z_mm = z * 1000.0
    
    results = [(pos, temp) for pos, temp in zip(z_mm, T) if pos <= max_distance * 1000]

    profile_data = np.column_stack((z_mm, T))
    csv_file_name = f'temperature_profile_phi{phi:.1f}.csv'

    np.savetxt(csv_file_name, profile_data, 
               header='Distance (mm), Temperature (K)', 
               delimiter=',',
               fmt='%.6e')

    print(f"\n✅ Results successfully saved to: {csv_file_name}")
    
    print("\n--- Temperature Profile (0 to 11 mm) ---")
    print("Distance (mm) | Temperature (K)")
    print("------------------------------")
    for r in results:
        print(f"{r[0]:13.4f} | {r[1]:14.2f}")

    ### PLOTTING THE TEMPERATURE PROFILE ###
    plt.figure()
    plt.plot(z_mm, T, label=f'T Profile (RADCAL)')
    
    plt.axvline(x=max_distance * 1000, color='r', linestyle='--', linewidth=0.8, label='11 mm Limit')
    plt.axhline(y=T_burner_plate, color='b', linestyle=':', linewidth=1, label=f'T Burner ({T_burner_plate:.1f} K)')
    
    plt.xlabel('Distance from Burner Plate (mm)')
    plt.ylabel('Temperature (K)')
    plt.title(f'Burner Flame (Q_tot=10 L/min, $\phi={phi}$)')
    plt.grid(True)
    plt.legend()
    plt.show()

else:
    print("\n--- No Temperature Profile Available ---")
    print("The solver failed. Please re-check the two-step stabilization approach parameters (e.g., initial guess flow rate).")