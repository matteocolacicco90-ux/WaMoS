import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. GLOBAL INPUTS (Fixed for all mechanisms) ---
# Burner/Domain
A_burner = 0.00113       # m^2
max_distance = 0.011     # m
p_lab = 101325          # Pa
width = 0.015            # m

# Conditions
T_burner_plate = 50.0 + 273.15  # 323.15 K
T_in = T_burner_plate
phi = 1.0
fuel_name = 'CH4'
oxidizer_str = "O2:0.21,N2:0.79"

# Flow Rate
V_fuel_norm_Lpm = 0.95   # L/min (Normal Conditions)

# LIST OF MECHANISMS TO COMPARE
# Ensure these files are in your Cantera data path or working directory
mechanisms_to_test = ['gri30.yaml', 'ffcm-1.yaml','KIBO.yaml']

# Dictionary to store results for plotting later
comparison_results = {}

# --- 2. SIMULATION FUNCTION ---
def run_flame_simulation(mech_file):
    print(f"\n{'='*40}")
    print(f"STARTING SIMULATION FOR: {mech_file}")
    print(f"{'='*40}")

    # Unique temporary file for this mechanism to avoid conflicts
    initial_guess_file = f'initial_guess_{mech_file.replace(".yaml", "")}.h5'

    try:
        # -- A. Initialize Gas --
        gas = ct.Solution(mech_file)
        gas.set_equivalence_ratio(phi=phi, fuel=f"{fuel_name}:1", oxidizer=oxidizer_str)
        
        # -- B. Calculate Mass Flow Rate (Specific to mechanism density) --
        # Calculate Density at Normal Conditions (0°C, 1 atm)
        gas.TP = 273.15, ct.one_atm
        rho_cn = gas.density_mass
        
        # Calculate Flow
        X_fuel = gas[fuel_name].X[0]
        V_fuel_norm_mps = V_fuel_norm_Lpm * (1e-3 / 60.0)
        V_tot_norm_mps = V_fuel_norm_mps / X_fuel
        
        mdot_target = rho_cn * V_tot_norm_mps / A_burner
        
        print(f"  > Calculated mdot: {mdot_target:.6e} kg/(m^2*s)")

        # -- C. Setup Flame Object --
        # Set inlet conditions
        gas.TP = T_in, p_lab
        
        grid = np.linspace(0, width, 61, endpoint=True)
        f = ct.BurnerFlame(gas, grid=grid)
        f.burner.mdot = mdot_target
        f.burner.T = T_burner_plate

        # Solver Settings
        tol_ss = [1.0e-5, 1.0e-13]
        tol_ts = [1.0e-4, 1.0e-10]
        f.flame.set_steady_tolerances(default=tol_ss)
        f.flame.set_transient_tolerances(default=tol_ts)
        f.set_max_jac_age(ss_age=10, ts_age=10)

        # -- D. Two-Step Solving Strategy --
        
        # Step 1: Initial Guess (Low Flow, Simple Physics)
        print("  > Step 1: Generating initial guess (10% flow)...")
        f.burner.mdot = mdot_target * 0.10
        f.energy_enabled = False
        f.transport_model = "mixture-averaged"
        f.solve(loglevel=0, auto=True, refine_grid=False)
        
        f.save(initial_guess_file, 'initial_guess')
        
        # Step 2: Target Solution (Full Physics)
        print("  > Step 2: Solving target flow with full physics...")
        # Restore to a NEW object to ensure clean settings
        f_target = ct.BurnerFlame(gas, grid=grid)
        f_target.restore(initial_guess_file, 'initial_guess')
        
        f_target.burner.mdot = mdot_target
        f_target.burner.T = T_burner_plate
        
        f_target.energy_enabled = True
        f_target.transport_model = "multicomponent"
        f_target.soret_enabled = True
        f_target.radiation_enabled = True
        
        f_target.set_refine_criteria(ratio=2.0, slope=0.025, curve=0.01, prune=0.01)
        f_target.solve(loglevel=1, auto=True, refine_grid=True)
        
        # -- E. Extract and Save CSV --
        z_mm = f_target.grid * 1000.0
        T = f_target.T
        
        # Filter for CSV output (0-11mm)
        csv_data = []
        for pos, temp in zip(z_mm, T):
            if pos <= max_distance * 1000:
                csv_data.append([pos, temp])
        
        csv_filename = f'profile_{mech_file.replace(".yaml","")}_phi{phi}.csv'
        np.savetxt(csv_filename, csv_data, delimiter=',', header='Distance(mm),Temperature(K)', fmt='%.6e')
        print(f"  > Saved: {csv_filename}")
        
        # Clean up temp file
        if os.path.exists(initial_guess_file):
            os.remove(initial_guess_file)
            
        return z_mm, T

    except Exception as e:
        print(f"  > FAILED: {mech_file} - {e}")
        return None, None

# --- 3. MAIN LOOP ---
plt.figure(figsize=(10, 6))

for mech in mechanisms_to_test:
    z, T = run_flame_simulation(mech)
    
    if z is not None:
        # Store for plotting
        comparison_results[mech] = (z, T)
        
        # Add to plot
        plt.plot(z, T, label=f'{mech} (T_max={max(T):.0f} K)')

# --- 4. FINALIZE PLOT ---
plt.axvline(x=max_distance * 1000, color='gray', linestyle='--', alpha=0.7, label='11 mm Limit')
plt.axhline(y=T_burner_plate, color='black', linestyle=':', label='Burner Surface')

plt.xlim(0, 15) # Show slightly past 11mm to see trend
plt.xlabel('Distance from Burner (mm)')
plt.ylabel('Temperature (K)')
plt.title(f'Mechanism Comparison (Q={V_fuel_norm_Lpm} L/min, $\phi={phi}$)')
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

# Save comparison plot
plt.savefig('mechanism_comparison.png', dpi=300)
plt.show()
