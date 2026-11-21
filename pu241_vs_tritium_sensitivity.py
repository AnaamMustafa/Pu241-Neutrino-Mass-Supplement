# Pu-241 vs tritium: neutrino-mass sensitivity from beta-decay endpoint spectra.
#
# Figures, in the order of the paper (all saved to the working directory):
#   Fig. 1  fig1_beta_spectrum()       -> fig1_betaspectrum.png
#   Fig. 2  fig2_fresh_source()        -> fig2_fresh_source.png
#   Fig. 3  fig3_sensitivity()         -> fig3_sensitivity.png
#   Fig. 4  fig4_pumass_scan()         -> fig4_pumass_scan.png
#   Fig. 5  fig5_sensitivity_ratio()   -> fig5_sensitivity_ratio.png

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend: figures are saved to files
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize, brentq
import argparse
import matplotlib as mpl
import math


# --- Physical Constants & Setup ---
day2sec = 86400
yr2sec = 365.25 * day2sec 
# Exposure time: 3 years
run_time = 3 * yr2sec
N_A = 6.022e23
bg_bin_norm = 0.01  # relative uncertainty of the U-237 background normalization (100-bin sideband)


# Tritium parameters
E0_T = 18592  # eV
TauT = 17.74 * yr2sec
G_T = 1 / TauT
M_mol_T = 3.0   # tritium molar mass [g/mol]
mass_T = 1.0    # tritium source mass [ug]
N_T = mass_T * 1e-6 * N_A * G_T * run_time / M_mol_T  # decays in the run (1e-6 converts ug to g)
C_T = 8.6e-13


# Pu-241 parameters
E0_Pu = 21780  # eV
TauPu = 20.68 * yr2sec
TauU = 9.74 * day2sec
GPu = 1 / TauPu
N_Pu = 100e-6 * N_A * GPu * run_time / 241.  # decays in the run; first factor = source mass in grams (100 ug, similar total activity to 1 ug of tritium)
C_Pu = 3.9e-13
branch_alpha = 2.5e-5  # alpha-decay branch of Pu-241 feeding U-237
U_density = 9.1e-6     # U-237 background spectral density near the endpoint [1/eV]
# Alternative normalization (same spectral amplitude near the endpoint as tritium):
#N_Pu = C_T/C_Pu * N_T
print(N_Pu)



# Energy grid: E is the distance to the endpoint in eV (E = 0 at the endpoint)
E = np.linspace(-40, 1, 1000)
E_Pu = E0_Pu + E
E_T = E0_T + E
dE = E[1] - E[0]



def background_spectrum(suppression_factor, N_Pu_chosen=N_Pu):
    # Flat U-237 background (counts per bin), divided by the suppression factor
    A_U = N_Pu_chosen * branch_alpha / suppression_factor
    u_raw = A_U * U_density * np.ones_like(E) * dE
    return u_raw



# --- Define the Forward Model ---
def generate_spectrum_Pu(m_nu_val, sigma_e_val, suppression_factor, N_Pu_chosen=N_Pu):
    """Generates the full smeared spectrum (Signal + Background)"""
    # Beta spectrum near the endpoint: N * C * sqrt(eps^2 - m^2) * eps * dE
    pu_raw = N_Pu_chosen * C_Pu * np.sqrt(np.maximum(E**2 - m_nu_val**2, 0)) * np.maximum(-E, 0) * dE
    u_raw = background_spectrum(suppression_factor, N_Pu_chosen)
    total_raw = pu_raw + u_raw

    sigma_bins = sigma_e_val / dE
    return gaussian_filter1d(total_raw, sigma_bins)


def generate_spectrum_T(m_nu_val, sigma_e_val):
    """Generates the smeared tritium spectrum (no background)"""
    # Beta spectrum near the endpoint: N * C * sqrt(eps^2 - m^2) * eps * dE
    T_raw = N_T * C_T * np.sqrt(np.maximum(E**2 - m_nu_val**2, 0)) * np.maximum(-E, 0) * dE

    sigma_bins = sigma_e_val / dE
    return gaussian_filter1d(T_raw, sigma_bins)


def get_chi2(m_nu_hyp, data_counts, data_errors, sigma_e_val, isotope, suppression_factor, N_Pu_chosen=N_Pu):
    """Computes Chi-squared for a hypothesized neutrino mass"""
    if isotope=='tritium':
        model_counts = generate_spectrum_T(m_nu_hyp, sigma_e_val)
    elif isotope=='plutonium':
        model_counts = generate_spectrum_Pu(m_nu_hyp, sigma_e_val, suppression_factor, N_Pu_chosen)
    else:
        print('Invalid particle type, assuming plutonium-241')
        model_counts = generate_spectrum_Pu(m_nu_hyp, sigma_e_val, suppression_factor, N_Pu_chosen)
    return np.sum(((data_counts - model_counts) / data_errors) ** 2)



def find_90_cl_upper_limit(true_m_nu, sigma_e_val, isotope, suppression_factor=1, N_Pu_chosen=N_Pu, m_nu_scan_points=1000):
    """Finds the 90% CL upper limit using the Asimov profile likelihood approach"""
    # 1. Generate the Asimov dataset (perfect data matching the true parameters)
    if isotope=='tritium':
        asimov_counts = generate_spectrum_T(true_m_nu, sigma_e_val)
        # 2. Define Poissonian errors (sqrt of counts)
        errors = np.sqrt(asimov_counts)
        errors[errors == 0] = 1.0  # Safeguard against zero counts
    elif isotope=='plutonium':
        asimov_counts = generate_spectrum_Pu(true_m_nu, sigma_e_val, suppression_factor, N_Pu_chosen)
        # 2. Define Poissonian errors (sqrt of counts)
        u_raw = background_spectrum(suppression_factor, N_Pu_chosen)
        errors = np.sqrt(asimov_counts + bg_bin_norm * u_raw)
        errors[errors == 0] = 1.0  # Safeguard against zero counts
    else:
        print('Invalid particle type, assuming plutonium-241')
        asimov_counts = generate_spectrum_Pu(true_m_nu, sigma_e_val, suppression_factor, N_Pu_chosen)
        # 2. Define Poissonian errors (sqrt of counts)
        u_raw = background_spectrum(suppression_factor, N_Pu_chosen)
        errors = np.sqrt(asimov_counts + bg_bin_norm * u_raw)
        errors[errors == 0] = 1.0  # Safeguard against zero counts

    # 3. Find minimum Chi2 (which is 0 for an Asimov dataset at the true value)
    min_chi2 = get_chi2(true_m_nu, asimov_counts, errors, sigma_e_val, isotope, suppression_factor, N_Pu_chosen)
    target_chi2 = min_chi2 + 2.71  # 2.71 delta-chi2 threshold for 1-sided 90% CL

    # 4. Scan upwards from the true value to find where delta_chi2 crosses 2.71
    m_nu_scan = np.linspace(true_m_nu, true_m_nu + 0.4, m_nu_scan_points)
    for m_hyp in m_nu_scan:
        current_chi2 = get_chi2(m_hyp, asimov_counts, errors, sigma_e_val, isotope, suppression_factor, N_Pu_chosen)
        if current_chi2 >= target_chi2:
            return m_hyp

    return m_nu_scan[-1] # Fallback boundary


def find_90_cl_root(true_m_nu, sigma_e_val, isotope, suppression_factor=1, N_Pu_chosen=N_Pu):
    """Same Asimov dataset, errors and delta-chi2 = 2.71 criterion as
    find_90_cl_upper_limit(), but the crossing is found with a root finder instead of
    a 0.0004 eV scan, which gives smooth curves."""
    if isotope == 'tritium':
        asimov = generate_spectrum_T(true_m_nu, sigma_e_val)
        errors = np.sqrt(asimov)
    else:
        asimov = generate_spectrum_Pu(true_m_nu, sigma_e_val, suppression_factor, N_Pu_chosen)
        errors = np.sqrt(asimov + bg_bin_norm * background_spectrum(suppression_factor, N_Pu_chosen))
    errors[errors == 0] = 1.0
    delta = lambda m: get_chi2(m, asimov, errors, sigma_e_val, isotope, suppression_factor, N_Pu_chosen) - 2.71
    try:    return brentq(delta, true_m_nu, true_m_nu + 0.4, xtol=1e-7)
    except ValueError: return true_m_nu + 0.4          # same cap as the scan


# --- Figure style (sensitivity-ratio and fresh-source figures) ---
FULL_WIDTH  = 10          # width [in] of full-width (multi-panel) figures
PANEL_SIZE  = (4.8, 5.6)  # half-width panel: width = 10 in x 0.48 (fraction of line width)
FONT        = 14          # font size for all text
DPI         = 130
PLOT_STYLE  = {
    'font.size':          FONT,
    'axes.labelsize':     FONT,  'axes.labelweight':  'normal',
    'axes.titlesize':     FONT,  'axes.titleweight':  'bold',   # titles bold
    'figure.titlesize':   FONT,  'figure.titleweight': 'bold',  # suptitle bold
    'xtick.labelsize':    FONT,  'ytick.labelsize':   FONT,
    'legend.fontsize':    FONT,
}

#%% Plotting functions


def fig1_beta_spectrum():
    selected_mass = 0.2                                     #True neutrino mass (eV)
    background_suppressions = np.geomspace(1,1000,4)     # Suppression of U background for Pu-241 decay
    resolution = 1                                    # Energy resolution (eV)
    
    cmap = mpl.colormaps['RdYlGn']
    n_lines=len(background_suppressions)
    colors = cmap(np.linspace(0, 1, n_lines))

    # Raw (unsmeared) signal profile; np.maximum avoids negative arguments of the square root
    pu_raw = N_Pu * C_Pu * np.sqrt(np.maximum(E**2 - selected_mass**2, 0)) * np.maximum(-E, 0) * dE

    # Tritium spectrum
    T_raw = N_T * C_T * np.sqrt(np.maximum(E**2 - selected_mass**2, 0)) * np.maximum(-E, 0) * dE
    T_smeared = generate_spectrum_T(selected_mass, resolution)
    
    plt.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    ax.plot(E, T_smeared, label=r'Smeared Tritium Spectrum ($\sigma_e$ = ' + f'{resolution} eV)',
             color='purple', lw=2.5)    
    ax.plot(E, T_raw, label=r'Signal Profile ($T_{raw}$)', color='tab:green', linestyle='--', lw=2)
    ax.plot(E, pu_raw, label=r'Signal Profile ($pu_{raw}$)', color='tab:orange', linestyle='--', lw=1)

    for i,f_sup in enumerate(background_suppressions):
        u_raw = background_spectrum(f_sup)
        
        # Total unsmeared raw plutonium and uranium spectrum
        total_raw_pu = pu_raw + u_raw
        # Generate the final smeared spectrum using the forward model function
        total_smeared = generate_spectrum_Pu(selected_mass, resolution, f_sup)
        ax.plot(E, total_smeared, label=rf'Total Spectrum ($\sigma_e$ = {resolution} eV, $f_{{\rm bg}}$={f_sup})',
                     color=colors[i], lw=2.5)
        ax.plot(E, u_raw, color=colors[i], linestyle=':', lw=2)
    
    ax.axvline(0, color='black', linestyle='-', alpha=0.3, label=r'Endpoint $E_0$')
    ax.axvline(0 - selected_mass, color='purple', linestyle='-.', alpha=0.6,
                    label=r'Mass Threshold ($E_0 - m_{\nu}$)')
    
    ax.set_yscale('log')
    ax.set_ylim(1e2,1e6)
    ax.set_xlim(-10,1)
    ax.legend()
    ax.set_xlabel('Distance to endpoint $E-E_0$ (eV)')
    ax.set_ylabel('Counts / Intensity (Log Scale)')
    plt.savefig('fig1_betaspectrum.png')



def fig2_fresh_source():
    """U-237 buildup after purification: instantaneous suppression f_bg^eff(t)
    (solid) and the suppression f_bg^avg(T) averaged over a cleaning cycle of length T
    (dash-dot), both relative to secular equilibrium."""
    print("\nFresh source: instantaneous vs cycle-averaged suppression ...")
    tau_Pu = TauPu
    tau_U  = TauU                                # U-237 mean lifetime [s]

    t_days = np.logspace(-1, 3, 400)             # time since purification [d]
    t_sec  = t_days * day2sec

    N_Pu_t = np.exp(-t_sec / tau_Pu)
    # Bateman solution of dN_U/dt = b*lam_Pu*N_Pu - lam_U*N_U with N_U(0)=0
    N_U_t  = (branch_alpha * tau_U / (tau_Pu - tau_U) *
              (np.exp(-t_sec/tau_Pu) - np.exp(-t_sec/tau_U)))
    N_U_t  = np.maximum(N_U_t, 1e-30)

    SB_equil = 1.0 / (branch_alpha * tau_U / tau_Pu)
    f_bg_eff = (N_Pu_t / N_U_t) / SB_equil       # instantaneous suppression

    # cycle-averaged suppression for a cleaning cycle of length T = t_days:
    # <N_U>/N_U^eq = 1 - (1 - exp(-T/tau_U)) / (T/tau_U), f_avg = inverse of this
    x_T      = t_days * day2sec / tau_U
    f_bg_avg = 1.0 / (1.0 - (-np.expm1(-x_T)) / x_T)

    t_50 = -tau_U * np.log(0.5) / day2sec
    t_90 = -tau_U * np.log(0.1) / day2sec

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=PANEL_SIZE)
        ax.grid(True, ls='--', alpha=0.3, color='lightgray', which='both')

        ax.fill_between(t_days, 1, f_bg_eff, where=f_bg_eff > 1.,
                        color='green', alpha=0.15, label='Fresh-source advantage')
        ax.plot(t_days, f_bg_eff, color='steelblue', lw=2.5,
                label=r'Instantaneous $f_{\rm bg}^{\rm eff}(t)$')
        ax.plot(t_days, f_bg_avg, color='darkorange', lw=2.5, ls='-.',
                label=r'Time-averaged $f_{\rm bg}^{\rm avg}(T)$')
        ax.axhline(1., color='red', ls='--', lw=2, label='Secular equilibrium')
        ax.axvline(t_50, color='gray', ls=':',  lw=1.5)
        ax.text(t_50*0.90, 30, '50%', rotation=90, ha='right', va='center', color='gray')
        ax.axvline(t_90, color='gray', ls='--', lw=1.5)
        ax.text(t_90*0.90, 30, '90%', rotation=90, ha='right', va='center', color='gray')

        ax.annotate(r'10$\times$ better' + '\nat $t$ = 1 day',
                    xy=(1, 10), xytext=(0.12, 1.4), fontsize=FONT, color='green',
                    arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(0.1, 1000); ax.set_ylim(0.8, 1000)
        ax.set_xlabel('Time since purification $t$\nor cycle length $T$ (days)')
        ax.set_ylabel(r'Effective $f_{\rm bg}$ relative to equilibrium')
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig('fig2_fresh_source.png', dpi=DPI)
        plt.close()
    print("Saved: fig2_fresh_source.png")


def fig3_sensitivity():
    resolutions = np.geomspace(0.01, 10, 40)            # Energy resolution grid (eV)
    true_mass = 0.2                                     # True neutrino mass (eV)
    background_suppressions = np.geomspace(1,1000,8)     # Suppression of U background for Pu-241 decay
    
    cmap = mpl.colormaps['RdYlGn']
    n_lines=len(background_suppressions)
    colors = cmap(np.linspace(0, 1, n_lines))
    
    results_pu = np.zeros((len(background_suppressions), len(resolutions)))
    results_T = np.zeros(len(resolutions))
    
    plt.rcParams.update({'font.size': 14})
    
    fig = plt.figure(figsize=[10,5])
    fig.patch.set_facecolor('white')
    ax_total_activity = fig.add_axes([0.1,0.15,0.35,0.75])
    ax_nearby_activity = fig.add_axes([0.50,0.15,0.35,0.75])
    ax_cbar = fig.add_axes([0.9,0.15,0.02,0.75])
    ax_total_activity.set_facecolor('white')
    ax_nearby_activity.set_facecolor('white')
    ax_cbar.set_facecolor('white')
    ax_total_activity.grid(True, ls='--', alpha=0.3, color='lightgray', which='both')
    ax_nearby_activity.grid(True, ls='--', alpha=0.3, color='lightgray', which='both')
    
    N_Pu_total_activity = N_Pu
    N_Pu_nearby_activity = C_T/C_Pu * N_T
    
    for j,res in enumerate(resolutions):
        limit = find_90_cl_upper_limit(true_mass, res, isotope='tritium')
        results_T[j]=limit
    ax_total_activity.plot(resolutions, results_T, label='tritium', color='purple')
    ax_nearby_activity.plot(resolutions, results_T, label='tritium', color='purple')
    print(results_T)
    
    for i, f_sup in enumerate(background_suppressions):
        for j, res in enumerate(resolutions):
            limit = find_90_cl_upper_limit(true_mass, res, isotope='plutonium', suppression_factor=f_sup, N_Pu_chosen=N_Pu_total_activity)
            results_pu[i][j]=limit
        ax_total_activity.plot(resolutions, results_pu[i], color=colors[i])
        print(results_pu[i])
        
    ax_total_activity.hlines(true_mass, xmin=min(resolutions), xmax=max(resolutions), linestyles='--', colors='k', label='true mass')
    ax_total_activity.set_xscale('log')
    ax_total_activity.set_xlabel(r'$\sigma_e$')
    ax_total_activity.set_xlim(left=0.05, right=10)
    ax_total_activity.set_ylabel(r'90% CL Upper Limit Sensitivity (eV)')
    ax_total_activity.set_ylim(bottom=0.19, top=0.45)
    ax_total_activity.set_title('Same total activity', fontweight='bold')

    for i, f_sup in enumerate(background_suppressions):
        for j, res in enumerate(resolutions):
            limit = find_90_cl_upper_limit(true_mass, res, isotope='plutonium', suppression_factor=f_sup, N_Pu_chosen=N_Pu_nearby_activity)
            results_pu[i][j]=limit
        ax_nearby_activity.plot(resolutions, results_pu[i], color=colors[i])
        print(results_pu[i])
        
    ax_nearby_activity.hlines(true_mass, xmin=min(resolutions), xmax=max(resolutions), linestyles='--', colors='k', label='true mass')
    ax_nearby_activity.set_xscale('log')
    ax_nearby_activity.set_xlim(left=0.05, right=10)
    ax_nearby_activity.set_xlabel(r'$\sigma_e$')
    ax_nearby_activity.set_ylim(bottom=0.19, top=0.45)
    ax_nearby_activity.set_title('Same activity in final eV', fontweight='bold')
    ax_nearby_activity.set_yticklabels([])
    ax_nearby_activity.legend()  
    

    
    norm = mpl.colors.LogNorm(vmin=min(background_suppressions), vmax=max(background_suppressions))
    sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=norm) 
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cbar)
    cbar.ax.set_yscale('log')
    cbar.ax.invert_yaxis()
    cbar.set_label('Background suppression factor')
    

    plt.savefig('fig3_sensitivity.png')



def fig4_pumass_scan():
    resolutions = np.geomspace(0.05, 10, 40)            # Energy resolution grid (eV)
    pu_mass_factors = np.geomspace(1E-4,1E-1,7)        # Pu-241 source masses in grams (compared with 1 ug of tritium)
    suppression_factor_1 = 10                        # suppression of U background
    suppression_factor_2 = 100
    true_neutrino_mass = 0.2                        # eV

    results_pu = np.zeros((len(pu_mass_factors), len(resolutions)))
    results_T = np.zeros(len(resolutions))
    activity_factor = np.zeros(len(pu_mass_factors))
    
    plt.rcParams.update({'font.size': 14})

    fig = plt.figure(figsize=[10,5])
    fig.patch.set_facecolor('white')
    ax_10 = fig.add_axes([0.1,0.15,0.35,0.75])
    ax_100 = fig.add_axes([0.50,0.15,0.35,0.75])
    ax_cbar = fig.add_axes([0.9,0.15,0.02,0.75])
    ax_10.set_facecolor('white')
    ax_100.set_facecolor('white')
    ax_cbar.set_facecolor('white')
    ax_10.grid(True, ls='--', alpha=0.3, color='lightgray', which='both')
    ax_100.grid(True, ls='--', alpha=0.3, color='lightgray', which='both')
    
    cmap = mpl.colormaps['RdYlGn']
    n_lines=len(pu_mass_factors)
    colors = cmap(np.linspace(0, 1, n_lines))

    for i, m_pu in enumerate(pu_mass_factors):
        n_pu_variable = m_pu * N_A * GPu * run_time / 241
        activity_factor[i] = n_pu_variable/N_Pu
        for j, res in enumerate(resolutions):
            limit = find_90_cl_upper_limit(true_neutrino_mass, res, isotope='plutonium', suppression_factor=suppression_factor_1, N_Pu_chosen=n_pu_variable)
            results_pu[i][j]=limit
        ax_10.plot(resolutions, results_pu[i], color=colors[i])
        print(results_pu[i])
    
    for j,res in enumerate(resolutions):
        limit = find_90_cl_upper_limit(true_neutrino_mass, res, isotope='tritium')
        results_T[j]=limit
    ax_10.plot(resolutions, results_T, label=r'tritium(1 $\mu$g)', color='purple')
    print(results_T)
    
    ax_10.hlines(true_neutrino_mass, xmin=min(resolutions), xmax=max(resolutions), linestyles='--', colors='k', label='true mass')
    ax_10.set_xscale('log')
    ax_10.set_xlabel(r'$\sigma_e$')
    ax_10.set_ylabel(r'90% CL Upper Limit Sensitivity (eV)')
    ax_10.set_ylim(bottom=true_neutrino_mass-0.01)
    ax_10.set_xlim(left=0.05, right=10)
    ax_10.set_yticks([0.2, 0.25, 0.3, 0.35])
    ax_10.set_title('Suppression factor 10', fontweight='bold')

    for i, m_pu in enumerate(pu_mass_factors):
        n_pu_variable = m_pu * N_A * GPu * run_time / 241
        activity_factor[i] = n_pu_variable/N_Pu
        for j, res in enumerate(resolutions):
            limit = find_90_cl_upper_limit(true_neutrino_mass, res, isotope='plutonium', suppression_factor=suppression_factor_2, N_Pu_chosen=n_pu_variable)
            results_pu[i][j]=limit
        ax_100.plot(resolutions, results_pu[i], color=colors[i])
        print(results_pu[i])
    
    for j,res in enumerate(resolutions):
        limit = find_90_cl_upper_limit(true_neutrino_mass, res, isotope='tritium')
        results_T[j]=limit
    ax_100.plot(resolutions, results_T, label=r'tritium(1 $\mu$g)', color='purple')
    print(results_T)


    norm = mpl.colors.LogNorm(vmin=min(activity_factor), vmax=max(activity_factor))
    sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=norm) 
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cbar)
    cbar.ax.set_yscale('log')
    cbar.ax.invert_yaxis()
    cbar.set_label(r'Activity Pu / Activity T')
    
    
    ax_100.hlines(true_neutrino_mass, xmin=min(resolutions), xmax=max(resolutions), linestyles='--', colors='k', label='true mass')
    ax_100.set_xscale('log')
    ax_100.set_xlabel(r'$\sigma_e$')
    ax_100.set_ylim(ax_10.get_ylim())
    ax_100.set_xlim(left=0.05, right=10)
    ax_100.set_yticks([0.2, 0.25, 0.3, 0.35])
    ax_100.set_yticklabels([])
    ax_100.set_title('Suppression factor 100', fontweight='bold')
    ax_100.legend()
    plt.savefig('fig4_pumass_scan.png')


def fig5_sensitivity_ratio():
    """Sensitivity ratio UL_Pu/UL_T with endpoint normalization
    (N_Pu*C_Pu = N_T*C_T, about 206 ug of Pu-241 vs 1 ug of tritium)."""
    m_true  = 0.2                                 # true neutrino mass (eV)
    print(f"\nSensitivity ratio (endpoint norm, m_true={m_true} eV) ...")

    N_Pu_end = N_T * (C_T / C_Pu)
    mass_Pu  = N_Pu_end * 241. * TauPu / (1e-6 * N_A * run_time)
    print(f"  Endpoint norm: {mass_Pu:.1f} ug, m_true = {m_true} eV")

    def ul_T(se):
        return find_90_cl_root(m_true, se, 'tritium')

    def ul_Pu_f(se, f_bg):
        return find_90_cl_root(m_true, se, 'plutonium', f_bg, N_Pu_end)

    sigmas = np.logspace(-2, 1, 25)
    f_bgs  = np.logspace(0, 3, 25)
    ratio  = np.zeros((len(f_bgs), len(sigmas)))
    ulT    = np.array([ul_T(s) for s in sigmas])
    for i, f in enumerate(f_bgs):
        for j, s in enumerate(sigmas):
            ratio[i, j] = ul_Pu_f(s, f) / ulT[j]
        print(f"  f_bg={f:.1f}", flush=True)

    col = {1:'#e41a1c', 10:'#ff7f0e', 100:'#984ea3', 250:'steelblue', 1000:'#2ca02c'}
    sf  = np.logspace(-2, 1, 60)
    utf = np.array([ul_T(s) for s in sf])

    with plt.rc_context(PLOT_STYLE):
        fig = plt.figure(figsize=(FULL_WIDTH, 5.6))
        # equal-width axes; the gap between them holds the colorbar and its labels
        gs   = fig.add_gridspec(1, 2, left=0.085, right=0.985, top=0.92, bottom=0.17,
                                wspace=0.62)
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]

        # Left: 2D heatmap (colorbar placed beside the axes, not stealing width from them)
        ax = axes[0]
        nr   = mcolors.Normalize(vmin=1.0, vmax=ratio.max())
        mesh = ax.pcolormesh(sigmas, f_bgs, ratio, shading='auto', cmap='RdYlGn_r', norm=nr)
        cax  = ax.inset_axes([1.04, 0.0, 0.05, 1.0])
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.set_label(r'$\mathrm{UL_{Pu}} / \mathrm{UL_T}$')
        cs = ax.contour(sigmas, f_bgs, ratio, levels=[1.05, 1.1, 1.2, 1.5],
                        colors=['black','white','white','white'],
                        linewidths=[2.5,1.5,1.5,1.5])
        ax.clabel(cs, inline=True, fontsize=FONT,
                  fmt={1.05:'1.05×', 1.1:'1.1×', 1.2:'1.2×', 1.5:'1.5×'})
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(1e-2, 10); ax.set_ylim(1, 1000)
        ax.set_xlabel(r'Energy resolution $\sigma_e$ (eV)')
        ax.set_ylabel(r'Background suppression $f_{\rm bg}$')
        ax.set_title('2D heatmap')

        # Right: ratio vs sigma_e
        ax = axes[1]
        for f, c in col.items():
            rv = [ul_Pu_f(s, f)/utf[k] for k, s in enumerate(sf)]
            ax.plot(sf, rv, color=c, lw=2.5, label=rf'$f_{{\rm bg}}$ = {f}')
        ax.axhline(1., color='black', ls='--', lw=2, label='Equal sensitivity')
        ax.fill_between([1e-2,10], 0.97, 1.0, color='green', alpha=0.08)
        ax.fill_between([1e-2,10], 1.0,  2.6, color='red',   alpha=0.06)
        ax.set_xscale('log')
        ax.set_xlabel(r'Energy resolution $\sigma_e$ (eV)')
        ax.set_ylabel(r'$\mathrm{UL_{Pu}} / \mathrm{UL_T}$')
        ax.set_title(r'Ratio vs $\sigma_e$ at fixed $f_{\rm bg}$')
        ax.legend(loc='upper right', handlelength=1.5)
        ax.grid(True, ls='--', alpha=0.4, which='both')
        ax.set_xlim(1e-2, 10); ax.set_ylim(0.97, 2.6)

        plt.savefig('fig5_sensitivity_ratio.png', dpi=DPI)
        plt.close()
    print("Saved: fig5_sensitivity_ratio.png")


# Figures in the order of the paper
fig1_beta_spectrum()
fig2_fresh_source()
fig3_sensitivity()
fig4_pumass_scan()
fig5_sensitivity_ratio()
