# Pu241-Neutrino-Mass-Supplement
Supplementary plots and analysis materials for the Pu-241 neutrino mass paper.
## Additional plots (`Plots paper/`)

The paper shows Figs. 1, 3, 4 and 5 for a true neutrino mass of 0.2 eV. This folder has the same four plots for other true masses: 0, 50, 100 and 150 meV. The mass is in the file name (`mnu050meV` means m_ν^true = 0.05 eV). Constants, energy grid, spectrum model and the 90% CL procedure are the same for all plots.

- **`fig1_betaspectrum_*`: spectral decomposition near the endpoint.** The tritium spectrum and the Pu-241 spectrum are shown with the Pu-241 source normalized by its total activity. The U-237 background is shown for background suppression factors f_bg = 1, 10, 100 and 1000. This shows how the background compares with the signal in the last tens of eV.
- **`fig3_sensitivity_*`: expected 90% CL upper limit versus energy resolution σ_e** for Pu-241 and tritium. Left panel: equal total activity. Right panel: equal activity in the endpoint region. Higher background suppression gives a better limit.
- **`fig4_pumass_scan_*`: how much extra Pu-241 is needed to match tritium.** Shows the upper limit as a function of the Pu-241 amount (activity) compared with the tritium reference, for f_bg = 10 and 100, at different energy resolutions. The crossing point shows the activity at which Pu-241 becomes as sensitive as tritium.
- **`fig5_sensitivity_ratio_*` (made with `extra_sensitivity_ratio_plots.py`): sensitivity ratio UL_Pu/UL_T** with the endpoint normalization N_Pu·C_Pu = N_T·C_T, which corresponds to about 206 µg of Pu-241 versus 1 µg of tritium (3-year exposure). Left: heatmap of the ratio versus σ_e and f_bg, with contours at 1.05, 1.1, 1.2 and 1.5. Right: the ratio versus σ_e for fixed f_bg. A ratio of 1 means equal sensitivity, and a ratio above 1 means tritium is better.

**Scripts**
- `pu241_vs_tritium_sensitivity.py` produces Figs. 1–5 of the paper (m_ν^true = 0.2 eV).

