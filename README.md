# Pu241-Neutrino-Mass-Supplement
Supplementary plots and analysis materials for the Pu-241 neutrino mass paper.
## Additional plots (`Plots paper/`)

This folder has the same four plots as Figs. 1, 3, 4 and 5 of the paper, but for other true neutrino masses: 0, 50, 100 and 150 meV. The paper's own figures (m_ν^true = 0.2 eV) are not included here. The mass is in the file name, so `mnu050meV` means m_ν^true = 0.05 eV.

- `fig1_betaspectrum_*`: Pu-241 and tritium spectra near the endpoint, with the U-237 background for background suppression f_bg = 1, 10, 100 and 1000.
- `fig3_sensitivity_*`: expected 90% CL upper limit versus energy resolution σ_e for Pu-241 and tritium, at equal total activity (left) and equal endpoint activity (right).
- `fig4_pumass_scan_*`: extra Pu-241 activity needed to match the tritium sensitivity, for f_bg = 10 and 100.
- `fig5_sensitivity_ratio_*`: ratio UL_Pu/UL_T at equal endpoint activity (about 206 µg of Pu-241 versus 1 µg of tritium), as a heatmap in σ_e and f_bg (left) and versus σ_e at fixed f_bg (right). A ratio above 1 means tritium is more sensitive.

All plots use the same constants, energy grid, spectrum model and 90% CL procedure. `pu241_vs_tritium_sensitivity.py` produces the paper's Figs. 1–5, and `extra_sensitivity_ratio_plots.py` makes the ratio plots for other masses (`python3 extra_sensitivity_ratio_plots.py 0.15`). 
**Scripts**
- `pu241_vs_tritium_sensitivity.py` produces Figs. 1–5 of the paper (m_ν^true = 0.2 eV).

