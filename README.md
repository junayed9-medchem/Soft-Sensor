# Pharmaceutical Process Monitoring — MSPC & Raman Soft Sensor

A Python implementation of two complementary Process Analytical Technology (PAT)
tools applied to an industrial-scale penicillin fermentation benchmark:

1. **Multivariate Statistical Process Control (MSPC)** real-time fault detection
   from process sensor data using PCA-based control charts
2. **Raman Soft Sensor** — real-time prediction of penicillin and substrate
   concentrations directly from Raman spectroscopy, eliminating the need
   for slow, costly offline laboratory assays

Both tools are applied to the publicly available **IndPenSim** dataset
(Goldrick et al., 2019), a validated simulation of a 100,000 L industrial
penicillin fed-batch fermentation process.

---

## Background

Pharmaceutical manufacturing requires continuous monitoring of critical
process parameters (CPPs) and critical quality attributes (CQAs) to ensure
product safety and consistency. Traditional quality control relies on
offline laboratory testing a slow, destructive process that cannot
catch problems in real time.

**Process Analytical Technology (PAT)**, as defined in FDA's PAT Guidance
(2004) and ICH Q8(R2), enables manufacturers to measure and control quality
in real time using data-driven tools. This project implements two of the
most important PAT methods recognised by FDA:

- **MSPC** for process monitoring and fault detection
- **Raman spectroscopy soft sensors** for real-time CQA prediction

---

## Dataset

**IndPenSim** Industrial Penicillin Simulation Dataset
Goldrick S., Duran-Villalobos C., Jankauskas K., Lovett D., Farid S.,
Lennox B. (2019). Modern day monitoring and control challenges outlined
on an industrial-scale benchmark fermentation process.
*Computers & Chemical Engineering*, 130, 106471.

Available at: https://data.mendeley.com/datasets/pdnjz7zz5x/1

The dataset contains 100 batches from a 100,000 L penicillin bioreactor:
- Batches 1-30   : Recipe-driven control (normal — used for training)
- Batches 31-60  : Operator-controlled (normal)
- Batches 61-90  : Advanced Process Control using Raman spectroscopy
- Batches 91-100 : Known process faults (used for fault detection testing)

Each batch includes:
- 23 online process variables (temperature, pH, dissolved oxygen, etc.)
  sampled every 0.2 hours over ~230 hours of fermentation
- 2,200 Raman spectroscopy channels (205-2400 cm-1)
- Offline assay measurements (penicillin, substrate, biomass)

## Methods

### Part 1: Multivariate Statistical Process Control (MSPC)

#### PCA Process Model
A PCA model is trained on 30 normal batches (34,500 observations,
23 process variables). The model captures the normal correlation
structure of the process. 8 principal components explain 83.5%
of the total variance.

#### Hotelling T-squared Statistic
Measures deviation within the PCA model space — detects shifts
in the normal process correlation structure.

    T2 = sum( t_a^2 / lambda_a )  for a = 1 to A

Control limit based on the F-distribution (Kourti & MacGregor, 1995).

#### SPE / Q Statistic
Measures deviation outside the model (residuals) — detects novel
fault patterns not captured during normal training.

    SPE = sum( e_j^2 )  for j = 1 to p

Control limit based on chi-squared approximation
(Jackson & Mudholkar, 1979).

#### Contribution Plots
Decomposes the T-squared statistic into per-variable contributions
to identify which process variable caused the alarm. Standard tool
for root cause analysis in pharmaceutical manufacturing.

#### Isolation Forest (Comparison)
A modern ML anomaly detection method included as a comparison
against the interpretable MSPC approach.

#### MSPC Results (95% confidence limits)

    PCA components     : 8  (83.5% variance explained)
    T2 limit           : 15.51
    SPE limit          : 15.44
    Faults detected    : 9 / 10  (90%)
    False alarm rate   : 10.3%
    Isolation Forest AUC : 0.662

---

### Part 2: Raman Soft Sensor

#### Spectral Preprocessing Pipeline
1. Savitzky-Golay smoothing (window=15, poly=3) — reduces instrument noise
2. Standard Normal Variate (SNV) — removes multiplicative scatter effects
3. First derivative (SG) — removes additive baseline drift

#### PLS Regression
Partial Least Squares regression maps Raman spectra to analyte
concentrations. Separate models are trained for penicillin and
substrate. Component count selected by 5-fold grouped cross-validation.

#### Variable Importance in Projection (VIP)
Identifies which Raman shift wavenumbers (cm-1) contribute most to
each prediction. VIP >= 1.0 indicates analytical significance and
corresponds to known chemical absorption bands.

#### Cross-Validation Strategy
5-fold grouped cross-validation ensures complete batches appear in
either training or validation — never split across folds. This gives
a realistic estimate of generalisation to a new batch.

#### Soft Sensor Results

    Target       | PLS Components | RMSECV     | R2 (CV)
    -------------|----------------|------------|--------
    Penicillin   |       7        | 4.689 g/L  |  0.781
    Substrate    |      11        | 5.276 g/L  |  0.797

#### Fault Batch Behaviour
Prediction error increases after fault onset in all 10 fault batches,
demonstrating that process deviations produce measurable Raman spectral
changes. This connects spectroscopic monitoring to process fault detection.

---

## Key Findings

1. MSPC detects 90% of known fault batches at a 10.3% false alarm
   rate using online process sensor data alone.

2. PLS soft sensor achieves R2 = 0.781 for penicillin and
   R2 = 0.797 for substrate from Raman spectra in a fully
   cross-validated framework.

3. Fault batches produce measurable Raman spectral changes, causing
   prediction error to increase after fault onset — connecting
   spectroscopic monitoring to fault detection.

4. PCA contribution plots correctly identify the most affected process
   variables in fault batches, providing actionable diagnostic
   information for root cause analysis.

---

## References

1. Goldrick S. et al. (2019). Modern day monitoring and control challenges
outlined on an industrial-scale benchmark fermentation process.
Computers & Chemical Engineering, 130, 106471.

2. Kourti T. & MacGregor J.F. (1995). Process analysis, monitoring and
diagnosis using multivariate projection methods. Chemometrics and
Intelligent Laboratory Systems, 28(1), 3-21.

3. Jackson J.E. & Mudholkar G.S. (1979). Control procedures for residuals
associated with principal component analysis. Technometrics, 21(3), 341-349.

4. Wold S. et al. (2001). PLS-regression: a basic tool of chemometrics.
Chemometrics and Intelligent Laboratory Systems, 58(2), 109-130.

5. FDA Guidance for Industry: PAT (2004).
https://www.fda.gov/media/71012/download

---

## Author

Md Junayed Nayeen, Ph.D.
Master of Data Science (2027), University of Pittsburgh
