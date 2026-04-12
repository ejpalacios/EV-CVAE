# Synthetic EV Charging Session Generation Using a Conditional Variational Autoencoder



## Overview

This work proposes a Conditional Variational Autoencoder (CVAE) for generating synthetic electric vehicle (EV) charging sessions from real transaction-level data. The model is trained on session-level features describing plug-in duration, charging duration, delivered energy, charging delay, and cyclical time-of-week, while conditioning on day-of-week and managed charging status.

The generated synthetic data is evaluated using distributional similarity metrics (Wasserstein distance, Spearman correlation MAE, ACF MAE) and utility-based validation via the Train-on-Synthetic-Test-on-Real (TSTR) protocol.

\---

## Repository Structure

```
├── data/
│   ├── CrowdCharge\_Transactions.parquet      # Raw CrowdCharge session data (Electric Nation)
│   └── GreenFlux\_Transactions.parquet        # Raw GreenFlux session data (Electric Nation)
├── Preprocessing\_V3\_4\_7\_Combined.ipynb       # Data preprocessing pipeline
├── V3\_4\_8\_Combined.ipynb                     # CVAE model training, evaluation \& generation
└── README.md
```

\---

## Notebooks

### `EV\_charging\_transactions\_Preprocessing\_Final.ipynb`

Merges the CrowdCharge and GreenFlux datasets, filters outliers, selects and engineers features, applies cyclic time encoding, and saves the preprocessed dataset ready for model training.

### `EV\_CVAE\_Final.ipynb`

Defines and trains the CVAE model, produces training diagnostics, evaluates generated data against real data using distributional and utility-based metrics, and generates synthetic charging sessions conditioned on day-of-week and managed/unmanaged status.

\---

## Data

The raw data is sourced from the [Electric Nation](https://www.electricnation.org.uk/) smart charging trial (CrowdCharge and GreenFlux datasets), covering residential EV charging sessions from early 2017 to end of 2018 across 602 charging points (\~157k sessions combined).

\---

## Engineered Features

|Feature|Description|
|-|-|
|`PluggedInTime`|Total plug-in duration (hours)|
|`ChargingDuration`|Active charging duration (hours)|
|`ConsumedkWh`|Energy delivered during the session (kWh)|
|`TimeGapHours`|Time between plug-in and charging start (hours)|
|`WeekTime\_sin`|Sine component of cyclical week-time encoding|
|`WeekTime\_cos`|Cosine component of cyclical week-time encoding|
|`DayOfWeek`|Day of the week (0–6, one-hot encoded, conditioning variable)|
|`Part\_of\_Managed\_Group`|Smart charging group membership (0/1, conditioning variable)|

\---

## Model

The CVAE uses a fully connected MLP architecture for both encoder and decoder:

* **Input dimension:** 14 (6 continuous features + 8-dimensional condition vector)
* **Hidden layers:** 2 × 256 neurons with ReLU activations
* **Latent dimension:** 64
* **Reconstruction loss:** Gaussian negative log-likelihood (NLL)
* **Regularisation:** KL divergence toward standard normal prior (β = 0.5)
* **Optimiser:** Adam, lr = 3×10⁻⁴, 60 epochs, batch size 256

\---

## Requirements

```
polars
numpy
torch
matplotlib
scipy
scikit-learn
pandas
```

The notebooks are designed to run in **Google Colab** with data files stored in Google Drive. File paths will need to be adjusted accordingly.

\---

