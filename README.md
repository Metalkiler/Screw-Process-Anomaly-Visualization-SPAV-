# SPAV  — Screw Process Anomaly Visualization

Following the Industrial Revolutions 4.0 and 5.0, the assembly industry faces mounting pressure to reduce costs and boost efficiency using AI‑driven solutions. Modern screwdriver systems generate real‑time angle‑torque data that form tightening curves, which must be frequently recalibrated and inspected to prevent defective assemblies. We propose modeling normal screwing profiles with unsupervised AutoEncoders (AE) to detect deviations indicative of faults, complemented by intuitive visual analytics. Our Python module, Screw Process Anomaly Visualization (SPAV), offers functions for creating four types of plots—Global Error, Local Error, Detected Anomalies and Anomaly Density via KDE—to support operators in understanding and validating anomaly detection. SPAV integrates seamlessly with the Scientific Python ecosystem and is compatible with [H2O](https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/intro.html) and [Keras](https://keras.io/) AE models.  

## Purpose and features

The quality assessment of tightening curves is especially important in anomaly detection because the screw tightening process is inherently dynamic, requiring frequent updates to defect catalogs and recalibration of screwing zones. Although defects are rare, detailed and automated inspection is essential to prevent faulty assemblies from progressing along the production line.

Screw Process Anomaly Visualization (SPAV) is a Python software module that was recently developed to address the specific challenge of performing a data-driven support to the calibration of screwing machines in manufacturing environments, which is a frequent issue in real-world industrial manufacturing settings. The primary goal of SPAV is to provide a suite of XAI visualization techniques (e.g. density plots) that enable users of all levels of expertise to easily identify critical anomalous points that arise during the screwing process from existing predictive ML models.

<img width="1749" height="792" alt="{FEFA4678-4B3D-4F2B-A8EB-B27E619898DE}" src="https://github.com/user-attachments/assets/d81c207e-2639-427d-bd1f-9b9c495c6814" />

## Example (codeocean executable run)
An [executable example](https://doi.org/10.24433/CO.1214166.v1) is available on CodeOcean. We invite users to execute and verify the full executable example to better understand how to operate the module and the types of plots that can be obtained through its functions.

## Installation Guide

All files required for running SPAV are under the `spav` folder of this repository.

First, we recommend installing the dependencies for this project, which are listed in a pip-friendly way in `requirements.txt`. From within the `spav` directory, execute in the command line:

```cmd
pip install requirements.txt
```

Following that, users can import `spav` from within their project directory containing the `spav` directory, either in full, per-module or per-function:

```python
import spav
from spav import auxiliary functions
from spav.plot_functions import plot_reconstruction
```

## Usage Guidelines

Plot functions are available under the `plot_functions` component of the `spav` module, while `auxiliary_functions` contain functions to help users in the data preprocessing stage.

Given a compatible AutoEncoder model and a preprocessed angle-torque pair dataset, users can create a global error plot that shows all screwing processes in the dataset colored by their (by default, normalized) anomaly scores:

```python
from spav.auxiliary functions import get_anomaly scores
from spav.plot_functions import plot_global_error

anomaly_df = get_anomaly_scores(model, screwing_df)
plot_global_error(test_df, anomaly_df, y="torque")
```

<img width="1649" height="936" alt="{774101E3-6CEF-49AD-A95F-CD13BBE368C2}" src="https://github.com/user-attachments/assets/9a24affa-1f7d-43b0-be70-d29fd047612d" />


## Authors
**Marta Moreno $^{1}$, Hugo Rocha $^{1}$ , André Pilastri $^{2}$, Guilherme Moreira $^{3}$, Luís Miguel Matos $^{1}$, Paulo Cortez $^{\star}$ $^{1,2}$\
$^{1}$ - ALGORITMI Centre, Minho University, Guimarães, Portugal\
$^{2}$ - EPMQ, CCG ZGDV Institute, Guimarães, Portugal\
$^{3}$ - Bosch Car Multimedia, Braga, Portugal\
$^{\star}$ - correponding author (pcortez@dsi.uminho.pt)**
