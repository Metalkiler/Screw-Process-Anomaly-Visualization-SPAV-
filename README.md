# SPAV  — Screw Process Anomaly Visualization

Following the Industrial Revolutions 4.0 and 5.0, the assembly industry faces mounting pressure to reduce costs and boost efficiency using AI‑driven solutions. Modern screwdriver systems generate real‑time angle‑torque data that form tightening curves, which must be frequently recalibrated and inspected to prevent defective assemblies. We propose modeling normal screwing profiles with unsupervised AutoEncoders (AE) to detect deviations indicative of faults, complemented by intuitive visual analytics. Our Python module, Screw Process Anomaly Visualization (SPAV), offers functions for creating four types of plots—Global Error, Local Error, Detected Anomalies and Anomaly Density via KDE—to support operators in understanding and validating anomaly detection. SPAV integrates seamlessly with the Scientific Python ecosystem and is compatible with [H2O](https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/intro.html) and [Keras] (https://keras.io/) AE models.  

## Purpose and features

With the advent of the recent industrial revolutions (Industry 4.0 and 5.0), several assembly companies are under pressure to lower costs and increase operational efficiency through Artificial Intelligence (AI) tools.
In particular, screwdriver systems are often used in the assembly industry (e.g. electronic components), generating hundreds of angle-torque real-time pair values for a single screw tightening process. The tightening curve can be subsequently analyzed for quality assessment purposes (e.g. anomaly detection). 
The quality assessment step is especially important because the screw tightening process is inherently dynamic, thus requiring frequent updates to defect catalogs and recalibration of screwing zones through human evaluation of tightening curves and quality estimates. Although defects are rare, detailed and automated inspection is essential to prevent faulty assemblies from progressing along the production line.

Currently, screwdriver systems provide a "Good or Fail" (GoF) status based on predefined normal screw historical catalogs, making it susceptible of generating false positives and false negatives.
Screw Process Anomaly Visualization (SPAV) is a Python software module that was recently developed to address the specific challenge of performing a data-driven support to the calibration of screwing machines in manufacturing environments, which is a frequent issue in real-world industrial manufacturing settings. The primary goal of SPAV is to provide a suite of XAI visualization techniques (e.g. density plots) that enable users of all levels of expertise to easily identify critical anomalous points that arise during the screwing process from existing predictive ML models.

## Example (codeocean executable run)
An [executable example](https://doi.org/10.24433/CO.1214166.v1) is available on CodeOcean. We invite users to execute and verify the full executable example to better understand how to operate the module and the types of plots that can be obtained through its functions.

## Authors
**Marta Moreno $^{1}$, Hugo Rocha $^{1}$ , André Pilastri $^{2}$, Guilherme Moreira $^{3}$, Luís Miguel Matos $^{1}$, Paulo Cortez $^{\star}$ $^{1,2}$\
$^{1}$ - ALGORITMI Centre, Minho University, Guimarães, Portugal\
$^{2}$ - EPMQ, CCG ZGDV Institute, Guimarães, Portugal\
$^{3}$ - Bosch Car Multimedia, Braga, Portugal\
$^{\star}$ - correponding author (pcortez@dsi.uminho.pt)**

This project is licensed under the terms of the MIT license.
