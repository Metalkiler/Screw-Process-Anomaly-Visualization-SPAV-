# Screw-Process-Anomaly-Visualization-SPAV-


Following the Industrial Revolutions 4.0 and 5.0, the assembly industry faces mounting pressure to reduce costs and boost efficiency using AI‑driven solutions. Modern screwdriver systems generate real‑time angle‑torque data that form tightening curves, which must be frequently recalibrated and inspected to prevent defective assemblies. We propose modeling normal screwing profiles with unsupervised AutoEncoders to detect deviations indicative of faults, complemented by intuitive visual analytics. Our Python module, Screw Process Anomaly Visualization (SPAV), offers four key plots—Global Error, Local Error, Detected Anomalies, and Anomaly Density via KDE—to support operators in understanding and validating anomaly detection. PAV integrates seamlessly with the Scientific Python ecosystem and is compatible with H2O and Keras AE models.  


## Purpose and features

Following the recent industrial revolutions (Industry 4.0 and 5.0), several assembly companies are currently under pressure to lower costs and increasing operational efficiency through Artificial Intelligence (AI) tools. 
In particular, screwdriver systems are often used in the assembly industry (e.g., electronic components), generating hundreds of angle-torque real-time pair values for a single screw tightening. Then, the full tightening curve can be analyzed for quality assessment purposes (e.g., anomaly detection). 
The screw tightening process is inherently dynamic, requiring frequent updates to defect catalogs and recalibration of screwing zones through human evaluation of tightening curves and quality estimates. Although defects are rare, detailed and automated inspection is essential to prevent faulty assemblies from progressing along the production line. 

Currently, screwdriver systems provide a "Good or Fail" (GoF) status based on predefined normal screw historical catalogs, making it susceptible of generating false positives and false negatives.



## Example (codeocean executable run)

https://doi.org/10.24433/CO.1214166.v1



## Authors
Marta Moreno, Hugo Rocha, André Pilastri, Guilherme Moreira, Luís Miguel Matos and Paulo Cortez
