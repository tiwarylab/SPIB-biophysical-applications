# SPIB-biophysical-applications
Application of SPIB to study protein conformational dynamics and membrane permeation.

Files in the SPIB-inputs directory are configuration files in INI format for SPIB (https://github.com/tiwarylab/State-Predictive-Information-Bottleneck). Example command: 
```
python test_model_advanced.py -config examples/sample_config.ini
```

relevance_score.py calculates the relavance of different order parameters recorded in the BA-DMPC trajectory. The script requires NumPy, PyTorch, os, SPIB python libraries installed and have been tested for SPIB (ver July2021).
