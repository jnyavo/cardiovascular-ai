"""
PTB-XL ECG Dataset Package

This package provides functionality to load and work with the PTB-XL ECG dataset.
PTB-XL is a large publicly available electrocardiography dataset containing 21,837 clinical 12-lead ECGs
from 18,885 patients of 10 second length.

The dataset contains signals sampled at both 100Hz and 500Hz, and this loader can work with either sampling rate.
Each record contains 12 standard leads (I, II, III, aVR, aVL, aVF, V1-V6).

Example Usage:
    from data.ptb_xl import PTBXL
    ptbxl = PTBXL("path/to/ptbxl", sampling_rate=SamplingRate.HZ_100)
    record = ptbxl.load_record(1)
    record.draw_ecg(lead=Lead.I)  # Draws lead I of the ECG
"""

from .enums import Lead, SamplingRate

from .loader import PTBXL