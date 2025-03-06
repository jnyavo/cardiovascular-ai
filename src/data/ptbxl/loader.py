import ast
import os
import pandas as pd
import wfdb
import numpy as np
from data.ptbxl.enums import SamplingRate, Lead
from data.ptbxl.record import Record
from data.ptbxl.utils import aggregate_diagnostic

class PTBXL:
  """
  PTB-XL ECG Dataset Loader

  This class provides functionality to load and work with the PTB-XL ECG dataset.
  PTB-XL is a large publicly available electrocardiography dataset containing 21,837 clinical 12-lead ECGs 
  from 18,885 patients of 10 second length.

  The dataset contains signals sampled at both 100Hz and 500Hz, and this loader can work with either sampling rate.
  Each record contains 12 standard leads (I, II, III, aVR, aVL, aVF, V1-V6).

  Key Features:
  - Supports loading both 100Hz and 500Hz sampled data
  - Provides methods to load individual records or all records
  - Includes metadata handling
  - Visualization capabilities for ECG signals

  Example Usage:
      ptbxl = PTBXL("path/to/ptbxl", sampling_rate=SamplingRate.HZ_100)
      record = ptbxl.load_record(1)
      record.draw_ecg(lead=Lead.I)  # Draws lead I of the ECG
  """
  def __init__(self, base_path='dataset/ptb-xl', sampling_rate: SamplingRate=SamplingRate.HZ_100, metadata_filename='ptbxl_database.csv', scp_file="scp_statements.csv"):
    """
    Initialize the PTBXL loader.

    Parameters:
    - base_path (str): Local path where the PTB‑XL dataset is stored.
    - sampling_rate (str): Which folder to load from; e.g. '100' or '500'.
    - metadata_filename (str): Filename of the metadata CSV (e.g., ptbxl_database.csv).
    """
    current_path = os.getcwd()
    full_path = os.path.join(current_path,base_path, f"records{sampling_rate}" )
    if not os.path.exists(full_path):
        raise ValueError(f"Path {full_path} does not exist. Please check your base_path and sampling_rate.")
    self.full_path = full_path
    self.sampling_rate = sampling_rate
    self.metadata_path = os.path.join(base_path, metadata_filename)

    metadata_path = os.path.join(base_path, metadata_filename)
    
    # Load metadata CSV into a DataFrame.
    metadata = pd.read_csv(metadata_path, index_col='ecg_id')

    # Convert scp_codes from string representation of list to actual list
    metadata.scp_codes = metadata.scp_codes.apply(lambda x: ast.literal_eval(x))
    
    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(os.path.join(base_path, scp_file), index_col=0)
    
    # Filter for diagnostic records
    agg_df = agg_df[agg_df.diagnostic == 1]

    # Apply diagnostic superclass
    metadata['diagnostic_superclass'] = metadata.scp_codes.apply(lambda y: aggregate_diagnostic(y, agg_df))
    
    self.metadata = metadata
  
  def get_ptbxl_data_path(self, ecg_id):
    return os.path.join(
        f"{int(ecg_id / 1000) * 1000 :05d}",
        f"{ecg_id:05d}_{'lr' if self.sampling_rate == SamplingRate.HZ_100 else 'hr'}"
    )
  
  def _load_raw_data(self):
    df = self.metadata
    if self.sampling_rate == SamplingRate.HZ_100:
        data = [wfdb.rdsamp(os.path.join(self.full_path, f)) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(os.path.join(self.full_path, f)) for f in df.filename_hr]
    data = np.array([signal for signal, _ in data])
    return data
  
  def load_record(self, id: int):
    """
    Load a single ECG record using WFDB.

    Parameters:
    - record_name (int): id of the record.

    Returns:
    - wfdb.Record: The loaded record, or None if there's an error.
    """
    record_folder = self.full_path
    record_path = os.path.join(record_folder, self.get_ptbxl_data_path(id))
    record = Record(wfdb.rdsamp(record_path))
    return record

  def _aggregate_diagnostic(y_dic: dict, agg_df: pd.DataFrame):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

  def load_all(self):
      """
      Load all ECG records listed in the metadata CSV.

      Returns:
      - X (np.ndarray of arrays): Array containing the ECG signals.
      - Y (list): List of diagnostic labels corresponding to each record.
      
      Note: Loading all records may be time consuming and memory intensive.
      """
      X = []
      Y = []
      for idx, row in self.metadata.iterrows():
          record_id = row['filename']  # Assumes the record identifier is in the "filename" column.
          record = self.load_record(record_id)
          if record is not None:
              # record.p_signal contains the ECG waveform (as a 2D array: samples x channels).
              X.append(record.data)
              # Here we assume labels are in the "scp_codes" column. Modify if your metadata differs.
              Y.append(row['scp_codes'])
      # Using an object dtype array for X as the signals can have varying lengths.
      return np.array(X, dtype=object), Y

  @property
  def X(self):
      """
      Returns:
      - np.ndarray: All loaded ECG signals.
      
      Warning: Accessing X loads all records, which can be slow and memory intensive.
      """
      X, _ = self.load_all()
      return X

  @property
  def Y(self):
      """
      Returns:
      - list: All labels corresponding to the ECG records.
      
      Warning: Accessing Y loads all records, which can be slow and memory intensive.
      """
      _, Y = self.load_all()
      return Y

# Example usage:
if __name__ == "__main__":
  ptbxl = PTBXL()
  record = ptbxl.load_record(1)
  record.draw_ecg(Lead.I)
  record.draw_ecg(Lead.II)
