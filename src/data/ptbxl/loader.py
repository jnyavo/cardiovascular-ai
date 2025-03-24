import ast
import os
from joblib import dump, load
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
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
  def __init__(self, base_path='dataset/ptb-xl', sampling_rate: SamplingRate=SamplingRate.HZ_100, \
               metadata_filename='ptbxl_database.csv', scp_file="scp_statements.csv", label_encoder_path="bin/label_encoder.bin", multi_threaded=False):
    """
    Initialize the PTBXL loader.

    Parameters:
    - base_path (str): Local path where the PTB‑XL dataset is stored.
    - sampling_rate (str): Which folder to load from; e.g. '100' or '500'.
    - metadata_filename (str): Filename of the metadata CSV (e.g., ptbxl_database.csv).
    - scp_file (str): Filename of the SCP statements CSV (e.g., scp_statements.csv).
    - label_encoder_path (str): Path to save/load the label encoder.
    - multi_threaded (bool): Whether to load data using multiple threads or not.
    """
    current_path = os.getcwd()
    self.multi_threaded = multi_threaded
    full_path = os.path.join(current_path,base_path, f"records{sampling_rate}" )
    if not os.path.exists(full_path):
        raise ValueError(f"Path {full_path} does not exist. Please check your base_path and sampling_rate.")
    self.full_path = full_path
    self.base_path = os.path.join(current_path, base_path)
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
    metadata["superdiagnostic_len"] = metadata["diagnostic_superclass"].apply(lambda x: len(x))
    counts = pd.Series(np.concatenate(metadata.diagnostic_superclass.values)).value_counts()
    metadata["diagnostic_superclass"] = metadata["diagnostic_superclass"].apply(
        lambda x: list(set(x).intersection(set(counts.index.values)))
    )
    
    self.metadata = metadata
    self.label_encoder_path = os.path.join(current_path, label_encoder_path)
  
  def get_ptbxl_data_path(self, ecg_id):
    return os.path.join(
        f"{int(ecg_id / 1000) * 1000 :05d}",
        f"{ecg_id:05d}_{'lr' if self.sampling_rate == SamplingRate.HZ_100 else 'hr'}"
    )
  
  def _load_raw_data(self):
    df = self.metadata
    file_pattern = 'filename_lr' if self.sampling_rate == SamplingRate.HZ_100 else 'filename_hr'
    print("Loading dataset using single thread...", end="\n" * 2)
    return np.array([wfdb.rdsamp(os.path.join(self.base_path, f))[0] for f in df[file_pattern]])
  
  def _load_raw_data_mp(self):
    """Load all raw ECG data from the dataset using multiple threads.
    
    Returns:
        np.ndarray: Array containing all ECG signals
    """
    df = self.metadata
    print("Loading dataset using multiple threads...", end="\n" * 2)
    file_pattern = 'filename_lr' if self.sampling_rate == SamplingRate.HZ_100 else 'filename_hr'
    file_paths = [os.path.join(self.base_path, f) for f in df[file_pattern]]
    
    # Use parallel processing to speed up loading
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial
    
    def load_single_record(path):
      return wfdb.rdsamp(path)[0]
  
    
    # Load records in parallel using multiple threads
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        data = list(executor.map(load_single_record, file_paths))
    
    return np.array(data)
    
  
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
    record = Record(wfdb.rdsamp(record_path), self.metadata.loc[id].diagnostic_superclass, self.metadata.loc[id].scp_codes)
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
      - Y (np.ndarray): Array containing the labels corresponding to the ECG records.
      - Y_data (pd.DataFrame): DataFrame containing the metadata for the records.
      - classes (list): List of unique diagnostic classes.
      
      Note: Loading all records may be time consuming and memory intensive.
      """

      metadata = self.metadata
      data = self._load_raw_data_mp() if self.multi_threaded else self._load_raw_data()

      # Filter metadata for records with at least one diagnostic superclass
      X_data = data[metadata["superdiagnostic_len"] >= 1]
      Y_data = metadata[metadata["superdiagnostic_len"] >= 1]

      mlb: MultiLabelBinarizer
      if os.path.exists(self.label_encoder_path):
          print(f"Loading label encoder from {self.label_encoder_path}")
          mlb = load(self.label_encoder_path)
      else:
          mlb = MultiLabelBinarizer()
          mlb.fit(Y_data["diagnostic_superclass"])
          # Save the fitted label encoder
          dump(mlb, self.label_encoder_path)
      y = mlb.transform(Y_data["diagnostic_superclass"].values)

      return X_data, y, Y_data, mlb.classes_

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

if __name__ == "__main__":
  ptbxl = PTBXL()
  record = ptbxl.load_record(1)
  record.draw_ecg(Lead.I)
  record.draw_ecg(Lead.II)
