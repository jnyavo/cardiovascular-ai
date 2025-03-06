import os
import requests
import wfdb

from data import ptbxl

def dowload_dataset_with_wget(url: str, database_name: str | None = None):
  """
    Download a file from a given URL using wget.
    Args:
        url (str): The URL of the file to download.
        database_name (str | None): The name of the database to download.
    Returns:
        None
  """
  current_dir = os.getcwd()
  destination_dir = os.path.join( current_dir ,"dataset", database_name if database_name else "")

  os.makedirs(destination_dir, exist_ok=True)

  os.system(f"wget -r -N -c -np -P {destination_dir} {url}")




def dowload_dataset(url: str, dataset_name: str | None = None):
  """
    Download a file from a given URL and save it to a specified directory.
    Args:
        url (str): The URL of the file to download.
    Returns:
        None
  """	
  
  folder = "dataset"
  os.makedirs(folder, exist_ok=True)

  name = dataset_name if dataset_name else os.path.basename(url)

  file_path = os.path.join(folder, name)

  response = requests.get(url)

  if response.status_code == 200:
      with open(file_path, "wb") as f:
          f.write(response.content)
      print(f"File downloaded and saved as {file_path}")
  else:
      print(f"Failed to download file. Status code: {response.status_code}")




def download_dataset_from_physionet(database_name: str):
  """
    Download the dataset from PhysioNet.
    Args:
        database_name (str): The name of the database to download.
    Returns:
        None
  """	
  current_dir = os.getcwd()
  full_path = os.path.join(current_dir, "dataset")
  os.makedirs(full_path, exist_ok=True)
  wfdb.dl_database(database_name, dl_dir=full_path, overwrite=True)




if __name__ == "__main__":
    database = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
    dowload_dataset_with_wget(database)