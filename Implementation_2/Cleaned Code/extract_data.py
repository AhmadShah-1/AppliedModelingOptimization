# header of the script with a description of its purpose and the author
"""
This script is designed to extract data from a compressed file (diabetes-data.tar.Z) and save the contents as clean CSV files.
Author: Chris Ognibene
"""
# This script performs the following tasks:
# 1. It imports necessary libraries for handling compressed files, file paths, and data manipulation.
# 2. It sets the directory path for the compressed file and the output directory for the extracted data.
# 3. It checks if the directory for extracted data exists, and if not, it creates it.
# 4. It extracts the compressed file, reads the contents into a dataframe, and saves it as clean CSV files in the specified output directory.


# import necessary libraries for handling compressed files, file paths, and data manipulation
import tarfile 
from pathlib import Path 
import ncompress 
import io 
import pandas as pd 

# set the directory path for the compressed file and the output directory for the extracted data
base_path = Path("Cleaned Code").parent
extract_file_path = base_path / "data" / "diabetes" / "diabetes-data.tar.Z"

# check if directory for extract data exists, if not create it
if Path(base_path / "data" / "extracted_data").is_dir():
    print("Extracted data directory exists")
else:
    Path(base_path / "data" / "extracted_data").mkdir(parents=True, exist_ok=True)

# check if directory for transformed data exists, if not create it
if Path(base_path / "data" / "transformed_data").is_dir():
    print("Transformed data directory exists")
else:
    Path(base_path / "data" / "transformed_data").mkdir(parents=True, exist_ok=True)


# extract the compressed file and read the contents into a dataframe, then save as clean CSV files
with open(extract_file_path, "rb") as f:
    compressed_data = f.read()
    decompressed_data = ncompress.decompress(compressed_data)
    
    # Ensure mode="r" is explicitly set for the memory-backed file object
    with tarfile.open(fileobj=io.BytesIO(decompressed_data), mode="r") as tar:
        for i, member in enumerate(tar.getmembers()):
            if member.isfile():
                f = tar.extractfile(member)
                
                # Read into a dataframe (adjust 'sep' if it's tabs or semicolons)
                df = pd.read_table(f, sep='\t', header=None, names=['Date', 'Time', 'Code', 'Value'])
                
                # Save as a clean CSV
                clean_name = member.name.replace("/", "_") + ".csv"
                df.to_csv(base_path / "data" / "extracted_data" / clean_name, index=False)

# create the transformed data from the extracted data by unstacking the Code column as individual columns per code and filling in the values from the Value column, then save as a clean CSV
# only for files 1-70 since those are the patient data files

for i, file in enumerate(Path(base_path / "data" / "extracted_data").glob("*.csv")):
    df = pd.read_csv(file)
    
    if i in range(0,70):
        df['PATIENT_ID'] = i+1
        df.set_index('PATIENT_ID', inplace=True)
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

        # Save transformed data
        clean_name = f"{file.name.replace('.csv', '')}_transformed.csv"
        df.to_csv(base_path / "data" / "transformed_data" / clean_name, index=True)