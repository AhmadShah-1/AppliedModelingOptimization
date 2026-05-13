# header of the script with a description of its purpose and the author
"""
This script is designed to transform the extracted diabetes data by unstacking the Code column as individual columns per code and filling in the values from the Value column, then saving it as a clean CSV file.
Author: Chris Ognibene
"""
# This script performs the following tasks:
# 1. It imports necessary libraries for handling compressed files, file paths, and data manipulation.
# 2. It sets the directory path for the compressed file and the output directory for the extracted data.
# 3. It checks if the directory for extracted data exists, and if not, it creates it.
# 4. It extracts the compressed file, reads the contents into a dataframe, and saves it as clean CSV files in the specified output directory.


# import necessary libraries for handling compressed files, file paths, and data manipulation
from pathlib import Path 
import pandas as pd 

# set the directory path for the compressed file and the output directory for the extracted data
base_path = Path("Cleaned Code").parent

transformed_files_path = base_path / "data" / "transformed_data"

if Path(base_path / "data" / "final_data").is_dir():
    print("Final data directory exists")
else:
    Path(base_path / "data" / "final_data").mkdir(parents=True, exist_ok=True)

concat_data = pd.DataFrame()

# check if directory for transformed data exists, if not create it
if Path(base_path / "data" / "transformed_data").is_dir():

    for i, file in enumerate(transformed_files_path.glob("*.csv")):
        df = pd.read_csv(file, index_col=0)
        
        concat_data = pd.concat([concat_data, df], axis=0)

        # Save concatenated data as a clean CSV
        clean_name = "final_cleaned_data.csv"
        concat_data.to_csv(base_path / "data" / "final_data" / clean_name, index=True)

else:
    print(f"{Path(base_path / "transformed_data")} directory does not exist.")
