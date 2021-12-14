# Install the enviornment by requirements.txt
conda install --file requirements.txt

# How to run
cd code && python main.py

# Folder Structure:

code: 程式碼

data: 資料
   |
   |-- DSC*.JPG: original pictures for a scene under different exposures
   |   
   |-- result.hdr: recovered HDR image
   |
   |-- joint_bilateral.jpg: tone-mapped image
   |
   |-- reinhard.jpg: tone-mapped Image