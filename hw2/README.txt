# Install the enviornment by requirements.txt
conda install --file requirements.txt

# How to run
cd code && python main.py

# Folder Structure:

code: 程式碼

data: 資料
   |
   |-- resized_*.jpg: original pictures of different angle for panoramas.
   |   
   |-- data.txt: data path and focal length for each images

result.png: 想要拿去投票的
result_non_cropped.png: 留有黑邊的結果