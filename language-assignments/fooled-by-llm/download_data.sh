ZIP_PATH="data/raw/data.zip"
gdown 1IoTRrJNDJqvaG3hnUpnHQyGvPAJbO8y3 --output $ZIP_PATH
unzip $ZIP_PATH -d data/raw
rm $ZIP_PATH
