# Script for running everything :))
echo "setting up..."
bash setup.sh
echo "Running the script with VADER..." 
python process_news.py --sentiment vader
echo "Running the script with textblob..." 
python process_news.py --sentiment textblob

