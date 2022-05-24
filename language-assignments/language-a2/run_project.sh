# Script for running everything :))
echo "setting up..."
bash setup.sh
echo "Running the script with VADER..." 
pipenv run python process_news.py --sentiment vader
echo "Running the script with textblob..." 
pipenv run python process_news.py --sentiment textblob
echo "plotting comparison..."
pipenv run python plot_sentiments.py
