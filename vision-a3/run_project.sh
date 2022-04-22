# Script for running everything :))
echo "installing dependencies"
pip install pipenv
pipenv install
echo "Running main script!"
pipenv run python transfer_cnn.py
echo "All done!"

