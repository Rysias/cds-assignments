# Script for running everything :))
echo "installing dependencies"
pip install pipenv
python -m pipenv install
echo "Running main script!"
python -m pipenv run python transfer_cnn.py --epochs 10
echo "All done!"

