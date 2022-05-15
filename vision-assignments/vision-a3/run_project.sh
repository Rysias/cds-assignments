# Script for running everything :))
echo "installing dependencies"
bash setup.sh
echo "Running main script!"
python -m pipenv run python transfer_cnn.py --epochs 10
echo "All done!"

