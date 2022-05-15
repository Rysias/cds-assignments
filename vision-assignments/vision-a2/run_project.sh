# Script for running everything :))
echo 'installing dependencies'
pip install pipenv
python -m pipenv install
echo "Running the scripts!..."
echo "Run logistic regression"
pipenv run python logistic_regression.py --dataset "cifar10"
echo "Run neural network"
pipenv run python nn_classifier.py --epochs 3 --dataset "cifar10"
echo "All done!"

