# Script for running everything :))
echo "installing dependencies"
pip install pipenv
pipenv install
echo "Running the scripts!..."
echo "Run logistic regression"
pipenv run python logistic_regression.py --dataset "cifar10"
echo "Run neural network"
pipenv run python nn_classifier.py --epochs 3 --dataset "mnist"
echo "All done!"

