# Script for running everything :))
echo "installing dependencies"
pip install pipenv
pipenv install --dev
echo "running tests"
pipenv run python -m pytest
echo "Running the scripts..."
echo "Run logistic regression..."
pipenv run python logistic_regression.py
echo "Done! Running neural network..."
pipenv run python dnn_text.py --epochs 50
echo "Done with everything!"

