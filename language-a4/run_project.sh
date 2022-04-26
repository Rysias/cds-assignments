# Script for running everything :))
echo "installing dependencies"
pip install pipenv
pipenv install --dev
echo "running tests"
pipenv run python -m pytest
echo "Running the scripts..."
echo "Run logistic regression..."
pipenv run python logistic_regression.py --dataset "../../CDS-VIS/toxic/VideoCommentsThreatCorpus.csv"
echo "Done! Running neural network..."
pipenv run python dnn_text.py --dataset "../../CDS-VIS/toxic/VideoCommentsThreatCorpus.csv" --epochs 50
echo "Done with everything!"

