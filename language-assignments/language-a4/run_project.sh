# Script for running everything :))
echo "installing dependencies"
bash setup.sh
echo "Running the scripts..."
echo "Run logistic regression..."
pipenv run python logistic_regression.py --dataset "../../../CDS-LANG/toxic/VideoCommentsThreatCorpus.csv"
echo "Done! Running neural network..."
pipenv run python dnn_text.py --dataset "input/VideoCommentsThreatCorpus.csv" --epochs 5
echo "Done with everything!"

