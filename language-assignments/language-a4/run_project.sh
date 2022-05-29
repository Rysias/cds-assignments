# Script for running everything :))
echo "installing dependencies"
bash setup.sh
echo "## Running the scripts... ##"
echo "augmenting data"
pipenv run python augment_data.py
echo "Run logistic regression..."
pipenv run python logistic_regression.py --dataset "input/VideoCommentsThreatCorpus.csv"
echo "Done! Running neural network..."
pipenv run python dnn_text.py --epochs 20 --dropout 0.1
echo "Done with everything!"
cat output/dnn_text.txt

