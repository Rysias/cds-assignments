# Script for running everything :))
echo "installing dependencies"
pip install pipenv
pipenv install --dev
pipenv run python -m pytest
echo "Running the script!..."
pipenv run python network_analysis.py --data-path "input"

