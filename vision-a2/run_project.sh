# Script for running everything :))
echo "installing dependencies"
pip install pipenv
pipenv install
echo "Running the example script!..."
pipenv run python network_analysis.py --data-path "../../../CDS-LANG/network_data"

