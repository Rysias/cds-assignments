echo "### SETTING UP ENVIRONMENT ###"
bash setup.sh
echo "### DOWNLOADING DATA ###"
pipenv run python download_data.py
echo "### RUNNING PROJECT ###"
pipenv run bash prompt_pipeline.sh "gpt-j-6b"
