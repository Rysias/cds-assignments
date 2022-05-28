echo "### SETTING UP ENVIRONMENT ###"
bash setup.sh
echo "### DOWNLOADING DATA ###"
pipenv run bash download_data.sh
echo "### RUNNING PROJECT ###"
pipenv run bash prompt_pipeline.sh "gpt-j-6b"
