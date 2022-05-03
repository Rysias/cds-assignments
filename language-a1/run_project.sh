# Script for running everything :))
echo "setting up..."
bash setup.sh
echo "Running the scripts..."
echo "collocate single text"
pipenv run python collocate_single_text.py --file-name "Dickens_Bleak_1853.txt" --search-term "bleak"
echo "Done with everything!"

