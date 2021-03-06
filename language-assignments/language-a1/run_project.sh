# Script for running everything :))
echo "setting up..."
bash setup.sh
echo "downloading data..."
pipenv run bash download_data.sh
echo "Running the scripts..."
echo "Collocating 'bleak' in 'Bleak House'..."
pipenv run python collocate_single_text.py --file-name "Dickens_Bleak_1853.txt" --search-term "bleak"
echo "collocate all texts for love..."
pipenv run python collocate_all_texts.py --window-size 3 --search-term "love" --test-mode
echo "finding both love and hate in 'Oliver Twist'..."
pipenv run python collocate_multi_words.py --file-name "Dickens_Oliver_1839.txt" --search-terms love hate --window-size 3
echo "creating a bleak plot..."
pipenv run python plot_bleak.py --file-path "output/Dickens_Bleak_1853_bleak.csv" 
echo "Done with everything!"

