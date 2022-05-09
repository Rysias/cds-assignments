# Script for running everything :))
echo "setting up..."
bash setup.sh
echo "Running the scripts..."
echo "Collocating 'bleak' in 'Bleak'..."
python collocate_single_text.py --file-name "Dickens_Bleak_1853.txt" --search-term "bleak"
echo "collocate all texts for love..."
python collocate_all_texts.py --window-size 3 --search-term "love"
echo "finding both love and hate in Dickens..."
python collocate_multi_words.py --search-terms love hate --window-size 3
echo "Done with everything!"

