# Script for running everything :))
echo "installing dependencies"
pip install pipenv
pipenv install
echo "Download data"
python download_data.py
echo "Running the scripts!..."
echo "Run single image similarity"
pipenv run python find_similar_imgs.py --img-name "image_0003.jpg"
echo "calculate all image similarities"
pipenv run python dir_similarity.py 
echo "All done!"

