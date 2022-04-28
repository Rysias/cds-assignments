# Script for running everything :))
echo "installing dependencies"
pip install pipenv
pipenv install
echo "Running the scripts!..."
echo "Run single image similarity"
pipenv run python find_similar_imgs.py --img-name "testimg.jpg"
echo "calculate all image similarities"
pipenv run python dir_similarity.py 
echo "All done!"

