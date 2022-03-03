# Assignment 1: Most similar images
Here are solutions for finding most similar images as described in `assignment1.md`. All solutions are python scripts with cmd-arguments. Documentation for each script can be found using the `--help` flag. In addition to the scripts, there is a jupyter notebook for experimenting with the different solutions. Most similar images are defined using chi-squared from the `compareHist`-function in `cv2`. Main packages used are `numpy`, `pandas`, and `cv2`. To increase processing speed, multiprocessing is also used. 

## Scripts
1. `find_similar_imgs.py`: a script for completing the basic task as described in `assignment1.md`. Takes a filename (the source image) as argument and, optionally, a directory and number of cores to use. Outputs an image with the 3 most similar images with their distances as well as a csv with similarities. 
2. `dir_similarity.py`: Find closest 3 images for every image in the supplied directory.

## Other files
- `experiments.ipynb`: code sketches and experiments

## TODO
- improve nsmallest performance (DONE)
- Add logging to `dir_similarity.py` (DONE)