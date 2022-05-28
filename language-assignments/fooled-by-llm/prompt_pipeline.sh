# Add argument for model-name
MODEL_NAME=$1;
NUM_SENTENCES=2;
NUM_EACH=15;

echo "Running project for $MODEL_NAME"
python create_news_examples.py -ns $NUM_SENTENCES -na $NUM_EACH
python clean_news.py
python generate_news.py --model-name $MODEL_NAME
python clean_prompts.py --model-name $MODEL_NAME
python split_dataset.py --model-name $MODEL_NAME