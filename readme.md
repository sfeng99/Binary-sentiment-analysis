# The-state-of-art models for Binary sentiment analysis

## Training data preparation

Put your training data (a tab separated file .tsv) to glue_data/SST-2/. The data should have two columns 'sentence' which contain the original text and 'label', which have two type of value, 0 or 1. In the directory, there already has example training dataset train.tsv and you can replace it with your own one.

## Fine tunning with the-state-of-art model

Run the bash file for whichever models you like by changing the parameters **--model_type** and **--model_name_or_path**. Your can refer [here](https://huggingface.co/transformers/pretrained_models.html) for more detailed information of these two parameters. And this [website](https://paperswithcode.com/sota/sentiment-analysis-on-sst-2-binary) is the rank information on similar task.

```
export GLUE_DIR=./glue_data/
export TASK_NAME=SST-2

python run_glue.py \
  --model_type albert \
  --model_name_or_path albert-xxlarge-v2 \
  --task_name $TASK_NAME \
  --do_train \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir /tmp/$TASK_NAME/
```

Note that the model will be automatically saved to /tmp/SST-2 directory. 

Then run the script to train the model

```
bash run_glue.sh
```



## Prediction

Change the model_type you use at initializer of class **SentimentAnalyzer** in predict.py. There are two methods you can do prediction.

### 1. Predict single sentence

```
from predict import SentimentAnalyzer
clf = SentimentAnalyzer()
print(clf.predict('The movie is terribly boring in places. '))
```

### 2. Predict from .txt file

Each line in .txt file is a sentence you want to predict. Put the file in glue_data/SST-2/ and run the following command

```
python predict.py
```

You will get 'yproba1_test.txt', each line of which is probability that the model predict the sentence to label 1.

# Binary-sentimental-analysis
