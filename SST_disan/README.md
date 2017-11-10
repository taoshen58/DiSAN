# DiSAN Implementation for SST
The introduction to Sentiment Classification task, please refer to [Stanford Sentiment Treebank (SST)](https://nlp.stanford.edu/sentiment/index.html) or paper.

## Python Package Requirements
* tqdm
* nltk

## Dataset
First of all, please clone repo and enter into project folder by:

	git clone https://github.com/forreview/DiSAN
	cd DiSAN/SST_disan

download data files into **dataset/** dir:

* [GloVe Pretrained word2vec](http://nlp.stanford.edu/data/glove.6B.zip)
* [SST dataset](http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip)

__Please check and ensure following files in corresponding folders for running under default hyper-parameters:__

* dataset/glove/glove.6B.300d.txt
* dataset/stanfordSentimentTreebank/datasetSentences.txt
* dataset/stanfordSentimentTreebank/datasetSplit.txt
* dataset/stanfordSentimentTreebank/dictionary.txt
* dataset/stanfordSentimentTreebank/original\_rt\_snippets.txt
* dataset/stanfordSentimentTreebank/sentiment_labels.txt
* dataset/stanfordSentimentTreebank/SOStr.txt
* dataset/stanfordSentimentTreebank/STree.txt

## 1. Run a Pre-trained Model to Verify the Result in Paper

### 1.1 Download Pre-processed Dataset and Pre-trained Model
Download URL is [here](https://drive.google.com/open?id=0B3Sd3TjOhd-JcnY4dkJOMFo0Ujg), **Please do not rename file after downloading!**.

#### 1.1.1 Download Pre-processed Dataset
* file name: *processed\_lw\_True\_ugut\_True\_gc\_6B\_wel\_300.pickle*
* Download file to *result/processed_data*

#### 1.1.2 Download Pre-trained Model File 

* two files: *disan\_sst\_model.ckpt.data-00000-of-00001* and *disan\_sst\_model.ckpt.index*
* Download files to folder *pretrained_model/*, and specify the path to running params `--load_path`


### 1.2 Run the code
	python sst_main.py --mode test --network_type disan --model_dir_suffix pretrained --gpu 0 --load_path pretrained_model/disan_sst_model.ckpt
	
__notice:__

* Please specify the GPU index in param `--gpu` to run the code on specified GPU. And if gpu is not avaliable, the code will be run on CPU automatically.
* if you dont have enough GPU memory, please feel free to change `--test_batch_size` whose default value is 128.
* [For tensorflow newcomer] The augument to `--load_path` does not need *.ckpt.data* or *.ckpt.index* as postfix, just *.ckpt*.

## 2. Train a Model
Just run codes as follows after preparing dataset, do not need to download any pre-processed files:

	python sst_main.py --mode train --network_type disan --model_dir_suffix training --gpu 0

__notice:__

* Everytime you running the code will build a folder *result/model/xx\_model\_name\_xxx/* whose name begin with augument of `--model_dir_suffix`, which includes running log in *log* folder, tensorflow summary in *summary* folder and top-3 models in *ckpt* folder.
* Please specify the GPU index in param `--gpu` to run the code on specified GPU. And if gpu is not avaliable, the code will be run on CPU automatically.
* if you dont have enough GPU memory, please feel free to change `--test_batch_size` whose default value is 128, and `--train_batch_size` whose default value is 64.
* After processing the raw dataset, a pickle file will be stored in *result/pocessed_data* which can be employed in following code running without processing the raw data for time saving. 
* The detail of parameters can be viewed in file `configs.py`.

### Test Your Trained Model
The training will take about 4 hours on single GTX1080Ti, and at the end of training models, top 3 model, including step number, dev accuracy and test accuracy, will be display in the bash window. You can also check the tensorflow checkpoint files in *result/model/xxxx/ckpt*, and run in test mode which introduced in sec.1.







