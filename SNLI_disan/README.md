# DiSAN Implementation for SNLI
The introduction to NLI task, please refer to [Stanford Natural Language Inference (SNLI)](https://nlp.stanford.edu/projects/snli/) or paper.

## Python Package Requirements
* tqdm
* nltk

## Dataset

First of all, please clone repo and enter into project folder by:

	git clone https://github.com/forreview/DiSAN
	cd DiSAN/SNLI_disan

download data files into **dataset/** dir:

* [GloVe Pretrained word2vec](http://nlp.stanford.edu/data/glove.6B.zip)
* [SNLI dataset](https://nlp.stanford.edu/projects/snli/snli_1.0.zip)

__Please check and ensure following files in corresponding folders for running under default hyper-parameters:__

* dataset/glove/glove.6B.300d.txt
* dataset/snli\_1.0/snli\_1.0\_train.jsonl
* dataset/snli\_1.0/snli\_1.0\_dev.jsonl
* dataset/snli\_1.0/snli\_1.0\_test.jsonl

## 1. Run a Pre-trained Model to Verify the Result in Paper

### 1.1 Download Pre-processed Dataset and Pre-trained Model
Download URL is [here](https://drive.google.com/drive/folders/0B3Sd3TjOhd-JNjJNT2RoZU1NalU?usp=sharing), **Please do not rename file after downloading!**.

#### 1.1.1 Download Pre-processed Dataset
* file name: *processed\_lw\_True\_ugut\_True\_gc\_6B\_wel\_300\_slr\_0.97\_dcm\_no\_tree.pickle*
* Download file to *result/processed_data*

#### 1.1.2 Download Pre-trained Model File 

* two files: *disan\_snli\_model.ckpt.data-00000-of-00001* and *disan\_snli\_model.ckpt.index*
* Download files to folder *pretrained_model/*, and specify the path to running params `--load_path`


### 1.2 Run the code

	python snli_main.py --mode test --network_type disan --model_dir_suffix pretrained --gpu 0 --load_path pretrained_model/disan_snli_model.ckpt
	
__notice:__

* Please specify the GPU index in param `--gpu` to run the code on specified GPU. And if gpu is not avaliable, the code will be run on CPU automatically.
* The consuming memory is about 9GB with `--test_batch_size` set to 100, if your device have limited memory, please try to change `--test_batch_size` to smaller.
* [For tensorflow newcomer] The augument to `--load_path` does not need *.ckpt.data* or *.ckpt.index* as postfix, just *.ckpt*.

## 2. Train a Model
Just run codes as follows after preparing dataset, do not need to download any pre-processed files:

	python snli_main.py --mode train --network_type disan --model_dir_suffix training --gpu 0

__notice:__

* Everytime you running the code will build a folder *result/model/xx\_model\_name\_xxx/* whose name begin with augument of `--model_dir_suffix`, which includes running log in *log* folder, tensorflow summary in *summary* folder and top-3 models in *ckpt* folder.
* Please specify the GPU index in param `--gpu` to run the code on specified GPU. And if gpu is not avaliable, the code will be run on CPU automatically.
* The consuming memory is about 9GB with `--test_batch_size` set to 100, if your device have limited memory, please try to change `--test_batch_size` into smaller. In addition, with `--train_batch_size` set to 64, the minimum consuming memory is 5GB, you can also change it to smaller.
* After processing the raw dataset, a pickle file will be stored in *result/pocessed_data* which can be employed in following code running without processing the raw data for time saving. 
* The detail of parameters can be viewed in file `configs.py`.
* There are also baseline neural networks provided which are appeared in paper, respectively `exp_emb_attn`, `exp_emb_mul_attn`, `exp_bi_lstm_mul_attn`, `exp_emb_self_mul_attn` with the same order in paper, so you can pass one of these model name to `-network_type` to run baseline experiments.

### Test Your Trained Model
The training will take about 15 hours on single GTX1080Ti, and at the end of training models, top 3 model, including step number, dev accuracy and test accuracy, will be display in the bash window. You can also check the tensorflow checkpoint files in *result/model/xxxx/ckpt*, and run in test mode which introduced in sec.1.







