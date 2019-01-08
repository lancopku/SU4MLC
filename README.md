# Semantic-Unit-for-Multi-label-Text-Classification
Code for the article "Semantic-Unit-Based Dilated Convolution for Multi-Label Text Classification" (EMNLP 2018).

***********************************************************

## Requirements
* Ubuntu 16.0.4
* Python 3.5
* Pytorch 0.4.1 (updated)

**************************************************************

## Data
Our preprocessed RCV1-V2 dataset can be retrieved through [this link](https://drive.google.com/open?id=1oQ5_gPoRwAl7UGWTDNu4qATNtJ1l1kXd). (The json file of label set for evaluation is added for convenience.)

***************************************************************

## Preprocessing
```
python3 preprocess.py -load_data path_to_data -save_data path_to_store_data (-src_filter 500)
```
Remember to put the data (plain text file) into a folder and name them *train.src*, *train.tgt*, *valid.src*, *valid.tgt*, *test.src* and *test.tgt*, and make a new folder inside called *data*. 

***************************************************************

## Training
```
python3 train.py -log log_name -config config_yaml -gpus id (-label_dict_file path to your label set)
```
Create your own yaml file for hyperparameter setting.

****************************************************************

## Evaluation
```
python3 train.py -log log_name -config config_yaml -gpus id -restore checkpoint -mode eval
```

*******************************************************************

# Citation
If you use this code for your research, please kindly cite our paper:
```
@inproceedings{DBLP:conf/emnlp/LinSYM018,
  author    = {Junyang Lin and
               Qi Su and
               Pengcheng Yang and
               Shuming Ma and
               Xu Sun},
  title     = {Semantic-Unit-Based Dilated Convolution for Multi-Label Text Classification},
  booktitle = {Proceedings of the 2018 Conference on Empirical Methods in Natural
               Language Processing, Brussels, Belgium, October 31 - November 4, 2018},
  pages     = {4554--4564},
  year      = {2018}
}
```

