# nuclei segmentation using double-unet-aspp


## 1 Overview
### Task discription
This project is to record the whole process of using a two-stage method based on Unet model.
The main target is to get good semantic segmentation results of nuclei segmention.
We want to try to add ASPP, attention module, weight maps and boundary information to improve the final performance.
For doing instance segmentation, we will add stardist into the network to segment object by object.
Here are several attempts.

### Main work
1. utilize aspp
2. modify aspp
3. add weight map
4. use stardist

### Todo
- [x] Unet + ASPP
- [ ] Unet + Dense ASPP
- [x] Attention Unet + ASPP
- [ ] Unet + Attention + ASPP (Note: try se block, self attention and other attention block)
- [ ] Unet + Attention + Dense ASPP
- [x] Double Unet
- [x] Double Unet + ASPP
- [x] Double Attention Unet
- [x] Double Attention Unet + ASPP
- [ ] Double Unet + Attention + ASPP (Note: try se block, self attention and other attention block)
- [ ] Double Unet + Dense ASPP
- [ ] Double Unet + Attention + Dense ASPP
- [ ] Unet + Weight maps

## 2 Requirements
### Dependencies
- pytorch 1.11.0
- python 3.8.5
- Numpy, Matplotlib
- PIL, Skimage, OpenCV

### Dataset URL
- MoNuSeg 2018 (https://monuseg.grand-challenge.org/Data/)
- ER-IHC dataset (private dataset)

## 3 Directory structure

Note:
```train.py``` and ```predict.py``` are used for one-stage methods.
```train_1st.py```, ```train_2nd.py```, ```predict_1st.py``` and ```predict_2nd.py``` are used for two-stage methods. We can check the performance and save weights for the first stage. If the first stage is not paid attention, you can directly use the file with '_2nd'.


## 4 Train the model
Before starting training a model, the following parameters should be specified:
- model_name: which model will be used in this training
- model_detail: do more discriptions about this training need to be added
- loss_type: which loss will be used

To train a model
1. modify the parameters in ```cfg.py```
2. run the comment: ```python train.py```


## 5 Making predicition
To predict images
1. modify the parameters in ```cfg.py```
2. run the comment: ```python predict.py```


## 6 Post-processing

## 7 Results
### get common score
If you want to get common score, you can use ```evaluation.py```. This file includes calculating f1, iou, accuracy, precision and recall. Make sure the path in ```config.py``` is correct. And then run ```python evaluation.py```.

### comparison result  (ER-IHC dataset)
|method|iou|precision|recall|f1|
| --- | --- | --- | --- | --- |
|Unet|0.7510|0.8443|0.8748|0.8541|                       # PPT: |0.7312|0.8459|0.8426||0.5792|
|NestedUnet|0.7635|0.8872|0.8468|0.8637|                 # PPT: |0.7379|0.8391|0.8527|0.8479|0.5727|
|AttentionUnet|0.7721|0.8782|0.8657|0.8692|
|Unet+ASPP|0.7694|0.8734|0.8668|0.8673|
|NestedUnet+ASPP|0.7662|0.8951|0.8419|0.8656|
|AttentionUnet+ASPP|0.7721|0.8853|0.8582|0.8693|
|DoubleUnet(w/o 1st train)|0.7740|0.8939|0.8519|0.8710|   # Note: (w/o 1st train) means using sec_train.py directly to train the whole network, without using pretrained weights from fir_train.py
|DoubleUnet+ASPP(w/o 1st train)|0.7740|0.8773|0.8677|0.8704|
|DoubleAttentionUnet(w/o 1st train)|0.7646|0.8774|0.8579|0.8644|
|DoubleAttentionUnet+ASPP(w/o 1st train)|0.7611|0.8552|0.7334|0.8615|


### problem summary
1. noises (how to supress the noises)
2. 拼接问题


## Reference
1. DoubleUnet: A Deep Convolutional Neural Network for Medical Image Segmentation

Paper Link: https://arxiv.org/pdf/2006.04868.pdf

code Link: 
1. https://github.com/DebeshJha/2020-CBMS-DoubleU-Net/tree/44b69d60a2b8385dabaf123883f493d29774051c (Tensorflow)
2. https://github.com/AdamMayor2018/DoubleUnet-pytorch-implementation/tree/e846fb72ed90cd23bcde3eb569b8f14664424a48 (Pytorch)





