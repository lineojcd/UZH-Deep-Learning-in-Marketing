# CycleGAN for recommender system

Implemented by Xiao'ao Song

-0 the data product is contained in output_my/test_imgs/70000.zip file.

-1 mycyclegan folder contains all the code you need to run the program. the required dependency packages are specified in mycyclegan/requirements.txt

-2 How to use the model: (It will automatically load the pre-trained weights and generates images on the given test set)
``` bash
python3 myCycleGAN_test.py
```

-3 How to train the model from scratch: 
``` bash
python3 myCycleGAN_train.py
```

-4 How to resume previous training?  Change the 'load_iter' setting to an iter number that you specified (in the myCycleGAN_train.py file)
``` bash
python3 myCycleGAN_train.py
```

-5 train the model by WGAN loss instead of LS loss: 
``` bash
python3 myCycleGAN_WGAN_train.py
```
 



## Changes we made:

##### 1 numbers of residual blocks: from 9 to 6
##### 2 Max reply buffeer size set from 50 to 100
##### 3 Permute the training set during each epoch to increase randomization.
##### 4 Use a sub training set (20000 images per collection) from original set (train A: 98552; train B: 113220) 


 
 
   
