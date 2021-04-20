# Deep MultiLabel Image Hashing
This repository generates hash codes for LAMDA dataset images which are available at http://www.lamda.nju.edu.cn/data_MIMLimage.ashx?AspxAutoDetectCookieSupport=1. In my computer, they are located at: "/home/ubuntu/caffe/data/lamda/train/"  folder. 

# 1 
The first step is extract image features from VGG16 and locate all features into a single numpy file by "preprocess_and_extract_X.py" script. It creates the preprocessed_X.npy file. We choose this way since fine-tuning VGG16 requires much computational time its parameters are frozen. data.csv holds image file names.

# 2
Train the network and take a snapshot into ./model folder after training by dmlh2.py. targets.mat is a Matlab file which is available at LAMDA web site. It holds image labels.  

# 3 
If snapshot of the trained network was taken, then generate hash codes to ./hashCodes folder by generateCodes_all.py. If you want to get Hamming distance between two images, use generateCodes.py script. generateCodes_all.py on the other hand, generates hash codes for the whole dataset images.

Feel free asking any question.

More details about this study will be given upon request.
