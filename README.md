# Deep MultiLabel Image Hashing
This repository generates hash codes for LAMDA dataset images which are available at http://www.lamda.nju.edu.cn/data_MIMLimage.ashx?AspxAutoDetectCookieSupport=1. In my computer, they are located at: "/home/ubuntu/caffe/data/lamda/train/"  folder. 

# 1 
The first step is extract image features from VGG16 and locate all features into a single numpy file by "preprocess_and_extract_X.py" script. It creates the preprocessed_X.npy file. We choose this way since fine-tuning VGG16 requires much computational time its parameters are frozen. data.csv holds image file names.

# 2
Train the network and take a snapshot into ./model folder after training by dmlh2.py. I'll explain all the detail about it after. Just keep in mind that /home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/lamdaDataset/hashCodes/targets is a Matlab file which is available at LAMDA web site. It holds image labels.  

# 3 
The snapshot was taken. Now generate hash codes to ./hashCodes folder by generateCodes_all.py. If you want to get Hamming distance between two images, use generateCodes.py script.

Feel free asking any question.

More details about this study will be given upon request.
