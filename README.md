# DMLH2
This repository generates hash codes for LAMDA dataset images which are available at http://www.lamda.nju.edu.cn/data_MIMLimage.ashx?AspxAutoDetectCookieSupport=1.

# 1 
The first step is extract features from VGG16 and creates a single *.npy file by "preprocess_and_extract_X.py". 
Since fine-tuning VGG16 requires much computational time we choose this way. I mean parameters of VGG16 are frozen. In my
computer images are located at /home/ubuntu/caffe/data/lamda/train/  folder and preprocessed_X.npy was created. data.csv contains image file names.

# 2
Train the network and take a snapshot in ./model folder after training by dmlh2.py. I'll explain all the detail about it after. Just keep in mind that /home/ubuntu/Desktop/Thesis_Follow_Up_2/dmqRetrieval/lamdaDataset/hashCodes/targets is a Matlab file which is available at LAMDA web site. It contains image labels.  


# 3 
The snapshot was taken. Now generate hash codes to ./hashCodes folder by generateCodes_all.py. If you want to get Hamming distance between two images, use generateCodes.py script.

Feel free asking any question.


More details about this study will be given upon request.
