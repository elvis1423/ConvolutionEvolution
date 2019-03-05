# FaceNet

The FaceNet model is used to encode a face crop image to a 128-dimensional vector, with which downstream applications could do face verification/recognition or face clustering.

# Model

FaceNet is based on GoogLeNet<sup>[1]</sup> whose performance is better than AlexNet on image classification. The main specialty that helps GoogLeNet out perform AlexNet is the deeper and wider network architecture it used. Through carefully designing, GoogLeNet introduced Inception blocks. To limit the explosion of filters, 1x1 convolution layers are pervasively used.

Please look at the figure 1 to have a glance at FaceNet.

<img src='images/FaceNet_overview.png'/>

<center>Figure 1. FaceNet Overview</center>

From figure 1 we can see, the layers before Inception are inspired by AlexNet: large sized filters interleaving with Max pools