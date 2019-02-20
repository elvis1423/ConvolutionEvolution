# AlexNet Implementation Step by step

I want to organize my understanding of Convolutional Neural Networks by implementing the AlexNet$^{[1]}$. I would like to use TensorFlow as building blocks to practice programming skills. Image pre-process uses opencv library. 

## Model

| Name           | oc   | ic   | H    | W    | Pading | Stride |
| -------------- | ---- | ---- | ---- | ---- | ------ | ------ |
| Orig Image     |      | 3    | any  | any  |        |        |
| cropped        |      | 3    | 256  | 256  |        |        |
| input(random)  |      | 3    | 227  | 227  |        |        |
| filter 1       | 96   | 3    | 11   | 11   | 0      | 4      |
| activation 1_0 |      | 96   | 55   | 55   |        |        |
| maxpool1       |      |      | 3    | 3    | 0      | 2      |
| activation 1_1 |      | 96   | 27   | 27   |        |        |
| filter 2       | 256  | 96   | 5    | 5    | 2      | 1      |
| activation 2_0 |      | 256  | 27   | 27   |        |        |
| maxpool 2      |      |      | 3    | 3    | 0      | 2      |
| activation 2_1 |      | 256  | 13   | 13   |        |        |
| filter 3       | 384  | 256  | 3    | 3    | 1      | 1      |
| activation 3_0 |      | 384  | 13   | 13   |        |        |
| filter 4       | 384  | 384  | 3    | 3    | 1      | 1      |
| activation 4_0 |      | 384  | 13   | 13   |        |        |
| filter5        | 256  | 384  | 3    | 3    | 1      | 1      |
| activation 5_0 |      | 256  | 13   | 13   |        |        |
| maxpool 5      |      |      | 3    | 3    | 0      | 2      |
| activation 5_1 |      | 256  | 6    | 6    |        |        |
| FC 6           |      | 4096 |      |      |        |        |
| FC 7           |      | 4096 |      |      |        |        |
| FC 8 & softmax |      | 1000 |      |      |        |        |



**Effect of Dropout**: mitigate overfitting because neurons on same layer can't just depend on the presence of particular other neurons, they must learn more robust features based on randomly appearance of other neurons, thus reduce the co-adaptations between neurons.

The more robust features are useful in conjunction with many different random subsets of the other neurons.