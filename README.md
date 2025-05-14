# Transfer Learning for Condition Diagnosis and Prognosis of Similar Systems
This repository implements a transfer learning pipeline for using data of similar systems for condition diagnosis and prognosis of engineering systems. 
The core idea is to increase the amount of available training data by drawing on information from similar systems. Two methods of transfer learning are implemented: parameter transfer with fine-tuning and domain-adversarial training. Both methods can be used with different types of neural networks.
We would like to thank the developers and maintainers of the open-source Python packages that made this project possible. In particular, we would like to thank the developers of TensorFlow and Keras for providing powerful and flexible deep learning frameworks. Separately, we would also like to mention the developers of the keras-tcn (1) package, which is currently not an official part of the Keras library.

(1) Philippe Remy, Temporal Convolutional Networks for Keras, 2020, GitHub, GitHub repository, https://github.com/philipperemy/keras-tcn

## License
GNU General Public License v3.0
## Citation
The code is part of the following publication, which have to be cited when using it:

Braig, M.; Zeiler, P. (2025): A Study on Using Transfer Learning to Utilize Information from Similar Systems for Data-Driven Condition Diagnosis and Prognosis. IEEE Access, Volume xxx, pp. xxx - xxx, DOI: xxx
