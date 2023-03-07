# Hardware-Aware-Article-Repository

Authors: Philippe Drolet and Victor Yon
Using a modified version of the Pytorch library [1]

Repository associated with the article: "Hardware-aware Training Techniques Improving Robustness of Ex Situ Neural Network Transfer on passive TiO2 ReRAM Crossbars" available at URL: 
## Requirements

The code's library requirements are written in the requirements.txt file.

## Running the code

To train the hardware-aware network, run the main function, the training parameters can be changed using the settings.py file in the utils folder.

## Continuing the library

So far, as a proof of concept only the linear layer was implemented, the same could be done for other Pytorch layers such as Conv2D or Conv3D. Other variability sources could also be implemented as mentionned in the conclusion of the article.


## Contact information

Please contact philippe.drolet3@usherbrooke.ca or yann.beilliard@usherbrooke.ca if you have any questions or if you would like access to the conductance tuning imprecision data or the biasing scheme effects data or if you would like to contribute to this project. 

[1] Paszke, A. et al., 2019. PyTorch: An Imperative Style, High-Performance Deep Learning Library. In Advances in Neural Information Processing Systems 32. Curran Associates, Inc., pp. 8024–8035. Available at: http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf.
