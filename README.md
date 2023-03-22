# Hardware-Aware-Article-Repository

![figure1_no_background](https://user-images.githubusercontent.com/83427055/223422836-7b786cd5-9100-4251-b6e0-791cbe28cd66.png)


Code authors: Philippe Drolet and Victor Yon.  
Using a modified version of the Pytorch library [1]

Repository associated with the article: "Hardware-aware Training Techniques Improving Robustness of Ex Situ Neural Network Transfer on passive TiO2 ReRAM Crossbars".

Cite as:
```bibtex
@article{<main_author>_<year>,
  author = {<last_name_1>, <first_name_1> and <last_name_2>, <first_name_2>},
  doi = {<doi_url>},
  title = {{<Project title>}},
  year = {<year>},
  month = {<month>}
}
```

# How to use

## 1. Install code and dependencies

```bash
# Clone project
git clone https://github.com/3it-inpaqt/Hardware-Aware-Article.git
cd Hardware-Aware-Article

# Optional but recommended: create a virtual environment with python >3.8
python3 -m venv venv
source venv/bin/activate

# Install project dependencies
pip install -r requirements.txt
 ```

## 2. Run the main script

To train the hardware-aware network, run the main function, the training parameters can be changed using the settings.py file in the utils folder.

 ```bash
# If using a virtual environment, activate it
source venv/bin/activate

# Run the main script
python main.py
```

# Continuing the library

So far, as a proof of concept only the linear layer was implemented, the same could be done for other Pytorch layers such as Conv2D or Conv3D. Other variability sources could also be implemented as mentioned in the conclusion of the article.

# Contact information

Please contact philippe.drolet3@usherbrooke.ca or yann.beilliard@usherbrooke.ca if you have any questions or if you would like access to the conductance tuning imprecision data or the biasing scheme effects data or if you would like to contribute to this project. 

[1] Paszke, A. et al., 2019. PyTorch: An Imperative Style, High-Performance Deep Learning Library. In Advances in Neural Information Processing Systems 32. Curran Associates, Inc., pp. 8024–8035. Available at: http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf.
