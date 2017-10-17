# Modified notebooks and Python files for Keras 2 and Python 3 from the fast.ai Deep Learning course
The repository includes modified copies of the original Jupyter notebooks and Python files from the excellent
(and really unique) deep learning course "Practical Deep Learning For Coders" Part 1 and Part 2,
created by [fast.ai](http://fast.ai). The [original files](https://github.com/fastai/courses)
require Keras 1.

### Part 1
Located in the _nbs_ folder. Tested with Keras 2.0.6 on both Ubuntu 16.04 and Python 3.5 (installed through apt-get) and
MacOS 10.12.4 with Python 3.6 (installed with Homebrew). In Part 1 the Theano backend for Keras has been used. 

### Part 2
Located in the _nbs2_ folder. Tested with Keras 2.0.6 on Ubuntu 16.04 with Python 3.5 (installed through apt-get). In Part 2 the TensorFlow backend for Keras has been used.  

The files keras.json.for\_TensorFlow and keras.json.for\_Theano provide a template for the appropriate keras.json file, based on which one of the two backends needs to be used.

A Python 3 virtualenv has been used for both parts. In order to facilitate the installation of the required Python packages, this repository includes 
also the requirement files that can be used with the pip command. These files include additional packages that might be useful for further exploration.  

The comments that I inserted in the modules generally start with *"# -"* when they are not just *"# Keras 2"*.

### Notes about Part 2
#### Issues 
*rossman.ipynb*: section "Using 3rd place data" has been left out for lack of the required data

*spelling_bee_RNN.ipynb*: after the main part of the notebook, in the final "Test code ..." section I was not able to solve an issue with the K.conv1d cell not working 

*taxi_data_prep_and_mlp.ipynb*: section "Uh oh ..." has been left out. Caveat: running all the notebook at once exhausted 128 GB RAM; I was able to run each section individually only after resetting the notebook kernel each time

*tiramisu-keras.ipynb*: in order to run the larger size model I had to reset the notebook kernel in order to free up enough GPU memory (almost 12 GB) and jump directly to the model

#### Left-out modules
*neural-style-pytorch.ipynb* (found no way to load the VGG weights; it looks like some version compatibility issue)

*rossman_exp.py*

*seq2seq-translation.ipynb*

*taxi.ipynb*

*tiramisu-pytorch.ipynb*
