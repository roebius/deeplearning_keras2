# Modified notebooks and Python files for Keras 2 and Python 3 from the fast.ai Deep Learning course v.1
The repository includes modified copies of the original Jupyter notebooks and Python files from the excellent
(and really unique) deep learning course "Practical Deep Learning For Coders" Part 1 and Part 2, v.1,
created by [fast.ai](http://fast.ai).

The [original files](https://github.com/fastai/courses) require Keras 1. One main goal has been to modify the original files to the minimum extent possible. The comments added to the modules generally start with *"# -"* when they are not just *"# Keras 2"*.

The current version of the repository has been tested with **_Keras 2.1.2_**.
The previous version, tested with _Keras 2.0.6_, is available [here](https://github.com/roebius/deeplearning_keras2/releases).
### Part 1
Located in the _nbs_ folder. Tested on _Ubuntu 16.04_ and _Python 3.5_, installed through [Anaconda](https://www.anaconda.com), using the [Theano](http://deeplearning.net/software/theano/) 1.0.1 backend.  

### Part 2
Located in the _nbs2_ folder. Tested on _Ubuntu 16.04_ and _Python 3.5_, installed through [Anaconda](https://www.anaconda.com), using the [TensorFlow](https://www.tensorflow.org/) 1.3.0 backend.
A few modules requiring PyTorch were also tested, using [PyTorch](http://pytorch.org/) 0.3.0.  

The files _keras.json.for\_TensorFlow_ and _keras.json.for\_Theano_ provide a template for the appropriate _keras.json_ file, based on which one of the two backends needs to be used by Keras.

An _environment.yml_ file for creating a suitable [conda environment](https://conda.io/docs/user-guide/tasks/manage-environments.html) is provided. 


### Notes and issues about Part 2
*neural-style.ipynb*: due to a function parameter change in _Keras 2.1_, the _VGG16_ provided by _Keras 2.1_ has been used instead of the original custom module _vgg16\_avg.py_

*rossman.ipynb*: section "Using 3rd place data" has been left out for lack of the required data

*spelling_bee_RNN.ipynb* and *attention_wrapper.py*: due to the changed implementation of the recurrent.py module in Keras 2.1, the attention part of the notebook doesn't work anymore

*taxi_data_prep_and_mlp.ipynb*: section "Uh oh ..." has been left out. Caveat: running all the notebook at once exhausted 128 GB RAM; I was able to run each section individually only after resetting the notebook kernel each time

*tiramisu-keras.ipynb*: in order to run the larger size model I had to reset the notebook kernel in order to free up enough GPU memory (almost 12 GB) and jump directly to the model


#### Left-out modules
*neural-style-pytorch.ipynb* (found no way to load the VGG weights; it looks like some version compatibility issue)

*rossman_exp.py*

*seq2seq-translation.ipynb*

*taxi.ipynb*

*tiramisu-pytorch.ipynb*
