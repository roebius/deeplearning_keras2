# Modified notebooks and Python files for Keras 2 and Python 3 from the fast.ai Deep Learning course
The repository includes modified copies of the original Jupyter notebooks and Python files from the excellent
(and really unique) deep learning course "Practical Deep Learning For Coders" Part 1 and Part 2,
created by [fast.ai](http://course.fast.ai). The [original files](https://github.com/fastai/courses)
require Keras 1 and Python 2.

### Part 1
Located in the _nbs_ folder. Tested on both Ubuntu 16.04 with Python 3.5 (installed through apt-get) and
MacOS 10.12.4 with Python 3.6 (installed with Homebrew). In Part 1 the Theano backend for Keras has been used. 

### Part 2 (work in progress)
Located in the _nbs2_ folder. Tested on Ubuntu 16.04 with Python 3.5 (installed through apt-get). In Part 2 the TensorFlow backend for Keras has been used.  

The files keras.json.for\_TensorFlow and keras.json.for\_Theano provide a template for the appropriate keras.json file, based on which one of the two backends needs to be used.

A Python 3 virtualenv has been used for both parts. In order to facilitate the installation of the required Python packages, this repository includes 
also the requirement files that can be used with the pip command. These files include additional packages that might be useful for further exploration.

### Status and Notes about Part 2
*rossman.ipynb*: I left out the "Using 3rd place data" section

*spelling_bee_RNN.ipynb*: works as is. In the final "Test code ..." section I found one issue that I was not able to solve (K.conv1d not working) 

*taxi_data_prep_and_mlp.ipynb*: I left out the the "Uh oh ..." section. Caveat: running all the notebook at once used up all the RAM (128 GB); in order to run it I first generated the bcolz arrays until the end of the "MEANSHIFT" section, then I restarted the kernel of the notebook and, after executing the initial imports and path cells, I resumed the notebook execution from the "Formatting Features for Bcolz iterator" section (memory usage however was quite high with peaks at 100 GB RAM)

*identifying comments added to the files*: I used *"# -"* to start my comments when they are not just *"# Keras 2"* 

### In progress
*seq2seq-translation.ipynb* (in progress, currently the file in the _nbs2_ folder is still the original one)

*tiramisu-keras.ipynb*: (in progress, currently the modified file in the _nbs2_ folder is an experiment)

*tiramisu-pytorch.ipynb* (in progress, currently the file in the _nbs2_ folder is still the original one)

### Left out
*neural-style-pytorch.ipynb*: no plan to go through this: I found no way to load the VGG weights; it looks like some version compatibility issue

*rossman_exp.py*: no plan to go through this

*taxi.ipynb*: no plan to go through this
