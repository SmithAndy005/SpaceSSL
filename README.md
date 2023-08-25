# SpaceSSL

## Self Supervised Learning for the Exploration of Large, Unlabelled Space Physics Datasets

Often in Space Physics, we find that we have large, unlabelled datasets, only a small fraction of which contain observations of the rarely captured, fortuitous events we wish to investigate for scientific studies.  Here we present a solution based upon self-supervised learning whereby we train a model to produce descriptive embeddings that describe two dimensional datasets (e.g. auroral images or particle distributions).  We can then use the distance between embeddings to find similar observations, without manually labelling or inspecting the full dataset.

A paper on the use of SpaceSSL in space physics has been submitted to the Journal of Geophysical Research: Space Physics as a methods paper demonstrating its utility on unlabelled data from the MMS spacecraft and ground based auroral imagers.

## Requirements

We provide a [requirements txt](/requirements.txt) and [conda yml](/environment.yml) file from the macbook pro used during development of the models in Smith et al., (2023, in prep).  However, in short the requirements can all be installed through pip (we would recommend in a virtual environment, e.g. [conda](https://docs.conda.io/en/latest/miniconda.html)), with the main packages required for training being:

- [PyTorch](https://pytorch.org)
- [PyTorch Lightning](https://www.pytorchlightning.ai/index.html)
- [Torchvision](https://pytorch.org/vision/stable/index.html)
- [lightly](https://github.com/lightly-ai/lightly)
- [math](https://docs.python.org/3/library/math.html)

## Usage

The files provided here will train a SimSIAM model on the [OATH auroral dataset](http://tid.uio.no/plasma/oath/) [(see Clausen et al., 2018)](https://doi.org/10.1029/2018JA025274).  For the code to run, please download the OATH dataset (from the link above) into this folder (and check the filepaths are correct).  We thank the authors for producing a high quality, freely available dataset.

The first file [SSL_Train](/SSL_Train.py) will train and save the SimSIAM model.  The current parameter set matches those used in the JGR: SP manuscript (derived through an iterative grid search with [Weights and Biases](https://wandb.ai/home)), the principle hyperparameters that can be tuned are:

- momentum (line 21)
- batch_size (line 22)
- maximumm epochs (line 23)

The collate function (lines 59 - 74) determines the transformations that contribute to the comparisons made during the training.  The current choices correspond to those made for the OATH dataset, as described in the manuscript Smith et al., (2023, in prep).  The selections here depends upon the properties of the chosen data, and should be selected based upon domain knowledge.

The second file [SSL_CreateEmbeddings](/SSL_CreateEmbeddings.py) will evaluate the trained model on the provided folder of images, creating the embeddings for each image.  The code is such that an array of embeddings will be stored, along with a list of filenames that correspond to the generated embeddings.

## Acknowledgements

The raw all sky image data are available from the [THEMIS website](http://themis.ssl.berkeley.edu), but the data used here are from [Clausen et al., (2018)](https://doi.org/10.1029/2018JA025274).   We thank [Clausen et al., (2018)](https://doi.org/10.1029/2018JA025274) for their diligent processing, labelling and provison of THEMIS all sky image data.  We would also like to thank [Lightly/Susmelj](https://github.com/lightly-ai/lightly) for their informative tutorials (e.g. [Lightly SSL Tutorial](https://docs.lightly.ai/self-supervised-learning/index.html)).