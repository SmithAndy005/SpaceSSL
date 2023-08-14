# SpaceSSL

## Self Supervised Learning for the Exploration of Large, Unlabelled Space Physics Datasets

Often in Space Physics, we find that we have large, unlabelled datasets, only a small fraction of which contain observations of the rarely captured, fortuitous events we require for scientific studies.  In this repo we present a solution based upon self-supervised learning whereby we train a model to produce descriptive embeddings that describe two dimensional datasets.  We can then use the distance between embeddings to find similar observations, without labelling or inspecting the full dataset.

A paper on the use of SpaceSSL in space physics has been submitted to the Journal of Geophysical Research: Space Physics as a methods paper demonstrating its utility on unlabelled data from spacecraft and ground based auroral imagers.

## Requirements

We provide a [requirements txt](/requirements.txt) and [conda yml](/environment.yml) file from the machine used development of the models in Smith et al., (2023, in prep).  However, in short the requirements can all be installed through pip (we would recommend in a virtual environment, e.g. [conda](https://docs.conda.io/en/latest/miniconda.html)), with the main packages for training being:

- [PyTorch](https://pytorch.org)
- [PyTorch Lightning](https://www.pytorchlightning.ai/index.html)
- [Torchvision](https://pytorch.org/vision/stable/index.html)
- [lightly](https://github.com/lightly-ai/lightly)
- [math](https://docs.python.org/3/library/math.html)

## Usage

The files provided here will train a SimCLR model on the [OATH auroral dataset](http://tid.uio.no/plasma/oath/) [(Clausen et al., 2018)](https://doi.org/10.1029/2018JA025274).  For the code to run, please download the OATH dataset (from the link above) into this folder (and check the filepaths are correct).

The first file [SSL_Train](/SSL_Train.py) will train and save the SimCLR model.  The current parameter set matches those used in the JGR: SP manuscript, the principle hyperparameters that can be tuned are:

- batch_size (line )
- epochs (line )
- momentum (line )

The collate function (lines X - X) determines the transformations that contribute to the comparsons made during the training.  The current choices correspond to those made for the OATH dataset, as described in the manuscript Smith et al., (2023, in prep).  The selections here depends upon the data selected and ideally should be set based upon domain knowledge.

The second file will evaluate the trained model on the database of images, creating the embeddings for each image. The function "X" will then return the top images similar to provided image filename within the dataset.

## Acknowledgements

The raw all sky image data are available from the [THEMIS website](http://themis.ssl.berkeley.edu), but the data used here are from [Clausen et al., (2018)](https://doi.org/10.1029/2018JA025274).   We thank [Clausen et al., (2018)](https://doi.org/10.1029/2018JA025274) for their diligent processing, labelling and provison of THEMIS all sky image data.  We would also like to thank [Lightly/Susmelj](https://github.com/lightly-ai/lightly) for their informative tutorials (e.g. [Lightly SSL Tutorial](https://docs.lightly.ai/self-supervised-learning/index.html)).