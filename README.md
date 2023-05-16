# PRP - Python reservoir package

![example.png](figures/example.png)

PRAP - Python reservoir assimilation package was developed to evaluate different assimilation methods with different
parameterization and localization methods.

## Avaiable methods
### Assimilation
* Ensemble Smoother with Multiple Data Assimilations (ES-MDA) ([Emerick and Reynolds, 2013](https://doi.org/10.1016/j.cageo.2012.03.011);
[Emerick, 2016](https://doi.org/10.1016/j.petrol.2016.01.029))
* Generalized Iterative Ensemble Smoother (GIES) ([Luo, 2022](https://doi.org/10.1007/s10596-021-10046-1)) (under
development)

### Localization
* Pseudo-optimal localization (POL) ([Lacerda et al., 2019](https://doi.org/10.1016/j.petrol.2018.08.056))
* Correlation-based localization (CL) ([Luo and Bhakta, 2020](https://doi.org/10.1016/j.petrol.2019.106559))
* Pseudo-optimal localization with random-shuffle threshold (POL-RS)
([Ranazzi and Sampaio, 2022](https://doi.org/10.1016/j.petrol.2022.110589))

### Parameterization with GANs
* Deep convolutional GAN (DCGAN) ([Radford et al., 2015](https://doi.org/10.48550/arXiv.1511.06434))
* Wassertein GAN (WGAN) ([Arjovsky et al., 2017](https://doi.org/10.48550/arXiv.1701.07875))
* Boundary Equilibrium GAN (BEGAN) ([Berthelot et al., 2017](https://doi.org/10.48550/arXiv.1703.10717))
* EBGAN
* WGAN-GP
* DCGAN with R1 regularization (DCGAN-R1) ([Mescheder et al., 2018](https://doi.org/10.48550/arXiv.1801.04406);
[Karras et al., 2020](https://doi.org/10.48550/arXiv.1912.04958))

### Datasets
* Two-facies channelized (ref here)
* Stanford three-facies (ref here)
* Reservoir classifier dataset
* UNISIM-II-H processed realizations


| Google Drive files | &nbsp; Preprocessed numpy files
| :--- | :----------
| [Datasets](https://drive.google.com/open?id=1j-1VCur8vW04MrAp79DqNKUgRnOn417g) | Main Dataset folder
| &boxvr;&nbsp; [channel_2d](https://drive.google.com/open?id=1eXB89F3_1w0MiYuUUm6VaFMkKSZiKEeO) | Two-facies channelized
| &boxvr;&nbsp; [3facies](https://drive.google.com/open?id=1euKgA2MYn4E_NMUl5imV80g1O62PlchR) | Stanford three-facies
| &boxvr;&nbsp; [UNISIM-II](https://drive.google.com/open?id=1ZRh21f5Y-YqjKDf9RfnUzwTcou6fry6I) | all UNISIM-II-H field properties preprocessed
| &boxur;&nbsp; [classifier](https://drive.google.com/open?id=1OF1FKvOT9_d-S5RGO7YkTbtha70V3WV3) | Dataset to train Reservoir Classifier
| [Networks](https://drive.google.com/open?id=1bZGLQIo7pX6wckAzn6iG4k5I7GCwfxD7) | Main Networks folder
| &boxvr;&nbsp; [classifier](https://drive.google.com/open?id=1m9Rzbuc_3P9f5oYhIYUwCHv0AH6cEYLx) | Pre-trained Reservoir Classifier Network
| &boxur;&nbsp; [3facies](https://drive.google.com/open?id=1IXa6V4w9T9cNLTjzrEg6pzSbzmZ1SpeJ) | Pre-trained 3facies GAN (reference case)


