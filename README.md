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

### Parameterization
* Deep convolutional GAN (DCGAN) ([Radford et al., 2015](https://doi.org/10.48550/arXiv.1511.06434))
* Wassertein GAN (WGAN) ([Arjovsky et al., 2017](https://doi.org/10.48550/arXiv.1701.07875))
* Boundary Equilibrium GAN (BEGAN) ([Berthelot et al., 2017](https://doi.org/10.48550/arXiv.1703.10717))
* EBGAN
* WGAN-GP
* DCGAN with R1 regularization (DCGAN-R1) 

