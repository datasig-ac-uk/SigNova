# SigNova
We presente a semi-supervised anomaly detection framework designed to detect anomalies in radio astronomy data, with a focus on identifying radio frequency interference (RFI). The data we work with is captured by a time-frequency stream of complex numbers called visibility $V_{i,j}(t)\in\mathbb{C}$ for each antenna-pair $(i,j)$ in an array of $N_A$ antennas. To identify the features of the signal, for example on antenna $i$, we take the average of the **signatures** of $\{V_{i,1},\ldots,V_{i,N_A}\}$ resulting in a feature vector for antenna $i$. Using the signature map from rough path theory, we convert sequences of observations into a vector representation, allowing us to better analyze multivariate sequences of our streamed data.

# Dependencies
Our framework uses the following external modules, which should all be pulled in automatically if you use `pip` (sometimes `python3.10 -m pip install -U [module]` works better):
* python 3.10
* `pip3.10 install iisignature`, [iisignature](https://pypi.org/project/iisignature/) 
* `pip3.10 install distfit`, [distfit](https://pypi.org/project/distfit/)
* `pip3.10 install tqdm`, [tqdm](https://pypi.org/project/tqdm/)
* `pip3.10 install pysegments-0.3-cp310-cp310-linux_x86_64.whl` (download pysegments wheel in this directory), [pysegments github](https://github.com/datasig-ac-uk/pysegments)
* `pip3.10 install matplotlib`, [matplotlib](https://pypi.org/project/matplotlib/)
* `pip3.10 install scipy`, [scipy](https://pypi.org/project/scipy/)
* `pip3.10 install scikit-learn`, [sklearn](https://pypi.org/project/scikit-learn/)
* `pip3.10 install pynndescent`, [pynndescen](https://pypi.org/project/pynndescent/)
* `pip3.10 install omegaconf`, [omegaconf](https://pypi.org/project/omegaconf/)
* `pip3.10 install mne`, [mne](https://pypi.org/project/mne/)

# Files

First we need to accomodate the data in a pandas dataframe way that our framework can read. If the data is in .MS file, you have **to run `CASA_to_dataframe.py` in CASA**:
```
execfile('CASA_to_dataframe.py')
```
The path to the .MS file and the output needs to be set on `CASA_to_dataframe.py`. If the data is in the uvfits form, you can run:
```
python3.10 UVfits_to_pysegments.py -f path/to/file/data.uvfits -n NameOfPickle
```

The pandas dataframe has to have the following form:

```
          Ant1   Ant2     FrCh                             Stream
0          0.0    0.0    FrCh1  [[44379.87109375, -1.060924660123419e-06], [44...
1          0.0    1.0    FrCh1  [[42.1793212890625, 97.59571838378906], [31.73...
2          0.0    2.0    FrCh1  [[-17.46160888671875, 106.27327728271484], [41...
3          0.0    3.0    FrCh1  [[-224.1186981201172, 94.81196594238281], [-16...
4          0.0    4.0    FrCh1  [[-11.914325714111328, 62.37732696533203], [-5...
...        ...    ...      ...                                                ...
3121147  125.0  126.0  FrCh384  [[-3.8130815029144287, -45.403297424316406], [...
3121148  125.0  127.0  FrCh384  [[-33.647464752197266, -7.50682258605957], [-3...
3121149  126.0  126.0  FrCh384  [[23694.1875, 4.5934194758956437e-07], [23587....
3121150  126.0  127.0  FrCh384  [[51.8315315246582, 26.06503677368164], [-10.3...
3121151  127.0  127.0  FrCh384  [[23815.134765625, -6.114648840593873e-07], [2...

```
where the `Stream` column contains the complex data in a 2D stream (44379.87109375-1.060924660123419e-06j) with shape of the integration times.


`SigNova` is a framework designed for detecting radio frequency interference (RFI) in astronomical data. It loads the input files specified in the `config.yaml` file, computes the minimum Mahalanobis distance to a reference corpus of clean data, calculates the scores of inliers to calibrate the flagger, detects outliers in new data, saves the results, and generates a plot of the detected outliers. 

In the `config.yaml` file, you can specify the input files for the corpus, inliers (calibration), and test data. You can also update different parameters such as stream transformations, vectorization, pysegments parameters for localizing the outliers, and compute the nearest neighbor.

The _Mahalanobis distance_ is a measure of the distance between a point and a distribution, taking into account the covariance between variables. The framework uses the Mahalanobis distance to identify outliers in the input data. The _signature truncation level_ refers to the level of truncation applied to the signature map, which is used to vectorize sequences of observations. A higher level of truncation results in a lower-dimensional representation of the data, but may also result in a loss of information.

To run the framework, execute the `run_script.py` file using the command: 

```
python3.10 run_script.py
```

You can change the name and location of the output file by specifying the output_name parameter. The framework will generate a `.pkl` file containing the inliers/calibration score distribution. The scores tell us "how far" are the inliers from the corpus of clean data. 

EXPLAIN MORE. GET PLOT -> name in run script


Please note that this framework is still under development, and we would appreciate your discretion when using it. If you have any questions, please do not hesitate to contact me at p.arrubarrena@imperial.ac.uk. Credits to Maud Lemercier and Paola Arrubarrena from DataSig.


