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

To start, the data must be formatted into a pandas dataframe for compatibility with our framework. If the data is in .MS file format, you must **execute `CASA_to_dataframe.py` within CASA**. Ensure that the path to the .MS file and the desired output location are configured within `CASA_to_dataframe.py`:

```
execfile('CASA_to_dataframe.py')
```
If the data is in the uvfits form, you can use:
```
python3.10 UVfits_to_pysegments.py -f path/to/file/data.uvfits -n NameOfPickle
```

The pandas dataframe will look like this:

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
where the `Stream` column contains the complex data in a 2D stream format, such as (44379.87109375-1.060924660123419e-06j), with length corresponding to the integration times.

`SigNova` is a framework designed for detecting radio frequency interference (RFI) in astronomical data. It loads the input files specified in the `config.yaml` file, computes the minimum Mahalanobis distance to a reference corpus of clean data, calculates the scores of inliers to calibrate the flagger, detects outliers in new data, saves the results, and generates a plot of the detected outliers. 

In the `config.yaml` file, you can specify the input files for the corpus, inliers (calibration), and test data. You can also update different parameters such as:
* stream transformations: time, lead-lag, and base-point. (For visibility data we do not apply any).
* vectorization: signature truncation level, compute expected signature.
* pysegments:
  * signal_tolerance: $2^{sig _ tol}$ governs how much we split intervals in order to find an interval where the characteristic function is True. For example, if $sig\_tol$=3 we have $2^{sig\_tol}$=8, we will never go finer than 8.).
  * tolerance ($2^{tol}$ is the minimum length by which we can try to extend an interval on which the characteristic function is True. For example, say we are on the interval [0,64] where the characteristic function is True, we try to extend to the right, and $tol$=2, that is $2^{tol}$=4. If [0,64+4] returns False, we will stop there. and just say that [0,64] is True.).
  * distfit: use of `distfit` to fit a curve on the scores and which curve to chose (genexteme as default), choose a threshold (0.005 as default).
* nearest neighbor: compute score per frequency channel.

To run the framework, execute the `run_script.py` file using the command: 

```
python3.10 run_script.py
```

The framework uses the `flagger.get_inliers_scores` function in `run_script.py` to generate a `.pkl` file containing the distribution of inliers/calibration scores, indicating the deviation from the corpus of clean data.

Subsequently, the `flagger.flag` function in `run_script.py` generates a `.npy` file marking the presence of RFI with ones and zeros otherwise, arranged in a shape of (n_times, n_frequency_channels). Users can define the output location of the `.npy` file as needed.

For visualization, `run_script.py` produces a waterfall plot for a single dataset using the `flagger.plot_result` function, with customizable plot locations. To generate a waterfall plot with concatenated outputs from multiple datasets, `full_pysegments_plot.py` can be utilized.


Credits to Maud Lemercier and Paola Arrubarrena from DataSig. If you have any questions, please do not hesitate to contact Maud or me at p.arrubarrena@imperial.ac.uk. 


