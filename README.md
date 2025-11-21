# Wine Classification
Contributors/Authors
- Prabuddha Tamhane (PAT0216)
- Harrison Li (Harrisonlee0530)
- Shihan Xu (shihan66)
- Wesley Beard (Beardw)

## About
This project illustrates how our team endeavoured to build a classification model to predict whether a wine was red or white based on a set of wine quality features (ex. pH, residual sugars, etc.).  Four models were investigated with the best one being RBF SVM. It performed extremely well on our test data, with an accuracy of 0.9969 and f1 score of 0.9979. While there were still false positives and false negatives, indicated by the scores being less than one, this is not of great concern to us. Since we are not dealing with life-threatening or possible adverse outcomes should a prediction be incorrect, the high level of precision in our model has given us confidence to use it in production.

The dataset used in this project was an amalgamation of two datasets related to wine from the northern region of Portugal: specifically, Vinho Verde red and white samples. Each row represents a wine sample with 11 different features including pH, residual sugar, density etc. The datasets were distinguished by being either for red or white wines. It was created by Paulo Cortez, A. Cerdeira, F. Almeida, T. Matos, and J. Reis and can be sourced from the UC Irvine Machine Learning Repository [here](https://archive.ics.uci.edu/dataset/186/wine+quality).

## Report
The final report can be found [here](wine_classifier.ipynb).

## Dependencies
* `conda` (version 23.9.1 or higher)
* `conda-lock` (version 2.5.7 or higher)
* `jupyterlab` (version 4.4.7 or higher)
* `nb_conda_kernels` (version 2.5.1 or higher)
* Python and the packages listed in [environment.yml](environment.yml)

## Usage
When running the project for the first time, please run the following from the root of this repository:
```
conda-lock install --name wine-classifier conda-lock.yml
```
To run the analysis, please run the following from the root of this repository:
```
jupyter lab
```
Open `wine-classifier.ipynb` in JupyterLab.
Under Switch/Select Kernel choose "Python [conda env:wine-classifier]".

Lastly, under the "Kernel" menu click "Restart Kernel and Run All Cells...".

## License
The Wine Classification report, code, and additional documentation within this repository are licensed under the MIT license.

## References
Cortez, P., Cerdeira, A.L., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. Decis. Support Syst., 47, 547-553.

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Wine Quality [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C56S3T.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, *12*, 2825–2830.

VanderPlas et al., (2018). Altair: Interactive Statistical Visualizations for Python. Journal of Open Source Software, 3(32), 1057, https://doi.org/10.21105/joss.01057
