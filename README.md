# Support-Vector-Machine-Classification-of-Oncogenes
Built a support vector machine (SVM) from scratch (no scikit-learn libraries), used it for oncogene classification based on gene expression data.

The code files are svm.py and geneExpressionClassification.py.
The svm.py is the SVM class, which should be initialized and run on the
processed gene expression data that is in geneExpressionClassification.py, which is where all the
preprocessing of the data occurs. The svm class is imported in geneExpressionClassification.py.
In order to run the SVM, see the commented code below from geneExpressionClassification.py.
The SVM uses the training and testing X (gene expression level features) and Y (labels, either -1 or 1) 
from the processed data csv files. The three csv files must be in the same directory
as the code. The initialization involves defining if the kernel is linear or gaussian as a string.
The initialized class instance of the SVM also takes an optional argument for the sigma value
which is only used if it is a gaussian kernel on non-linear data.
The processed gene data used is in lines 270-271 for the training and testing feature data, X.
The labels for the feature data are in lines 278-280. The genes used for training and predicition
of the SVM can be adjusted in lines 270-271, for example to use the first 2 genes:
X_train = train.values[:,:2], or first 10 genes: X_train = train.values[:,:10].

To run the file, from the command line in the project folder:
CS5100\project>Python ./geneExpressionClassification.py

The first figure needs to be closed, which plots the principal components and explained variance.

The SVM results is outputted to the console. The linear SVM hyperplaned plot is displayed, and the
non-linear plot is saved as a pdf in the local directory.

The following sample commented code should be run at the end of the geneExpressionClassification.py file.

### Run SVM using non-linear classifier 

#### on training data - training parameters
#### use gaussian kernel, with a sigma of 5.0

    svm = SVM("gaussian", 5.0)
    svm.fit(X_train, y_train)

### Run SVM to predict testing data - classification
    svm.predict(X_test, y_test)

### Visualize the decision boundary contour plot
    svm.visualize()


### Run SVM using linear classifier

#### on training data, use linear kernel
#### use a sample of data from genes M22960 and X62654 as features

#### sample of training gene data from M22960 and X62654
                          # -1
    X_train2 = np.array([[330,631],
                         [305,814],
                         [196,712],
                          # 1
                         [2179, 1888],
                         [2501, 1907],
                         [1885, 743]])

#### train labels from sample gene data
    y_train2 = [-1, -1, -1, 1, 1, 1]

#### initialize linear classifier SVM
    svmLinear = SVM("linear")

#### fit on training data - training parameters
    svmLinear.fit(X_train2, y_train2)

#### sample of testing gene data from M22960 and X62654
               # -1
    X_test2 = [[-178,649],
               [456,1935],
               [182,661],
               [265,1339],
               [207,586],
               # 1
               [1850,1590],
               [2801,1292],
               [2444,1756],
               [2719,932]]

#### test labels from sample gene data
    y_test2 = [-1, -1, -1, -1, -1, 1, 1, 1, 1]

### Run SVM to predict testing data - classification
    y_predict2 = svmLinear.predict(X_test2, y_test2)

#### Visualize the linear hyperplane decision boundary plot
    svmLinear.visualize()

The graphs for the data preprocessing can also be optionally displayed from lines 210-216:

#### Plot PCs
    plot_principal_components(exp_var_ratio_train, cum_sum_train)
    plot_principal_components(exp_var_ratio_test, cum_sum_test)

#### Plot projections 
    plot_projections(X_pretrain, X_reduced_train)
    plot_projections(X_pretest, X_reduced_test)

and lines 358-361:

#### Visualize the correlation grid for the training data
    visualize_correlation_grid(train)
#### Visualize the correlation grid for the testing data
    visualize_correlation_grid(test)
