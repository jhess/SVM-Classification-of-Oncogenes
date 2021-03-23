import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler as stdscale
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import StratifiedShuffleSplit
import csv
from svm import SVM

## Justin Hess
##
## geneExpressionClassification.py

#Compute the Covariance matrices
def compute_cov_matrix(X):
    #first standardize the data into unit scale (mean = 0, variance = 1)
    X_std = stdscale().fit_transform(X)

    #mean_vec = np.mean(X_std, axis=0)
    #cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0] - 1)
    cov_mat = np.cov(X_std.T)

    return cov_mat, X_std

#Compute and Project the main Principal Components onto a New Subspace
def get_principal_components(X, Y):
    #Inputs: X and Y
    #X: the filtered dataset, training or testing
    #Y: the list of class labels
    
    #now we can get the eigenvalues and eigenvectors from the Covariance Matrix
    cov_matrix, X_std = compute_cov_matrix(X)

    sklearn_pca = sklearnPCA(n_components=10)
    Y_sklearn = sklearn_pca.fit_transform(X_std)
    exp_var = sklearn_pca.explained_variance_
    cum_exp_var = np.cumsum(exp_var)
    #these two variables are used in the graph, explained variance as final_features percentage of total dataset
    exp_var_ratio = sklearn_pca.explained_variance_ratio_*100
    cum_sum = np.cumsum(exp_var_ratio)

    sklearn_pca.explained_variance_ratio_[:10].sum()

    #Here we can concatenate the info if the patient has ALL or AML (0 for ALL, 1 for AML)
    X['type'] = Y
    cancer_type_dict = {'ALL':0,'AML':1}
    cancer_type_dict_reverse = {0:'ALL', 1:'AML'}
    X.replace(cancer_type_dict, inplace=True)
    Z = X['type'].map(cancer_type_dict_reverse)
    label=['ALL', 'AML']
    
    #Create PCs as final_features projection onto New Feature Space in final_features 2D graph
    sklearn_pca = sklearnPCA(n_components=3)
    X_reduced  = sklearn_pca.fit_transform(X_std)

    return exp_var_ratio, cum_sum, X, X_reduced

#Extract the main genes, i.e. features, most important to the data variance
def extract_features(data):
    
    data_1 = data.copy()
    labels = data_1.cancer.values
    #Scale/standardize the data using the preprocessing package from sci-kit learn
    #this subtracts the mean and scales to the unit variance
    features = preprocessing.scale(data_1.drop("cancer", axis = 1).values) 
    feature_names = np.array(data_1.drop("cancer", axis = 1).columns.tolist())

    #create our stratified shuffle split object from sci-kit learn
    stratified_shufflesplit_obj = StratifiedShuffleSplit(n_splits = 9, test_size = 0.1)

    # initialize our array to store the extracted final features
    final_features = []

    for train_index, test_index in stratified_shufflesplit_obj.split(features, labels): 
        #extract the test and training features from the object it deems relevant to the data variance
        X_train, X_test = features[train_index], features[test_index]
        #extract the corresponding test and training labels from the object it deems relevant to the data variance
        y_train, y_test = labels[train_index], labels[test_index]

        #select the features based on the highest percentile score
        select_percentile_obj = SelectPercentile(f_classif, percentile = 1)

        select_percentile_obj.fit(X_train, y_train)
    
        # get the most important features
        best_features = select_percentile_obj.get_support()

        #calculate the scores
        scores = select_percentile_obj.scores_
        scores = scores[best_features].tolist() 

        # now sort the extracted features
        extracted_features = feature_names[best_features].tolist()

        score_results = {extracted_features[i]:scores[i] for i in range(len(extracted_features))}

        # check to make sure during each iteration that the features is not emtpy, i.e. length of 0
        if len(final_features) == 0:
            final_features = set(sorted(score_results, key = score_results.get))
        else:
            final_features = set(final_features).intersection(set(sorted(score_results, key = score_results.get)))
    
    # create a final list of the final extracted features
    final_features = list(final_features)
    # add on the cancer column to show which patients had cancer in final visualization
    final_features.append('cancer')

    return final_features

# Plot principal components, explained variance
def plot_principal_components(exp_var_ratio, cum_sum):

    fig, ax = plt.subplots(figsize=(8,8))

    #Plot the Explained Variance as final_features percentage
    #about 65 percent of total variance can be explained by the first 10 principal components
    bars = plt.bar(range(10), exp_var_ratio, color = 'b',alpha=0.5, label='Explained Variance')
    line = plt.plot(range(10), cum_sum, color = '#F0A757', linestyle='-', label='Cumulative Variance')

    plt.title("Explained Variance by Different Principal Components")
    plt.xlabel("Principal Components")
    plt.xticks(range(10),['PC %s' %i for i in range(1,11)])
    plt.ylabel("Explained variance in percent")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Plot projections into feature spaces
def plot_projections(X, X_reduced):

    #Plot the PCs as final_features projection onto New Feature Space in final_features 2D graph
    fig = plt.figure(1, figsize=(10,6))
    plt.scatter(X_reduced[:, 0],  X_reduced[:, 1], c=X['type'], cmap='tab10', linewidths=10, label='AML')
    plt.scatter(X_reduced[:, 0],  X_reduced[:, 1], c=X['type'], cmap='tab10', linewidths=10, label='ALL')
    plt.title("Projection onto New Feature Space for Data")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.show()

#Visualize the correlation values for the gene expression data on heat maps
def visualize_correlation_grid(data):

    column_labels = data.columns
    row_labels = data.index
    #Scale the data using the preprocessing package from sci-kit learn
    data_1 = preprocessing.scale(data)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Create a heat map of the correlation values
    heatmap = ax.pcolor(data_1, cmap= 'coolwarm')

    fig.colorbar(heatmap)

    ax.set_xticks(np.arange(data_1.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data_1.shape[0]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap.set_clim(-1,1)
    plt.tight_layout()
    
    plt.xlabel("Gene Accession Number")
    plt.ylabel("Patient")
    plt.show()

#Datasets of ALL and AML
#Training Dataset
train = pd.read_csv("ALL_AML_data_set_train.csv")

y_train = list(pd.read_csv("actual.csv")[:38]['cancer'])
#Testing Dataset
test = pd.read_csv("ALL_AML_data_set_test.csv")

y_test = list(pd.read_csv("actual.csv")[38:]['cancer'])

#Data Cleaning

#remove the unnececssary "call" columns
#Training Dataset
train_v1 = [col for col in train.columns if "call" not in col]
train = train[train_v1]
#Testing Dataset
test_v1 = [col for col in test.columns if "call" not in col]
test = test[test_v1]

#do Transpose, now the rows are the patients and columns are genes
train = train.T
test = test.T
#clean data
train.columns = train.iloc[1]
train = train.drop(['Gene Description', 'Gene Accession Number']).apply(pd.to_numeric)

test.columns = test.iloc[1]
test = test.drop(['Gene Description', 'Gene Accession Number']).apply(pd.to_numeric)

#Get PCs for Training and Testing datasets
exp_var_ratio_train, cum_sum_train, X_pretrain, X_reduced_train = get_principal_components(train, y_train)
exp_var_ratio_test, cum_sum_test, X_pretest, X_reduced_test = get_principal_components(test, y_test)

# Plot PCs
plot_principal_components(exp_var_ratio_train, cum_sum_train)
#plot_principal_components(exp_var_ratio_test, cum_sum_test)

# Plot projections 
#plot_projections(X_pretrain, X_reduced_train)
#plot_projections(X_pretest, X_reduced_test)

#Visualize Data in Grid View Heatmap

#Here we can concatenate the info if the patient has ALL or AML (0 for ALL, 1 for AML)
labels = pd.read_csv("actual.csv", index_col = 'patient')
cancer_type_dict = {'ALL':0,'AML':1}
labels.replace(cancer_type_dict, inplace=True)

#Drop the type column as no longer needed for the correlartion heatmap
test.drop('type', axis=1, inplace=True)
train.drop('type', axis=1, inplace=True)

#Translate indexes to respective numeric type values for sorting purposes
train.index = pd.to_numeric(train.index)
train.sort_index(inplace = True)
test.index = pd.to_numeric(test.index)
test.sort_index(inplace = True)

#Do the same for the labels
labels.index = pd.to_numeric(labels.index)
labels.sort_index(inplace = True)

# Index the scores with the cancer types and concatinate with respective rows in testing data
labels_train = labels[labels.index <= 38]
train = pd.concat([labels_train,train], axis = 1)
labels_test = labels[labels.index > 38]
test = pd.concat([labels_test,test], axis = 1)

# Replace all infinite values with the mean values for their respective dataframes
test.replace(np.inf, np.nan, inplace = True)
test.fillna(value = test.values.mean(), inplace = True)
train.replace(np.inf, np.nan, inplace = True)
train.fillna(value = train.values.mean(), inplace = True)

# Append the test to the train data to include the entire the sample space before we calculate the correlations
train_and_test = train.append(test)

#Index by the location of the labels extrated/outputted by the features_pca function
train_and_test = train_and_test.loc[:,extract_features(train_and_test)]
train = train_and_test[train_and_test.index < 36]
test = train_and_test[train_and_test.index >= 36]
#Drop the cancer column as no longer needed for the correlartion heatmap
train = train.drop(['cancer'], axis=1)
test = test.drop(['cancer'], axis=1)
print("Training data:")
print(train) #patients 1-35
print("Testing data:")
print(test) #patients 36-72

print(len(train.values[0]))
print(len(test.values[0]))

# get first 10 number of genes as feature vectors for each patient
X_train = train.values[:,:10]
X_test = test.values[:,:10]

print("Train vals:")
print(X_train)
print("Test vals:")
print(X_test)
# get first 36 labels, which those patients are training data labels
y_train = labels.values[:35]
# get the rest of labels for testing/predicting (if needed)
y_test = labels.values[35:]

print("Test labels")
print(y_test.flatten())

# Flatten NumPy arrays for labels
y_train = y_train.flatten()
y_test = y_test.flatten()

# Convert "0" for AML label to "-1" for SVM classification purposes
y_train[y_train == 0] = -1
y_test[y_test == 0] = -1

print(y_train)
print(y_test)

# Run SVM using non-linear classifier 
# 
# on training data - training parameters
# use gaussian kernel, with a sigma of 5.0

svm = SVM("gaussian", 5.0)
svm.fit(X_train, y_train)

# Run SVM to predict testing data - classification
svm.predict(X_test, y_test)

# Visualize the decision boundary contour plot
svm.visualize()


# Run SVM using linear classifier
#
# on training data, use linear kernel
# use a sample of data from genes M22960 and X62654 as features

# sample of training gene data from M22960 and X62654
            #-1
X_train2 = np.array([[330,631],
            [305,814],
            [196,712],
            #1
            [2179, 1888],
            [2501, 1907],
            [1885, 743]])

# # train labels from sample gene data
y_train2 = [-1, -1, -1, 1, 1, 1]

# # initialize linear classifier SVM
svmLinear = SVM("linear")

# # fit on training data - training parameters
svmLinear.fit(X_train2, y_train2)

# # sample of testing gene data from M22960 and X62654
            #-1
X_test2 = [[-178,649],
               [456,1935],
               [182,661],
               [265,1339],
               [207,586],
               #1
               [1850,1590],
               [2801,1292],
               [2444,1756],
               [2719,932]]

# # test labels from sample gene data
y_test2 = [-1, -1, -1, -1, -1, 1, 1, 1, 1]

# # Run SVM to predict testing data - classification
y_predict2 = svmLinear.predict(X_test2, y_test2)

# # Visualize the linear hyperplane decision boundary plot
svmLinear.visualize()


#Visualize the correlation grid for the training data
visualize_correlation_grid(train)
# #Visualize the correlation grid for the testing data
visualize_correlation_grid(test)

