# !pip install tpot


from tpot import TPOTClassifier
import pandas as pd
import numpy as np
import os
import csv
import random
from sklearn.model_selection import (train_test_split,KFold,cross_validate)
import sys
from sklearn.metrics import (accuracy_score,precision_score,recall_score,confusion_matrix)
from ConfusionMatrixCalc import precisionPerClass, recallPerClass
import pickle
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer
#
### Those 4 commands can be run to use TPOT with the 4 datasets
## The last one does not work. We could not find a reason for it. Maybe it is due to the large size of the dataset.
#
# Arguments
# 1. Argument: Name of Output-Files
# 2. Argument: Filepath for CSV
# 3. Argument: Name of target variable
# 4. Argument: Number of minutes to run the algorithm
# 5. Argument: Use 1% fraction of dataset?
#
# python TPOT_executable.py "bankchurners" "datasets/bankChurners/bc_cleaned.csv" "Attrition_Flag" 2 False
# python TPOT_executable.py "hypothyroid" "datasets/hypothyroid/ht_cleaned.csv" "Class" 2 False
# python TPOT_executable.py "breastcancer" "datasets/breastcancer/breast-cancer-diagnostic_cleaned.csv" "MALIGNANT" 2 False
# python TPOT_executable.py "pokerhand" "datasets/pokerhand/pokerhand-normalized_cleaned.csv" "class" 2 True
#
# Command-Line Inputs
dataset_name = sys.argv[1]  # Name of Output-Files
filepath = sys.argv[2]      # Filepath for CSV
target_name=sys.argv[3]     # Name of target variable
max_time_mins=int(sys.argv[4])   # Number of minutes to run the algorithm

#
#
#
def customPrecision(y_true,y_pred):
    return precision_score(y_true,y_pred,average="micro")
def customRecall(y_true,y_pred):
    return recall_score(y_true,y_pred,average="micro")
#
### Important Parameters
#
population_size = 100
n_jobs=-3
scoring = "accuracy"
cv = KFold(n_splits=5, random_state=123, shuffle=True)
random_state=123
scoring_methods = {"accuracy":'accuracy',"precision":make_scorer(customPrecision),"recall":make_scorer(customRecall)}
#
### Getting the dataset
#
df = pd.read_csv(filepath)
if sys.argv[5]:
    df=df.sample(frac=0.01)
try:
    df = df.drop(["Unnamed: 0"],axis=1)
except:
    pass
df[target_name], target_classes = pd.factorize(df[target_name])
X = df[df.columns[1:]]
y = df[df.columns[0]]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)
#
### Dictionary for all possible pipeline-steps
#
config_dict = {

    # Classifiers
    'sklearn.naive_bayes.MultinomialNB': {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]
    },

    'sklearn.ensemble.RandomForestClassifier': {
        'n_estimators': [100],
        'criterion': ["gini", "entropy"],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf':  range(1, 21),
        'bootstrap': [True, False]
    },
    
    'sklearn.neighbors.KNeighborsClassifier': {
        'n_neighbors': range(1, 101)[::4],
        'weights': ["uniform", "distance"],
        'p': [1, 2]
    },
    
    'sklearn.svm.LinearSVC': {
        'penalty': ["l1", "l2"],
        'loss': ["hinge", "squared_hinge"],
        'dual': [True, False],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.]
    },

    'sklearn.linear_model.LogisticRegression': {
        'penalty': ["l1", "l2"],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'dual': [False]
    },
    
    # Different Kernels 
    'sklearn.kernel_approximation.Nystroem': {
        'kernel': ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly', 'linear', 'additive_chi2', 'sigmoid'],
        'gamma': np.arange(0.0, 1.01, 0.05),
        'n_components': range(1, 11)
    },
    
    # Decomposition/Aggregation
    'sklearn.decomposition.PCA': {
        'svd_solver': ['randomized'],
        'iterated_power': range(1, 11)
    },
    
    'sklearn.cluster.FeatureAgglomeration': {
        'linkage': ['ward', 'complete', 'average'],
        'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
    },
    
    #Scaling
    'sklearn.preprocessing.RobustScaler': {
    },

    'sklearn.preprocessing.StandardScaler': {
    },
    
    'sklearn.preprocessing.MinMaxScaler': {
    },

    
    # Selectors
    'sklearn.feature_selection.VarianceThreshold': {
        'threshold': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    },

    'sklearn.feature_selection.RFE': {
        'step': np.arange(0.05, 1.01, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesClassifier': {
                'n_estimators': [100],
                'criterion': ['gini', 'entropy'],
                'max_features': np.arange(0.05, 1.01, 0.05)
            }
        }
    },

    'sklearn.feature_selection.SelectFromModel': {
        'threshold': np.arange(0, 1.01, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesClassifier': {
                'n_estimators': [100],
                'criterion': ['gini', 'entropy'],
                'max_features': np.arange(0.05, 1.01, 0.05)
            }
        }
    }
}


#
### Main Function
#
if __name__ == "__main__":
    #
    ### Running the whole evolutionary algorithm for the selected time
    #
    pipeline_optimizer = TPOTClassifier(scoring=scoring,
        	                        population_size=population_size,
                                    n_jobs=n_jobs,             # Number of available CPUs - 1 
                                    max_time_mins=max_time_mins,  # Max time for the run in minutes 
                                    random_state=random_state,
                                    config_dict=config_dict,     
                                    warm_start=False,      # not used
                                    use_dask=True,         # More efficient optimization
                                    cv = cv,    # 5-Fold CV without stratification
                                    periodic_checkpoint_folder=None, # Progress-Tracking would be possible
                                    verbosity = 2) # Progress-Bar in Console
    pipeline_optimizer.fit(X_train,y_train)
    print("Best Pipeline:")
    print(pipeline_optimizer.fitted_pipeline_)
    best_pipeline = pipeline_optimizer.fitted_pipeline_
    #
    ### Adding an Imputer in front of the best pipeline and using cross-validation on the training set to get performance results
    #
    imputed_pipeline = Pipeline([("imputer",SimpleImputer(strategy="median")),("best",best_pipeline)])
    scores_cv=cross_validate(estimator=imputed_pipeline, X=X_train, y=y_train, cv=cv, scoring = scoring_methods)
    imputed_pipeline.fit(X_train,y_train)
    preds=imputed_pipeline.predict(X_test)
    #
    ### Result dictionary that can be used for visualization
    #
    scores_test = dict()
    scores_test["classes"] = best_pipeline.classes_
    scores_test["best_pipeline"] = best_pipeline
    scores_test["cv_accuracy_mean"] = np.mean(scores_cv["test_accuracy"])
    scores_test["cv_accuracy_var"] = np.var(scores_cv["test_accuracy"])
    scores_test["cv_precision_mean"] = np.mean(scores_cv["test_precision"])
    scores_test["cv_precision_var"] = np.var(scores_cv["test_precision"])
    scores_test["cv_recall_mean"] = np.mean(scores_cv["test_recall"])
    scores_test["cv_recall_var"] = np.var(scores_cv["test_recall"])
    scores_test["accuracy"] = accuracy_score(y_test,preds)
    scores_test["precision"] = precision_score(y_test,preds,average='micro')
    scores_test["recall"] = recall_score(y_test,preds,average='micro')
    scores_test["conf_matrix"] = confusion_matrix(y_test,preds)
    scores_test["precision_per_class"] = precisionPerClass(scores_test["conf_matrix"])
    scores_test["recall_per_class"] = recallPerClass(scores_test["conf_matrix"])
    #
    ### Saving the result dictionary with pickle
    #
    with open('tpot_results/'+dataset_name + '.pickle', 'wb') as handle:
        pickle.dump(scores_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Test-Scores Dictionary:")
    print(scores_test)
    #
    ### Saving the statistics for each generation in an accessible format
    #
    testedIndividuals=pipeline_optimizer.evaluated_individuals_
    generationMeanDict=dict() 
    testedIndividuals=pipeline_optimizer.evaluated_individuals_
    for key in testedIndividuals:
        currentGen = testedIndividuals[key]["generation"]
        if generationMeanDict.get(currentGen) is not None:
            generationMeanDict[currentGen].append(testedIndividuals[key]["internal_cv_score"])
        else:
            generationMeanDict[currentGen] = [testedIndividuals[key]["internal_cv_score"]]
    progressDict=dict()
    for generation in generationMeanDict:
        progressDict[generation]=dict()
        progressDict[generation]["mean"]=np.mean([x for x in generationMeanDict[generation] if (x <=1 and x>=0)])
        progressDict[generation]["variance"]=np.var([x for x in generationMeanDict[generation] if (x <=1 and x>=0)])
    print("Average Performance per Generation:")
    print(progressDict)  
    with open('tpot_results/'+dataset_name + '_progress.pickle', 'wb') as handle:
        pickle.dump(progressDict, handle, protocol=pickle.HIGHEST_PROTOCOL)