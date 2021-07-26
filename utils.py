import random
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, ShuffleSplit, KFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, precision_score,recall_score, auc, cohen_kappa_score, roc_curve

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD

from imblearn.over_sampling import SMOTE
import hiplot as hip
from utils import *

matplotlib.rcParams['figure.figsize'] = (10, 5)
matplotlib.rcParams['font.size'] = 15
random.seed(0)
pd.set_option('display.max_columns', None)
pd.options.display.float_format = "{:,.9f}".format

import warnings
warnings.filterwarnings('ignore')


def plot_count_plot_with_hue(df, column_name, hue_column):
    ax = sns.countplot(x=column_name, data=df, hue=hue_column)
    plt.title(f'Distribution of  {column_name}')

    total = len(df[column_name])
    for p in ax.patches:
        percentage = '{:.2f}%'.format(100 * p.get_height()/total)
        x = p.get_x() + p.get_width()
        y = p.get_height()
        ax.annotate(percentage, (x, y),ha='center')
    plt.show()

def plot_count_plot(df, column_name):
    ax = sns.countplot(x=column_name, data=df)
    plt.title(f'Distribution of  {column_name}')

    total = len(df[column_name])
    for p in ax.patches:
        percentage = '{:.2f}%'.format(100 * p.get_height()/total)
        x = p.get_x() + p.get_width()
        y = p.get_height()
        ax.annotate(percentage, (x, y),ha='center')
    plt.show()
    
def plot_catplot(df,x,y):
    df1 = df.groupby(x)[y].value_counts(normalize=True)
    df1 = df1.mul(100)
    df1 = df1.rename('percent').reset_index()

    g = sns.catplot(x=x,y='percent',hue=y,kind='bar',data=df1, height=8.27, aspect=11.7/8.27)
    g.ax.set_ylim(0,100)
    
def plot_learning_curve(estimator1, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    f = plt.figure()
    ax1 = f.add_subplot(111)
    if ylim is not None:
        plt.ylim(*ylim)
    # First Estimator
    train_sizes, train_scores, test_scores = learning_curve(
        estimator1, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax1.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="#ff9124")
    ax1.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="#2492ff")
    ax1.plot(train_sizes, train_scores_mean, 'o-', color="#ff9124",
             label="Training score")
    ax1.plot(train_sizes, test_scores_mean, 'o-', color="#2492ff",
             label="Cross-validation score")
    ax1.set_title("Logistic Regression Learning Curve", fontsize=14)
    ax1.set_xlabel('Training size (m)')
    ax1.set_ylabel('Score')
    ax1.grid(True)
    ax1.legend(loc="best")
    
    return plt


def plot_confusion_matix(y_pred, y_true):
    cnf_mx_train = confusion_matrix(y_pred, y_true)
    df_cm = pd.DataFrame(cnf_mx_train.T, columns=np.unique(y_true), index = np.unique(y_true))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Actual'
    sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16},  fmt='g')
    plt.show()

def plot_roc_auc(model, X, y):
    probs = model.predict_proba(X)
    preds = probs[:,1]
    fpr, tpr, threshold = roc_curve(y, preds)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label='GridSearchCV (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()

def plot_metrics(model, X, y_true):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:,1]
    print('accuracy_score', accuracy_score(y_pred, y_true))
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    print('auc', auc(fpr, tpr))
    print('balanced_accuracy_score', balanced_accuracy_score(y_pred, y_true))
    print('cohen_kappa_score', cohen_kappa_score(y_pred, y_true))
    print(classification_report(y_pred, y_true))
    plot_confusion_matix(y_pred, y_true)
    plot_roc_auc(model, X, y_true)
    
def ranked_prediction(model, X_sample, prediction_columns, return_complete=False):
    X_sample['prediction'] = model.predict(X_sample[prediction_columns])
    X_sample['prediction_probs'] = model.predict_proba(X_sample[prediction_columns])[:,1]

    ranked_prediction = X_sample[(X_sample['prediction']==1)].sort_values(by=['matched_transaction_id', 'prediction_probs'], ascending =False)
    if return_complete:
        return ranked_prediction
    else:
        return ranked_prediction[['receipt_id', 'matched_transaction_id', 'feature_transaction_id', 'prediction_probs']]