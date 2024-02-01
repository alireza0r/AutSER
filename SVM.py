from sklearn import svm
from feature_extractor import featureExtractor
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

features, labels = featureExtractor()
clf = svm.SVC()

#score = cross_val_score(clf, np.mean(features, axis=1), labels)

#train_fold = []
#test_fold  = []
KF = StratifiedKFold(n_splits=5, shuffle=True)
for i, (x_index, y_index) in enumerate(KF.split(np.mean(features, axis=1), labels)):
    clf.fit(np.mean(features[x_index], axis=1), labels[x_index])
    #train_pred = clf.predict(np.mean(features[x_index], axis=1), labels[x_index])
    #test_pred  = clf.predict(np.mean(features[y_index], axis=1), labels[y_index])
    #train_fold.append(train_pred)
    #test_fold.append(test_pred)

    print('-------------------')
    print('Fold:', i)
    print('Train:', clf.score(np.mean(features[x_index], axis=1), labels[x_index]))
    print('Test:', clf.score(np.mean(features[y_index], axis=1), labels[y_index]))

    disp = ConfusionMatrixDisplay.from_estimator(clf, 
                                                 np.mean(features[x_index], axis=1), 
                                                 labels[x_index], 
                                                 display_labels=['H', 'A', 'S', 'N'],
                                                 normalize='true')
    plt.title(f'Train, Fold {i+1}')
    plt.show()

    disp = ConfusionMatrixDisplay.from_estimator(clf, 
                                                 np.mean(features[y_index], axis=1), 
                                                 labels[y_index],
                                                 display_labels=['H', 'A', 'S', 'N'],
                                                 normalize='true')
    plt.title(f'Test, Fold {i+1}')
    plt.show()
