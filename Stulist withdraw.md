# Hyperparameter Log


Metric reminders:
    ROC AUC = How well your model seperates the two classes. 1 is perfect
    LOG LOSS = How close predicted probablities are. 0 is perfect
    Precision = tof all predicted positives how many where correct, higher means less fals alarms
    Recall = of all actual positives, how many did the model catch. Higher means fewer incorrects
    F1 = mean of precision and recall - higher is better.




## Experiment 001 - Baseline Logistic Regression
**Date:** 2025-06-27  
**Model:** LogisticRegression  
**Params:**  
- C = 1.0  
- max_iter = 1000  
- class_weight = 'balanced' 
- Threshold = 0.5851

**Preprocessing:**  
- TF-IDF (max_features=100)  
- StandardScaler  
- SimpleImputer (mean)  

**Results:**  
- Confusion Matrix:
    [[38696  4950]
    [ 2560 12326]] 
- Mean ROC-AUC: 0.9269
- ROC AUC: 0.9269 
- Log Loss: 0.3299
- Precision: 0.7135
- Recall: 0.8280
- F1 Score: 0.7665
 
**Notes:**  
Baseline run. Great ROC so really good at understanding what withdrawing looks like but need to tighten make it more confident and boost precision. Going to try adjusting the decision threshold, plotting f1 to try find an optimum




## Experiment 002 - Ratio adjusted Logistic Regression
**Date:** 2025-06-27  
**Model:** LogisticRegression  
**Params:**  
- C = 1.0  
- max_iter = 1000  
- class_weight = 'balanced'
- Threshold - *0.5851*

**Preprocessing:**  
- TF-IDF (max_features=100)  
- StandardScaler  
- SimpleImputer (mean)  

**Results:**  
- Mean ROC-AUC: 0.9269
- Log Loss:       0.3299
- ROC AUC:        0.9269
- Precision:      0.7866
- Recall:         0.7840
- F1 Score:       0.7853

**Notes:** 
Adjusted ratio seems to have increased precision a little bit and lowere recall - as exepected i suppose. Log loss is target of next iteration as this is a little higher then desired.




## Experiment 003 - Increased vetorizer feature max
**Date:** 2025-06-27  
**Model:** LogisticRegression  
**Params:**  
- C = 1.0  
- max_iter = 1000  
- class_weight = 'balanced'
- Threshold - 0.5851

**Preprocessing:**  
- TF-IDF (max_features=*500*)  
- StandardScaler  
- SimpleImputer (mean)  

**Results:**  
- Confusion Matrix:
    [[41147  2499]
    [ 1645 13241]]
- Mean ROC-AUC: 0.9682
- Log Loss:       0.2195
- ROC AUC:        0.9682
- Precision:      0.8412
- Recall:         0.8895
- F1 Score:       0.8647

**Notes:** 
Damn adjusted max features has reduced log loss alot! More features = bigger signal and less noise




## Experiment 004 - Major Sanity check 
**Date:** 2025-06-30 
**Model:** LogisticRegression  
**Params:**  
- C = 1.0  
- max_iter = 1000  
- class_weight = 'balanced'
- Threshold - 0.5851

**Preprocessing:**  
- TF-IDF (max_features=500)  
- StandardScaler  
- SimpleImputer (mean)  

**Results:**  
-Confusion Matrix:
    [41147  2499]
    [ 1645 13241]
- ROC-AUC scores per fold: [0.96633361 0.96615575 0.97145263 0.97044483 0.96675519]
- Mean ROC-AUC: 0.9682
- Log Loss:       0.2195
- ROC AUC:        0.9682
- Precision:      0.8412
- Recall:         0.8895
- F1 Score:       0.8647
-        Feature  Coefficient   Direction
    0    withdrawn    24.014446  Withdrawal
    1        comms    19.458463  Withdrawal
    2         type    16.475413  Withdrawal
    3       triage    12.309878  Withdrawal
    4    applicant    11.324300  Withdrawal
    10         mtt    -8.781702   Retention
    11         app    -4.696804   Retention
    12      attend    -4.414236   Retention
    13         did    -3.748225   Retention 

**Notes:** 
Major problems...Got great numbers but feature list reveals that its cheating through reading the openaction. Triage is also *obviously* going to be a major source of withdrawel but both of these things arent great indicators of measuring withdrawal. I think the ideal final output is a *reason for guess* type output so that we can see why the pipeline thinks they will withdraw. Really unsure if i should just remove openaction from the df.... Intrestingly MTT seems to be the top reason the algorithm assigns retention. 



## Experiment 005 - Major Sanity check pt.2
**Date:** 2025-06-30 
**Model:** LogisticRegression  
**Params:**  
- C = 1.0  
- max_iter = 1000  
- class_weight = 'balanced'
- Threshold - 0.5851
- Dropped column 'OpenAction'

**Preprocessing:**  
- TF-IDF (max_features=500)  
- StandardScaler  
- SimpleImputer (mean)  

**Results:**  
- Confusion Matrix:
    [[37103  6543]
    [ 5554  9332]]

- ROC-AUC scores per fold: [0.80282981 0.80066143 0.80423521 0.80483335 0.79887452]
- Mean ROC-AUC: 0.8023
- Log Loss:       0.5292
- ROC AUC:        0.8023
- Precision:      0.5878
- Recall:         0.6269
- F1 Score:       0.6067

-       Feature  Coefficient   Direction
0       teaching     4.527736  Withdrawal
1         online     4.347055  Withdrawal
2      practical     3.498317  Withdrawal
3     standalone     2.953106  Withdrawal
10  computerised    -2.926882   Retention
11         level    -2.550249   Retention
12            ei    -2.409732   Retention
13           toe    -2.401034   Retention

**Notes:** 
ooookaayy this time i ran it without openaction so we can see how bias that column makes it. as it turns out - pretty big loss in stats. most notably log loss is indicating practically random guessing. At least Roc suggests it knows what is withdraw and isnt but both precision and recall are pretty bad i.e alot of false positives and alot of incorrect




## Experiment 006 - Recovery
**Date:** 2025-06-30 
**Model:** *RandomTrees*
**Params:**  
- *n_estimators=200*
- *max_depth=20*
- *min_samples_split=5*
- *min_samples_leaf=2*
- *max_features='sqrt'*
- *class_weight='balanced'*
- *verbose=1*
- *random_state=42*
- *n_jobs=-1*


**Preprocessing:**  
- TF-IDF (max_features=500)  
- StandardScaler  
- SimpleImputer (mean)  

**Results:**  
- Confusion Matrix:
    [[39846  3800]
    [ 6117  8769]]

- ROC-AUC scores per fold: [0.86860226 0.86300847 0.87028826 0.86528948 0.86524315]
- Mean ROC-AUC: 0.8665 
- Log Loss:       0.4585
- ROC AUC:        0.8664
- Precision:      0.6977
- Recall:         0.5891
- F1 Score:       0.6388

**Notes:** 
New model!! Wow ok random trees has improved the situation a bit - mean roc is still pretty high i.e it still knows whats withdraw and retain pattern, log loss is better too! that makes sense - withdrawal isnt linearely correlated to alot of the data we are testing. Bit lopsided on the precision vs recall, we seem to favour preventing false positives then catching every true positive. Going to adjust the ration to try and catch more true positives as well as experiment with leafs and splits. Increasing N estimators seems to be a good shout - will need to start logging training time as its starting to get longer.




## Experiment 007 - Calibration
**Date:** 2025-06-30 
**Model:** RandomTrees
**Params:**  
- *n_estimators=500*
- *max_depth=30*
- *min_samples_split=10*
- *min_samples_leaf=4*
- *max_features='sqrt'*
- class_weight='balanced'
- verbose=1
- random_state=42
- n_jobs=-1
- Threshold - 0.5


**Preprocessing:**  
- TF-IDF (max_features=500)  
- StandardScaler  
- SimpleImputer (mean)  

**Results:**  
- Confusion Matrix:
    [[38229  5417]
    [ 4809 10077]]

- ROC-AUC scores per fold: [0.87178702 0.86781579 0.87587897 0.8711176  0.87101551]
- Mean ROC-AUC: 0.8715
- Log Loss:       0.4429
- ROC AUC:        0.8715
- Precision:      0.6504
- Recall:         0.6769
- F1 Score:       0.6634

**Notes:** 
Increased all hyperparameters to see what would happen. Graph revealed 0.5196 was best ratio for balanced F1. we see an minor increase in ROC, a small decrease in log loss and similar minor flucations in other stats. notably recall is much higher trading with a slightly lower precision. MCC is 0.546 - 0 means no better then random and 1 is perfect. model is reasonable. feature graph labled as  007 - Importantfeatures
