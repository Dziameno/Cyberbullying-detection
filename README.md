# Cyberbullying-detection
### Based on PolEval 2019 Task 6: Automatic Cyberbllying detection

### http://2019.poleval.pl/index.php/tasks/task6

### This project is based only on Task 6-1: Harmful vs non-harmful

#### Results after preprocessing:
``` 
class of tweets     training set    testing set

non-harmful             8608            821
harmful                  753            121
```

#### Results of SVC classifier:
Acc score: 88.85%
```
                predicted: 
actual:                         postive  negative
       positive                   816       5
       negative                   100      21
```

```
                precision    recall  f1-score   support

0-non-harmful       0.89      0.99      0.94       821
1-harmful           0.81      0.17      0.29       121

    accuracy                            0.89       942
   macro avg        0.85      0.58      0.61       942
weighted avg        0.88      0.89      0.86       942
```
 - Model has high accuracy for non-harmful tweets <br>
 - Class 0 has more instances than class 1, so model is biased to class 0.

#### Results of SVC classifier with class_weight='balanced':
Acc score: 89.38%
```
                predicted: 
actual:                         postive  negative
       positive                   795       26
       negative                    74       47
       
```
```
               precision    recall  f1-score   support

           0       0.91      0.97      0.94       821
           1       0.64      0.39      0.48       121

    accuracy                           0.89       942
   macro avg       0.78      0.68      0.71       942
weighted avg       0.88      0.89      0.88       942
```


#### Augmentation:
- back translation of harmful tweets polish->english->polish

Acc score: 88.21%
```
                predicted: 
actual:                         postive  negative
       positive                   818       3
       negative                   108      13
```

```
                precision    recall  f1-score   support

0-non-harmful       0.88      1.00      0.94       821
1-harmful           0.81      0.11      0.19       121

    accuracy                            0.88       942
   macro avg        0.85      0.55      0.56       942
weighted avg        0.87      0.88      0.84       942
```
- f1 is lower than without augmentation


