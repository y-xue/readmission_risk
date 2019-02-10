# Predicting ICU readmission using grouped physiological and medication trends. [Full paper](https://www.ncbi.nlm.nih.gov/pubmed/30213670).

## Background: 
Patients who are readmitted to an intensive care unit (ICU) usually have a high risk of mortality and an increased length of stay. ICU readmission risk prediction may help physicians to re-evaluate the patient’s physical conditions before patients are discharged and avoid preventable readmissions. ICU readmission pre- diction models are often built based on physiological variables. Intuitively, snapshot measurements, especially the last measurements, are effective predictors that are widely used by researchers. However, methods that only use snapshot measurements neglect predictive information contained in the trends of physiological and medi- cation variables. Mean, maximum or minimum values take multiple time points into account and capture their summary statistics, however, these statistics are not able to catch the detailed picture of temporal trends. In this study, we find strong predictors with ability of capturing detailed temporal trends of variables for 30-day readmission risk and build prediction models with high accuracy.

## Methods: 
We study physiological measurements and medications from the Multiparameter Intelligent Monitoring in Intensive Care II (MIMIC-II) clinical dataset. Time series of each variable are converted into trend graphs with nodes being discretized measurements of each variable. Then we extract important temporal trends by applying frequent subgraph mining on the trend graphs. The frequency of a subgraph is a good cue to find important temporal trends since similar patients often share similar trends regarding their pathophysiological evolution under medical interventions. Important temporal trends are then grouped automatically by non-ne- gative matrix factorization. The grouped trends could be considered as an approximate representation of pa- tients’ pathophysiological states and medication profiles. We train a logistic regression model to predict 30-day ICU readmission risk based on snapshot measurements, grouped physiological trends and medication trends.

## Results: 
Our dataset consists of 1170 patients who are alive 30 days after discharge from ICU and have at least 12 h of data. In the dataset, 860 patients were not readmitted and 310 were readmitted, within 30 days after discharge. Our model outperforms all comparison models, and shows an improvement in the area under the receiver operating characteristic curve (AUC) of almost 4% from the best comparison model.

## Conclusions: 
Grouped physiological and medication trends carry predictive information for ICU readmission risk. In order to build predictive models with higher accuracy, we should add grouped physiological and medication trends as complementary features to snapshot measurements.