# Model Card
## Model Details
- Model Type: Random Forest Classifier
- Training Framework: scikit-learn 1.5.1
- Model Inputs:
  - Categorical features: workclass, education, marital-status, occupation, relationship, race, sex, native-country 
  - Continuous features: age, fnlgt, capital-gain, capital-loss, hours-per-week
- Model Outputs: Binary classification output for salary prediction 
- Predicted classes: >50K, <=50K

## Intended Use
This model is intended to predict an individual's income based on demographic and employment features. Primary users 
would be data scientists, social and economic researchers, or businesses analyzing demographics.

## Training Data
- Dataset: UCI Census Income Dataset
- Source: https://archive.ics.uci.edu/dataset/20/census+income
- Size: 32,561 samples (80% for training)
- Features: The dataset contains 14 features, including categorical and continuous variables, describing personal 
information such as age, education, and work status. 
- Preprocessing:
  - Categorical variables are one-hot encoded using a OneHotEncoder. 
  - The target variable "salary" is binarized into >50K and <=50K using a LabelBinarizer. 
  - Missing categorical values (represented as '?') are handled by replacing them with a placeholder string 'missing'.

## Evaluation Data
- Dataset: Same as the training data, split into 20% for evaluation.
- Performance: The evaluation is conducted on a separate test set that was not seen during the training phase to assess model generalization.

## Metrics
The model is evaluated using the following metrics (computed using a test set of 6,513 samples):
- Precision: 0.7174 
- Recall: 0.6356
- F1 Score: 0.6740

## Ethical Considerations
- Because the model is built on sensitive demographic data such as race and sex, this could lead to biased predictions. 
It is essential that the potential for bias is fully understood and top-of-mind when deploying this model in the real
world.

## Caveats and Recommendations
- Model limitations:
  - The model is based on US Census data. Attempting to use this model for predictions in other parts of the world, or
  on datasets where the demographics differ significantly from this model's dataset, may cause degraded performance.
- Recommendations:
  - Retrain the model periodically to ensure that it keeps up with shifting demographics and economic trends.
  - Be particularly careful if making high-stakes decisions, as this model has the potential to make biased, unfair
  predictions against demographic groups.
