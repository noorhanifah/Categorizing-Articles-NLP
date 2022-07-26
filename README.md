# Categorizing Articles (NLP) Using TensorFlow
 
## Project Description
Text documents are essential since they are one of the richest sources of data for businesses. Text documents often contain crucial information which might shape the market trends or influence the investment flows. Therefore, companies often hire analysts to monitor the trend via articles posted online, tweets on social media platforms such as Twitter or articles from newspaper. However, some companies may wish to only focus on articles related to technologies and politics. Thus, filtering of the articles into different categories is required.

Often the categorization of the articles is conduced manually and retrospectively; wasting so much time and resources due to this arduous task. The task can be done using Natural Language Processing or NLP techniques, which A huge volumes of digital text or articles are sorted according to their category. 

The objective of this project is to effectively categorize the unseen articles into 5 categories namely Sport, Tech, Business, Entertainment and Politics.

## Running the Project
This model run using Python programming and the libraries available. The training and validation graph are plotted using Tensorboard. TensorFlow library is used to develop and train this model.

## Project Insight
To achieve the objective of this project, recurrent neural network approach is used considering the nature of the dataset. This project also used neural network for NLP model training. Since this project is analyzing articles which is text, the text need to be converted into something that can be understand by machine. Therefore, tokenization is used which will break raw text into small parts called tokens. Embedding is also applied in this neural network allowing a large input to be processes in this model. Embedding is the process of converting high-dimensional data to low-dimensional data in the form of a vector in such a way that the two are semantically similar. Other than that, this model training also use Bidirectional LSTM instead of only LSTM as input flow. This is because the Bidirectional LSTM input, flows in both directions, forward and backwards, thus storing information used for future cell processing. 

###### The detail architecture of this model shown as below:
![Model pyt](https://github.com/noorhanifah/Categorizing-Articles-NLP/blob/main/Model_arch.PNG)

![Model](https://github.com/noorhanifah/Categorizing-Articles-NLP/blob/main/Plot_model_arch.png)

## Accuracy
After cleaning and training the data, this model acheive up to 0.8 accuracy. 

###### Below shows the training model evalution which shows 95& accuracy.
![Training model evaluation](https://github.com/noorhanifah/Categorizing-Articles-NLP/blob/main/model_evaluation.PNG)

###### Based on the classification report this model give 0.95 accuracy with f1-score of more than 0.9. This shows that this model is able to predict the all five outcomes expected from this model. Therefore, the ability to categorize articles into Sport, Tech, Business, Entertainment and Politics can be achieve throught this model.
![Correlation](https://github.com/noorhanifah/Categorizing-Articles-NLP/blob/main/classification_report.PNG)

###### The best model out of all the aproach is Logistic Regression with Standard Scalar as they give a score of 0.824. Thus, will be selected for this project. 
![Best Model](https://github.com/noorhanifah/Heart-Attack-Prediction-Using-Machine-Learning/blob/main/Score/best_model.PNG)

###### Based on the classification report this model give 0.82 accuracy with f1-score of 0.82 and 0.83. This shows that this model is able to predict the two outcome expected from this model. Therefore, to know wheather someone has the possibility of having heart attack or not can be achieve throught this model.
![CR](https://github.com/noorhanifah/Heart-Attack-Prediction-Using-Machine-Learning/blob/main/Score/classification_report.PNG)

###### The training and the validation accuracy of this model can be observed from the plotted graph. From the graph, this model is able to learn at some point.
![Training and validation accuracy](https://github.com/noorhanifah/Categorizing-Articles-NLP/blob/main/validation_training_accuracy.PNG)

###### TensorBoard also is used to plot the all the training graph. 
![TensorBoard](https://github.com/noorhanifah/Categorizing-Articles-NLP/blob/main/Tensorboard/Tensorboard.PNG)

###### The accuracy graph shown on TensorBoard:
![TensorBoard Accuracy](https://github.com/noorhanifah/Categorizing-Articles-NLP/blob/main/Tensorboard/tensorboard_accuracy.PNG)

## A little discussion
For this dataset I belief that, a higher accuracy can be achieve through other approach of data cleaning and selecting features. For a better prediction especilly for medical purpose, a small 300 hundred data is not enough as there are many possibilities could occur. For example, in this dataset, age has a relatively low correlation with the outcome of having a heart attack. Older people is known to have a higher risk of having a heart attack, however, it does not apply to this dataset. Therefore, more testing need to be done to determine the ability of this model to make a better prediction.

## Build With
 ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
 ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
 ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
 ![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
 ![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)
 ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
 ![GoogleColab](	https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)

## Credit
The dataset can be downloaded from Kaggle dataset at https://www.kaggle.com/rashikrahmanpritom/heart-attack-analysis-prediction-dataset. 

Creator of the dataset.
1. Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.

For more dataset information: 
http://archive.ics.uci.edu/ml/datasets/Heart+Disease
