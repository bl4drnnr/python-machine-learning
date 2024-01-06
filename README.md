<h1 align="center">
  Python Machine Learning
</h1>

1. [Introduction](#introduction)
2. [Description of the research set](#description-of-the-researc-set)
3. [Data preprocessing](#data-preprocessing)
  1. [Histograms](#histograms)
  2. [Outliears](#outliears)

# Introduction

This project focuses on leveraging the abalone dataset to train six distinct machine learning models. The abalone dataset, renowned for its diverse features related to abalone species, provides a rich foundation for developing and evaluating these models. The primary objective is to harness the dataset's information to enhance the predictive capabilities of the machine learning models.

The six machine learning models employed in this project are carefully chosen to cover a spectrum of algorithms, ensuring a comprehensive exploration of predictive techniques. Each model is tailored to handle specific aspects of the abalone dataset, contributing to a robust and versatile solution.

Throughout the project, the emphasis is not only on training the models but also on optimizing their performance. This involves fine-tuning hyperparameters, implementing feature engineering strategies, and employing cross-validation techniques to ensure the models generalize well to unseen data.

The dataset's attributes, such as the physical measurements of abalones, are carefully analyzed to extract meaningful insights. Feature importance and correlation analyses are conducted to better understand the impact of individual attributes on the models' predictions.

The project also includes a thorough evaluation phase, where the models' performance metrics are scrutinized to assess their accuracy, precision, recall, and other relevant metrics. This evaluation provides valuable feedback for refining and selecting the most effective models.

Furthermore, the project incorporates best practices in machine learning, including data preprocessing techniques to handle missing values or outliers, scaling features for better convergence, and addressing potential biases in the dataset.

The ultimate goal of this project is to not only build accurate machine learning models but also to gain a deeper understanding of the underlying patterns within the abalone dataset. By doing so, the project contributes to the broader field of machine learning and reinforces the significance of leveraging diverse datasets for comprehensive model training and evaluation.

# Description of the research set 

First of all, to conduct a study, you need to select a data set. Selecting the appropriate data set is an important step in the machine learning process. The choice of dataset can have a significant impact on the performance and generalization of a machine learning model. There are many different parameters and variables to consider when choosing. This includes the relevance of the data, the size of the database, the quality of the data, the presence or absence of any values, whether the learning will be supervised or unsupervised, etc.

This project will use a dataset with a decision attribute because supervised machine learning will be used. There are many different sources from which you can obtain a free database for research use. In the case of this project, such a database was a database called Abalone from [archive.ics.uci.edu/dataset/1/abalone](https://archive.ics.uci.edu/dataset/1/abalone).

Predicting the age of abalone from physical measurements. The age of an abalone is determined by cutting the shell through the cone, staining it and counting the number of rings under a microscope - a tedious and time-consuming task. Other measurements that are easier to obtain are used to predict age.

This database contains 9 attributes, 1 of which - Rings - is a decision attribute: either as a continuous value or as a classification problem. The entire set contains 4,177 records. Below is a table describing which attributes are presented in this database, their names, units and types. It is worth paying attention to the fact that there are no missing values in this database, which will definitely facilitate the process of processing this data.

# Data preprocessing

Data preprocessing in machine learning refers to the systematic and methodical manipulation of raw data before entering it into machine learning algorithms. This essential step involves cleaning, transforming, and structuring the data to improve its quality, fill in missing values, and ensure compatibility with the selected model, thereby optimizing the overall performance and reliability of the AI system during the training process.

## Histograms

Data preprocessing can start by creating various charts to visualize the data. This step is very important because it allows you to visually assess which parameters of a specific data set will be more or less important and which can be completely ignored.

![hist_diameter](abalone/pics/hist_diameter.png)

![hist_height](abalone/pics/hist_height.png)

![hist_length](abalone/pics/hist_length.png)

![hist_rings](abalone/pics/hist_rings.png)

![hist_sex](abalone/pics/hist_sex.png)

![hist_shell_weight](abalone/pics/hist_shell_weight.png)

![hist_shucked_weight](abalone/pics/hist_shucked_weight.png)

![hist_viscera_weight](abalone/pics/hist_viscera_weight.png)

![hist_whole_weight](abalone/pics/hist_whole_weight.png)

## Outliears

**Outlier** – in machine learning, an outlier refers to an observation or data point that deviates significantly from the majority of the data set. Outliers can introduce noise and distort overall patterns in the data, which can lead to inaccurate or biased models. Identifying and handling outliers is crucial during data pre-processing in machine learning to ensure that the model is trained on reliable and representative information. Techniques such as statistical methods, visualization, and mathematical models are used to detect and resolve outliers, which contributes to the reliability and accuracy of the AI system.

An example of such values may be human body temperature. For example, the body temperature of a living person cannot be lower than 0 or higher than 100 degrees Celsius. As mentioned above, such values may appear during incorrectly entered values or may be some kind of anomaly. An important explanation is that outliers are very context-dependent. This means that technically a person's body temperature can be lowered below 0 degrees Celsius or heated above 100 degrees Celsius, but these are unlikely to be normal values for a living person.

As mentioned above, charts may not be sufficient to provide the necessary information about outliers. To obtain information about the data network on which preprocessing takes place, there is a special method in the Pandas package used to read the data that allows you to obtain such information – `describe`.

```python
data = pd.read_csv(’abalone/abalone.data’, names=names)
print(data.describe().T) # T means transposition
```

```
                 count      mean       std     min     25%     50%     75%      max
length          4177.0  0.523992  0.120093  0.0750  0.4500  0.5450   0.615   0.8150
diameter        4177.0  0.407881  0.099240  0.0550  0.3500  0.4250   0.480   0.6500
height          4177.0  0.139516  0.041827  0.0000  0.1150  0.1400   0.165   1.1300
whole_weight    4177.0  0.828742  0.490389  0.0020  0.4415  0.7995   1.153   2.8255
shucked_weight  4177.0  0.359367  0.221963  0.0010  0.1860  0.3360   0.502   1.4880
viscera_weight  4177.0  0.180594  0.109614  0.0005  0.0935  0.1710   0.253   0.7600
shell_weight    4177.0  0.238831  0.139203  0.0015  0.1300  0.2340   0.329   1.0050
rings           4177.0  9.933684  3.224169  1.0000  8.0000  9.0000  11.000  29.0000
```

The so-called [**interquartile range (IQR)**](https://en.wikipedia.org/wiki/Interquartile_range) – is a statistical measure used to assess the spread or dispersion of a data set. This is especially useful for identifying and handling outliers. IQR is defined as the range between the first quartile (Q1) and the third quartile (Q3) of a data set. It works as follows:

1. **Q1 (first quartile)**: The value below which 25% of the data falls.
2. **Q3 (third quartile)**: The value below which 75% of the data falls.
3. **IQR = Q3 - Q1**: represents the distribution of the middle 50% of the data. A larger IQR indicates greater variability in the central part of the data set.
4. **Outliers**: Values below Q1 − 1.5 * IQR or above Q3 + 1.5 * IQR are often considered potential outliers.

The code implementing this mathematical function for the DataFrame data type in which the examined data set is stored will be presented and described below:

```python
def find_and_delete_outliers(df):
    numeric_columns = df.select_dtypes(include=[’float64’, ’int64’]).columns
    Q1 = df[numeric_columns].quantile(0.25)
    Q3 = df[numeric_columns].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_mask = ((df[numeric_columns] < lower_bound) | (df[numeric_columns] > upper_bound)).any(axis=1)
    return df[~outliers_mask]
```