# Charity Analysis with a Neural Network

## Challenge - Overview

To continue our exploration into Machine Learning we take a deep dive into Neural Nets, and the TensorFlow module. For this analysis we aim to create a Binary Classifier Neural Net that would be capable of predicting whether applicants will be successful if funded by the company Alphabet Soup.

We read in a dataset containing details about every organization to receive funding, and whether the money was used successfully. We use this to train a model and have that model predict the whether a company will be successful. We split the data into training and testing and validate the performance of the model based on how correct the test predictions are.

## Challenge - Results

From our analysis we try to answer the following questions:

* **Data Pre-processing**:
  * What variable(s) are considered the target(s) for your model?
    * We use the `IS_SUCCESSFUL` column to be our target variable. This is what we hope to predict with the testing data.
  * What variable(s) are considered to be the features for your model?
    * After the data cleaning any variable in the dataset that is not `IS_SUCCESSFUL` is considered an independent variable that we will use to predict the target variable with.
    * These include: `APPLICATION_TYPE`,	`AFFILIATION`,	`CLASSIFICATION`,	`USE_CASE`,	`ORGANIZATION`,	`STATUS`,	`INCOME_AMT`,	`SPECIAL_CONSIDERATIONS`,	and `ASK_AMT`. We organize the numerical data with our data cleaning, as well as convert the categorical data into numerical values too.
  * What variable(s) are neither targets nor features, and should be removed from the input data?
    * In our original model we decide that `EIN` and `NAME`, identification columns, are not worth using to predict the success of the organization, so they are dropped at the beginning of the transforming/analysis.
* **Compiling, Training, and Evaluating the Model**:
  * How many neurons, layers, and activation functions did you select for your neural network model, and why?
    * For the challenge-follow-along we were guided to use:
      * 2 hidden layers, with 80 and 30 neurons respectively.
      * 1 output layer.
      * 2 different activation functions, `relu` and `sigmoid`.
    * For the optimization we experimented with a number of different options, most didn't improve much on the model predictions, however we settled on:
      * 3 hidden layers, with 100, 50, and 20 neurons respectively.
      * 1 output layer.
      * 3 different types of activation functions, `relu`, `tanh`, and `sigmoid`.
  * Were you able to achieve the target model performance?
    * **YES** - we were able to reach the 75% threshold for the test data predictions, in fact the model achieved an 81% correct prediction rate. We also validated this model on the test data, on which the model was 79.5% accurate.
  * What steps did you take to try and increase model performance?
    * As expected we tried a number of methods to increase model performance:
      * We "binned" the `INCOME_AMT` column to split up organizations into two categories: either they do have a listed income, or they do not.
      * As detailed above we adapted the number of neurons, layers, and the types of activation functions. This step required a lot of experimentation, before we were satisfied we were nearing peak efficiency.
      * Finally, and most importantly - **we did not drop the NAME column**. The original analysis had us drop this identification feature, but companies that have a history of being successful tend to continue being successful. This meant that if a company was mentioned multiple times we would expect the next mention of them to follow suit, whether they were successful or not. To do this we "binned" this column too, into two categories: ***featured once*** and ***featured more than once***. It was this step that gave us the much more accurate model.

## Challenge - Summary

Overall we made good use of the Neural Network model, and the TensorFlow module. A lot of experimentation was required to get the model above the 75% threshold, which is an accurate feeling of data analysis in the real world. We were consistently testing out binning different columns, dropping columns, changing the number of layers and neurons, and really trying to narrow down what would give this model the needed edge.

It was only when we chose to not drop the `NAME` identification column in the first step of data transformation did we start seeing predictions over the threshold. From this advancement we continued working on the neural networks properties to provide as accurate a model as we could.

For an alternate method of prediction you could use a LogisticRegression model, or a RandomForestClassifier. To explore this we can enter the scaled data from the analysis into these models, and score them on accuracy. Each achieved higher than the threshold, and the LogisticRegression model ran much quicker than the neural network when training data. It would be worth exploring this further, and comparing the other metrics of the models, to see if this accuracy can be trusted, or if we should be looking at the Classification Report too.

## Context

This is the challenge repo for Module 19 of the University of Toronto School of Continuing Studies Data Analysis Bootcamp Course - **Neural Networks and Deep Learning Models** - Python and TensorFlow. Following the guidance of the module we end up pushing this selection of files to GitHub.
