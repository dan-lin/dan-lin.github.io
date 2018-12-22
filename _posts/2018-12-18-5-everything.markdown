---
title: "A Comparison of XGBoost, LightGBM and CatBoost"
layout: post
date: 2018-12-18 20:00
tag: gradient-boosting
image: https://koppl.in/indigo/assets/images/jekyll-logo-light-solid.png
headerImage: false
projects: true
hidden: true # don't count this post in blog pagination
description: "Longer description"
category: project
author: danlin
externalLink: false
---

Gradient Boosting Machines (GBMs) are a family of powerful Machine Learning models. This article investigates three of the most popular GBM libraries: XGBoost, LightGBM and CatBoost. We begin with an overview of how GBMs work before diving into the advantages, disadvantages and special features of each model. Finally, we conduct two experiments followed by their results and conclusions.



## 1.  Gradient Boosting Machines

GBMs are a popular technique both in industry and in current machine learning research. To understand how they work, let's divide the discussion into two parts as derived from their name: "boosting" followed by "gradient". 

The central idea behind **boosting** is that many simple models (known as "weak learners") can be combined together to form a single powerful predictor. Boosting adds weak learners together sequentially to the existing model which begins with a single weak learner. Each new addition is selected to improve the performance of the overall model in a greedy fashion and they do not alter the predictions of previous weak learners. Typically, weak learners can be as simple as decision trees. 

GBMs utilise boosting to combine these weak learners and seek to minimise the residuals in the data (the difference between the predictions of the model and the true response values). After a new weak learner is added to the model, the difference between the true response $$y$$ and new prediction $$\hat{y}$$ becomes the new target of the next weak learner, i.e. new weak learners are added to predict the new residuals iteratively. This is where the **"gradient"** aspect of GBMs comes in: minimising the loss function in a GBM is congruent to minimising some function using gradient descent. We add weak learners in a GBM to reduce the loss step-by-step, by moving in the direction of the residuals.

GBMs often provide very strong predictive accuracy - they are one of the most popular models in data science competitions such as Kaggle. They are flexible and versatile, e.g. by being able to optimise a variety of loss functions and in handling missing data. However, GBMs are highly prone to overfitting and they can be computationally expensive.

### Further reading

There are an abundance of resources available on the topic of GBMs. T. Parr and J. Howard provide a fantastic and intuitive explanation of gradient boosting[^1].  An article by P. Grover gives an introduction to gradient boosting[^2], whilst Machine Learning Mastery takes a more practical approach[^3]. Naturally, J.H. Friedman's original paper[^4] on GBMs is a must read to gain a full understanding of these models.



## 2.  XGBoost, LightGBM and Catboost

![Screenshot](https://raw.githubusercontent.com/dan-lin/dan-lin.github.io/init-branch/assets/images/experiments/gbms_comparison_01.png)



## 3.  Experiments

There are many experiments that compare speed and performance benchmarks of GBMs against each other, with widely varying results. 

### Datasets

The performances of XGBoost, LightGBM and CatBoost are compared on two different datasets:

1. **Olympic Athletes**: Contains a record of athletes in the Olympics, including a selection of their biological and team information, the Olympic year, sport, and what type of medal they won (if appropriate). The aim is to predict whether an athlete has won a medal. Special characteristics:

   - Binary response variable
   - Class imbalance

2. **US Flight Delays**: Contains a record of flights to and from cities in the US, including the flight dates and times, journey details and any delays. The aim is to predict the delay upon arrival. This dataset does not have any special characteristics, but has a nice balance of categorical and non-categorical features.


### Methodology

Each dataset is cleaned before modelling occurs, with processes such as removing features that aren't useful for prediction and removing null values in the response variable. 

For XGBoost, categorical features must be encoded into a different form as it cannot handle categorical features directly. One-hot encoding is used if there is a small number of categories in the feature, however if there are many distinct categories then binary encoding is used. This saves on computational space and time without sacrificing performance (as noted in this study ^X).

After cleaning the dataset, the following steps are used to train and fit the GBM models, as recommended in this article ^X:

1. Set no. of weak learners
2. Tune other hyperparameters
3. Train final no. of trees and learning rate
4. Fit final model with these optimised hyperparameters

Scikit-learn's cross validation and grid search functions are used to standardise across all models. Grid search is used instead of random search/Bayesian optimization across the hyperparameter space, though the others are more than valid approaches as well. 

Each GBM model was trained using 32 CPU cores. Evaluation is conducted on a hold old test set, with a size ratio of 80:20 between training and test sets for each dataset.



## 4.  Results

![Screenshot](https://raw.githubusercontent.com/dan-lin/dan-lin.github.io/init-branch/assets/images/experiments/gbms_comparison_02.png)

![Screenshot](https://raw.githubusercontent.com/dan-lin/dan-lin.github.io/init-branch/assets/images/experiments/gbms_comparison_03.png)



## 5.  Conclusion

TODO:

- Write comparison table and add to md. 1
  - Research new bullet pt
  - Check existing bullet pts for a) accuracy and b) descriptive details
- Copy paste results tables into md. 0.5
- Write conclusions to each notebook (x2) 0.5
- Write conclusions to md. 0.5
- Recheck references in md. 0.25
- Proof read md. 0.5
- Merge notebooks and utils file to Github 0.25
- Create 'data' folder in directory 0.25
- Write README for gradient_boosting_machines directory 0.25
- Proof read all notebooks, utils, README and md. 0.5
- Attempt to publish on local 0.25


### References

[^1]: T. Parr, J. Howard. [*How to explain gradient boosting*](https://explained.ai/gradient-boosting/index.html).  
[^2]: P.Grover. *[Gradient Boosting from scratch](https://medium.com/mlreview/gradient-boosting-from-scratch-1e317ae4587d)*
[^3]: Machine Learning Mastery. *[Gentle Introduction to Gradient Boosting Algorithms](https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/)*
[^4]: J.H. Friedman. *[Greedy Function Approximation: A Gradient Boosting Machine](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf)*
[^5]: A. Swalin. *[CatBoost vs. Light GBM vs. XGBoost](https://towardsdatascience.com/catboost-vs-light-gbm-vs-xgboost-5f93620723db)*

