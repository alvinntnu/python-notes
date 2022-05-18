#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Overview

# ## What is Machine Learning?

# Simply put, ML is the science of programming computers so that they can learn from data.
# 

# 
# ```{epigraph}
# 
# [Machine learning is the] field of study that gives computers the ability to learn without being explicitly programmed.
# 
# -- Arthur Samuel, 1959
# ```

# ```{epigraph}
# 
# A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with eexperience E.
# 
# -- Tom Mitchell, 1997
# 
# ```

# ## Why use machine learning?

# - Limitations of rules/heuristics-based systems:
#     - Rules are hard to be exhaustively listed.
#     - Tasks are simply too complex for rule generalization.
#     - The rule-based deductive approach is not helpful in discovering novel things.

# ## Types of Machine Learning

# We can categorize ML into four types according to the amount and type of supervision it gets during training:
# 
# - Supervised Learning
# - Unsupervised Learning
# - Semisupervised Learning
# - Reinforcement Learning

# ### Supervised Learning
# 
# - The data we feed to the ML algorithm includes the desired solutions, i.e., labels.
#     - Classification task (e.g., spam filter): the target is a categorical label.
#     - Regression task (e.g., car price prediction): the target is a numeric value.
# - Classification and regression are two sides of the coin.
#     - We can sometimes utilize regression algorithms for classification.
#     - A classic example is Logistic Regression.

# - Examples of Supervised Learning
#     - K-Nearest Neighbors
#     - Linear Regression
#     - Logistic Regression
#     - Support Vector Machines (SVMs)
#     - Decision Trees and Random Forests
# 

# ### Unsupervised Learning
# 
# - The data is unlabeled. We are more interested in the underlying grouping patterns of the data.
#     - Clustering
#     - Anomaly/Novelty detection
#     - Dimensionality reduction
#     - Association learning

# - Examples of Unsupervised Learning
#     - Clustering
#         - K-means
#         - Hierarchical Clustering
#     - Dimensionality reduction
#         - Principal Component Analysis

# ### Semisupervised Learning
# 
# - It's a combination of supervised and unsupervised learning.
# - For example, we start with unlabeled training data and use unsupervised learning to find groups. Users then label these groups with meaningful labels and then transform the task into a supervised learning.
# - A classific example is the photo-hosting service (e.g., face tagging).

# ### Reinforcement Learning
# 
# - This type of learning is often used in robots.
# - The learning system, called an Agent, will learn based on its observation of the environment. During the learning process, the agent will select and perform actions, and get rewards or penalties in return. Through this trial-and-error process, it will figure the most optimal strategy (i.e., policy) for the target task.
# - A classific example is DeepMind's AlphaGo.
