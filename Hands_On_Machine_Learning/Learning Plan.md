# Weekly Study Structure (20 Hours)
Example schedule:

|Day|Activity|
|---|---|
|Monday|reading + notes (3h)|
|Tuesday|coding notebooks (4h)|
|Wednesday|coding notebooks (4h)|
|Thursday|experiments + deeper learning (4h)|
|Friday|project work (3h)|
|Weekend|optional review / catch-up (2h)|

# Portfolio Goals
1. housing price predictor
    
2. spam classifier
    
3. customer churn model
    
4. MNIST neural network
    
5. CNN image classifier
    
6. sentiment analysis model

7. recommendation system

# The Fundamentals of Machine Learning
## Week 1 —  ML Foundations,  Workflow & Training Models
### Chapters 1 & 2:
- [x] [[The Machine Learning Landscape]]
- [ ] End-to-End Machine Learning Project

**Concepts:**
- supervised vs unsupervised learning
- training vs testing
- ML pipelines
- feature engineering
- data cleaning
- feature engineering
- pipelines with  scikit-learn
#### Practice:
Run the full housing price project in the book.
#### Mini project:  
Rebuild the housing project using a **different dataset** (Kaggle housing or Airbnb prices).

### Chapter 4: Training Models
**Concepts:**
- linear regression
- gradient descent  
- regularization 
- polynomial regression
#### Practice:
*Implement:*
- batch gradient descent  
- stochastic gradient descent    
- Experiment with learning rates.
#### Mini project:  
Build a **bike rental demand predictor**.

## Week 2 — Classification
### Chapter 3:  Classification
**Concepts:**
- logistic regression
- confusion matrix
- precision / recall
- ROC curves
#### Practice:
Train classifiers on the digit dataset.
**Compare:**
- logistic regression
- SGD classifier
#### Mini project:  
- Build a **spam detection classifier**.

## Week 3 — Decision Trees, Ensembles & Dimensionality Reduction
### Chapters:
- Decision Trees
- Ensemble Learning & Random Forests
- Dimensionality Reduction
#### Concepts:
- random forests
- boosting
- PCA
#### Practice:
**Compare performance of:**
- decision trees
- random forests 
- gradient boosting
- Visualize PCA projections.
#### Mini project: 
- Customer churn prediction model.

## Bonus — Unsupervised Learning Techniques

# Neural Networks and Deep Learning
## Week 4 — Neural Networks
### Chapter  9: Introduction to Artificial Neural Networks
#### Concepts:
- perceptrons
- activation functions
- back-propagation
#### Practice:
Train a neural network using PyTorch.
- Dataset:  MNIST digit recognition.
#### Mini project:  
Build a **handwritten digit classifier**.

## Week 5 — Deep Neural Networks
### Chapters 10, 11:
- Building Neural Networks with PyTorch
- Training Deep Neural Networks
#### Concepts:
- weight initialization
- batch normalization
- dropout
- training loops
#### Practice:
Write your own PyTorch training loop:
- for epoch in range(epochs):  
    optimizer.zero_grad()  
    loss.backward()  
    optimizer.step()
#### Mini project:  
Build a **multi-layer deep neural network**.

## Week 6 — CNNs, RNNs & Image Models
### Chapter 12: Deep Computer Vision using Convolutional Neural Networks
 #### Concepts:
- convolution layers
- pooling
- feature maps
#### Practice:
Train a CNN on CIFAR-10.
#### Mini project:  
Image classifier for:
- animals
- clothing
- or medical images

### Chapter 13: Processing Sequences using RNNs and CNNs
#### Concepts
- recurrent layers
- ARMA
- Forecasting
- Sequences
- Memory Cells

## Week 7 — NLP & Transformers
### Chapters 14, 15, 16, 17:
- Natural Language Processing with RNNs and Attention
- Transformers for Natural Language Processing and Chatbots
#### Concepts:
- embeddings
- transformers
#### Practice:
Build:
- sentiment analysis model

## Week 8 — Unsupervised NN, GANs & Reinforcement Learning
### Chapters 18, 19:
- Autoencoders, GANs, and Diffusion Models
- Reinforcement Learning
#### Concepts:
- GANs
- reinforcement learning
- agents and rewards

