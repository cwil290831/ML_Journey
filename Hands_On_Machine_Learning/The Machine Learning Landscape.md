Chapter 1
# Training Supervision, 10 - 16
## Supervised Learning
The training set fed to the algorithm includes the desired solutions called *labels* or *target value*.
- Example Algorithms
	- classification, chp. 3
	- regression, chp. 4
- Use cases:
	- spam detection 
	- price of a car given features
- *Labeled Training Set* - the training data given represents the desired solution for the output of the model. 
## Unsupervised Learning
The training set is unlabeled and the system tries to learn without a teacher.
- Example Algorithms, chp. 8
	- clustering
	- visualization
	- dimensionality reduction
	- anomaly detection
	- association rule learning
- Use cases
	- detect similar groups of users
	- feature extraction
	- fraud prevention
	- novelty detection
## Semi-supervised Learning
Algorithms that can handle data that is only *partially labeled*.
- Majority are a combination of unsupervised to generate labels followed by supervised on the now fully labeled dataset. 
-  Use cases
	- Google Photos detects same person in several photos, user then labels that person, all future photos have the person labeled. 
## Self-supervised Learning
Algorithms that generate a *fully labeled* dataset from a *fully unlabeled* one. 
- Unsupervised to generate labels, then supervised on the fully labeled dataset. 
- Example Algorithms
	- Transfer Learning
	- deep neural networks, chp. 11
	- large language models (LLM), chp. 15
- Use Cases
	- image masking where the model is asked to recover the original image.
		- inputs: masked images
		- labels: original images used
	- pet classification
## Reinforcement Learning
The system must learn by itself what is the best strategy to get the most *rewards* over time. 
- The system observes an environment then selects and performs actions in order to get positive or negative rewards. 
- These strategy, better known as a *policy*, defines what action(s) the system should chose given a situation. 
- Example Algorithms
	- Agents
- Use cases:
	- Robots
# Batch vs. Online Learning, 17 - 21
Whether a system can learn incrementally from a stream of incoming data or it requires the full dataset. 
## Batch Learning
The system must be training using all the available data. 
- *Offline Learning* - after a system is trained, it is launched into production and operates via applying what it has learned. It **does not** continue to learn after it has been launched. 
- Model performance decays over time, the rate of decay varies based on the *data drift* (model rot) that occurs due to the world evolving while the model remains stagnant. 
- *Batch Learning* can be quite costly because in order to combat *data drift* the model needs to be retrained on new data constantly. This retraining can be automated but because *batch learning* requires a full dataset to work, a lot of computing resources, time and money are used every time it needs to be retrained. 
	- This makes it a poor choice for domains where the data changes rapidly, such as the finance sector. 
- Example
	- Random Forest
## Online Learning
The system is trained incrementally by feeding it *mini-batches* (batches) of data instances sequentially. This allows the system to learn about new data on the fly.
- *Out-of-Core Learning* - models that require training on huge datasets that cannot fit in one machine's memory can be trained using *online learning* because the model is fed new data and trained on this data incrementally.
	- **Note:** OOC learning is often done offline. 
- *Learning Rate* - how fast a system should adapt to changing data. 
	- *Catastrophic Interference* - when the learning rate of the system is too high and the system readily **forgets** the old data. 
	- *Inertia* - when the system's learning rate is slower it will be less sensitive to noise in the new data or to outliers.
- *Online Learning* is sensitive to bad data which can cause the performance of the system to decline. 
	- The rate of decline is dependent on data quality and learning rate. 
	- Notably, if your system is live your clients will likely notice this decline. 
	- **Risk Reduction** comes from close system monitoring and either switching off the learning or reverting back to a previous uncorroded model. 
		- anomaly detection could be very helpful here. 

# Instance-Based vs. Model-Based Learning, 21 - 27
How well does a system *generalize* when given new instances of data. A system needs to be able to make good predictions based on the training examples. 
## Instance-Based Learning
The system learns the training examples by heart, then *generalizes* to new cases by using a similarity measure to compare them to the learned examples (or a subset of them).
- *Instance-Based* is often good for small datasets that have a high change rate.
	- However, it doesn't scale very well because it often requires a version of [[#Batch Learning]]. Subsequently, it doesn't work well with high-dimensional data such as images.
- *Measure of Similarity* - how similar one point of data is to another. 
## Model-Based Learning
A model is selected based on the data and trained with the goal of making quality *inferences* on new data. 
### Model Selection
A critical step in *model-based learning* is choosing which model to use and fully specifying its architecture. #toLearn (what is model architecture)
### Model Training 
The model is run with the goal of *tuning the parameters*  so that they best fit the training data with the hope that these *parameters* help the model *generalize* well on new data. 
- *Model Parameters* - values that are learned by the model via training. 
- *Model Hyperparameter* - a parameter, set before training, used to constrain the learning algorithm that produces the production model. 
	- *Regularization* - makes a model simpler and reduces the risk of *overfitting* when training a model. 
- *Performance Measure* - the tool used to determine how good or bad your model performs. A performance measure is selected based on the model. 
	- *Utility Function* - defines how good your model is. Also know as a "fitness function".
	- *Cost Function* - defines how bad your model is. Also know as the *loss function*.
- *Train set* - a subset of the data that is used to train one or more models. Note, this is a unique subset to both *test* and *validation*.
- *Test Set* - a subset of the data that is given to the chosen model as 'unseen data' after it has been trained to evaluate a model's performance. Note, this is a unique subset to both *validation* and *train*.
- *Validation Set* - a subset of the data used to evaluate selected trained models to select the 'best'. Note, this is a unique subset to both *test* and *train*.
- *Train-Validation Set* - the combination of the *train set* and the *validation set* used to train the 'best' model. 
**Note:** Example code for a linear model on [25:26]
### Model Inferences
The *inferences*, predictions, a model makes are made on new data based on what it learned while it was being trained. 

# Challenges of Machine Learning, 27 - 34
1. Insufficient Data Quantity
2. Non-representative Training Data/ Data Mismatch
3. Poor-Quality Data
4. Irrelevant Features
5. Overfitting the Training Data
6. Underfitting the Training Data
7. Deployment Issues

# Testing & Validating, 35 - 38
1. Hyperparameter Tuning
2. Model Selection
3. Holdout Validation
4. Cross-Validation
5. Train-Test-Validate Split
