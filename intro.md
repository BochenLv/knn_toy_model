Generally speaking, using Hypernets to implement your own AutoML task consists of several key components, of which the most important ones are designing your own search space which has the form of a Hyperspace, the Hypermodel which is sampled from the search space using a searcher and the Estimator which recieves a Hypermodel, evaluates it and then returns the corresponding rewards such that the searcher can update the returned Hypermodel based on the rewards. We introduce the way of designing search space in Section \ref. Then we focus on the Hypermodel in Section \ref. Discussion about how an Estimator works is presented in Section \ref. Finally, we provide a toy example, designing an autoML task with k-Nearest Neighbour, to help the readers walk through the full pipeline of implementing Hypernets to your own tasks.

## kNN as a toy example
To apply your end-to-end AutoML model built with Hypernets on real dataset, the readers usually first define a search space, which mainly includes transformation of the data, feature engineering and the estimators, the most important part and the major work you did for designing your own AutoML model with Hypernets. With this search space in hand, the readers then choose a searcher defined in Hypernets, such as randomsearcher() and ..., whose function is to get a sample from the search space. The searcher is then passed to your Hypermodel. Finally, the search method of your Hypermodel is called to repeat searching in the search space, sampling a full-pipeline model from the search space, fiting the sampled model, evaluating its performance and then updating the searcher. 
### Designing a search space

### Constructing a toy_kNN as the Hypermodel and searching the search space 

### Evaluating the Hypermodel with Estimator

### Get your best model
