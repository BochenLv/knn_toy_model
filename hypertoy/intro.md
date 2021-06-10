Generally speaking, using Hypernets to implement your own AutoML task consists of several key components, of which the most important ones are designing your own search space which has the form of a Hyperspace, the Hypermodel which is sampled from the search space using a searcher and the Estimator which recieves a Hypermodel, evaluates it and then returns the corresponding rewards such that the searcher can update the returned Hypermodel based on the rewards. We introduce the way of designing search space in Section \ref. Then we focus on the Hypermodel in Section \ref. Discussion about how an Estimator works is presented in Section \ref. Finally, we provide a toy example, designing an autoML task with k-Nearest Neighbour, to help the readers walk through the full pipeline of implementing Hypernets to your own tasks.

## Easy deploying of your AutoML task
To apply your end-to-end AutoML models built with the Hypernets, the readers usually first design a search space, which mainly includes transformations of the data, feature engineerings, and the desired estimators, the most important part and the major work you did for designing your AutoML model with the Hypernets. With this search space in hand, the readers then choose a searcher from those defined in the Hypernets, such as RandomSearcher, whose major function is to repeatedly 'search' samples from the search space. This searcher is then passed as an argument to your model, a Hypermodel object. Finally, the search method of the Hypermodel is called to repeat the following procedures: searching in the search space, sampling a full-pipeline model from the search space, fitting the sampled model, evaluating its performance, and then updating the searcher until the end. The above process is summarized as follows with 4 lines of codes after loading the data:
```python
#Load the data and suppose that the task is multi-classification
from sklearn.model_selection import train_test_split
X, y = load_your_data()
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.1)

#Design a search space
search_space = get_your_search_space

#Choose a searcher from the Hypernets.searchers
searcher = Your_searcher(search_space, other_arguments)

#Pass the searcher as an argument to your model, a Hypermodel object
model = Your_Hypermodel(searcher, task='multiclass', other_arguments)

#Call the 'search' method
model.search(X_train, y_train, X_eval=X_test, y_eval=y_test)
```

### Designing a search space

### Constructing a toy_kNN as the Hypermodel and searching the search space 

### Evaluating the Hypermodel with Estimator

### Get your best model
try save