# A Brief Tutorial for Implementing AutoML Tools with Hypernets

Parameter tuning is an inevitable step for successfully implementing a machine learning model. Even for a simple model as K-nearest neighbors(KNN) for the classification task, we need to at least determine the number of the neighbors and the distance metric to be used to predict the label of a given example. Let alone models which have much more tunable parameters and have to be trained multiple times before we can pick suitable values for their parameters. Furthermore, tuning parameters in a brute force approach is inefficient while using an advanced search method takes intensive efforts. Can we focus more on parts of machine learning like designing novel models while only perform procedures like parameter tuning in a simple and happy way? 

The answer is positive. 

```Hypernets```, a unified Automated Machine learning(AutoML) framework, offers us a very simple way to solve such problems. Taking the parameter tuning problem of the KNN model as an example, using a ```search_param``` function from the ```Hypernets```, the only required work for us is to define a function serving as the measure of the quality of a set of given parameters.
```python
from sklearn import neighbor

def score_function(X_train, y_train, X_evl, y_evl, 
                   n_neighbors=Choice([3, 5, 6, 10, 20]),
                   weights=Choice(['uniform', 'distance']),
                   algorithm=Choice(['auto', 'ball_tree', 'kd_tree', 'brute']),
                   leaf_size=30,
                   p=Choice([1, 2])):
    # The avaliable values for each tunable parameter are those provided by the list
    # elements of the argument of Choice(). For example, the parameter "n_neighbors",
    # the number of the nearest neighbors used to predict the label of a given example, 
    # can be chosen as 3, 5, 6, 10, and 20. 
    model = neighbors.KNeighborsClassifier(n_neighbors, weights, algorithm, leaf_size, p)
    model.fit(X_train, y_train)
    scores = model.score(X_evl, y_evl) #This score is taken as mean accuracy of the model on (X_evl, y_evl)
    return scores
```
Currently, there is no need to know how the function ```search_param``` is able to perform parameter tuning by utilizing the above score function--we only manually provide possible values we like to ```Choice()``` for each tunable parameter. Now let's use the ```search_param``` function with the gird search algorithm, or other search algorithms such as random search or Monte-Carlo Tree search, to find the suitable parameter values for our KNN model by simply passing the ```score_function``` defined above as an argument to it:
```python
import hypernets.utils.param_tuning as pt
history = pt.search_params(score_function, 'grid', max_trials=10, optimize_direction='max')
```
The best model parameters can be obtained by calling the following method of ```history```
```python
best_param = history.get_best().sample
```

This is not the whole story. 

Parameter tuning is only a fraction of the full-pipeline AutoML process and ```Hypernets``` is capable of doing far more things than just tuning parameters. In the following sections, we will briefly introduce ```Hypernets``` as an AutoML framework and wish to clarify: 
- the basic building blocks of ```Hypernets```;
- basic procedures to develop an AutoML tool for parameter tuning problem and the more general full-pipeline machine learning modeling;
- some advanced features of ```Hypernets```.

*******

```Hypernets``` is an AutoML framework that allows the users to easily develop various kinds of AutoML and Automated Deep Learning(AutoDL) tools without reinventing some necessary components which are often common to such tools. Before ```Hypernets```, there already existed many AutoML tools. However, these tools are usually designed for some specific purposes thus not convenient to be generalized to other ones. As a result, the AutoML community may have to take a lot of efforts to repeatedly develop some common parts before deploying their AutoML models due to the lack of an underlying AutoML framework. 

```Hypernets``` can save such efforts to a large extent while offering more possibilities. 
- First, it decouples the basic components of a general AutoML procedure (Fig. \ref) as four distinct parts: the ```HyperSpace```, the ```Searcher```, the ```HyperModel```, and the ```Estimation Strategy``` (Fig. \ref). This idea is motivated by allowing users to manipulate different components of an AutoML tool built with ```Hypernets``` accordingly for different purposes. 
- Second, the ```HyperSpace``` is designed to be a powerful search space. The ```HyperSpace``` consists of three different kinds of space: the **module space**, the **parameter space** and the **connection space**, where the module space is designed to contain various machine learning models, data preprocessing or feature engineerings, the parameter space provides the parameters to be searched for machine learning models and the connection space determines the way how different module spaces connect. These connected module spaces and parameter spaces finally give us a highly comprehensive search space which is able to describe the full-pipeline machine learning modeling ranging from data preprocessing to model ensemble. 
- Third, ```Hypernets``` provides many search algorithms including simple methods, such as Random Search and Grid Search, and advanced ones such as Monte-Carlo Tree Search. Users can not only simply choose one from these efficient search methods but also similarly design new search algorithms. 
- Finally, ```Hypernets``` also supports many advanced techniques to further improve performances of the trained machine learning models. For example, users can apply early stopping to accelerate the training process and prevent overfitting; data cleaning can be applied to improve data quality; data drift detection can be enabled to improve the generalization ability of the model, etc. 

Based on the above brief introduction, using the ```Hypernets``` to implement an AutoML task can now be decomposed as three parts: designing the **search space**, an instance of the ```Hyperspace```, constructing the **Hypermodel** which will be sampled from the search space using a searcher provided by ```Hypernets``` during the search process, and building the **Estimator** which receives a sampled Hypermodel, evaluates it and then returns the corresponding rewards such that the searcher can update the Hypermodel to be sampled based on the rewards. 

We will provide a toy example, designing an AutoML tool for KNN, to help the readers walk through the full pipeline of implementing the ```Hypernets``` to an AutoML task. 

To reveal the core features and ideas of ```Hypernets```, we first continue to solve the problem defined in the very beginning--how to perform parameter tuning of KNN automatically using ```Hyernets```--but in a different manner: we view the parameter tuning problem as a complete AutoML task and develop a complete AutoML tool for this task from scratch using ```Hypernets```. For simplicity, we only consider the classification task, and the regression case can be easily generalized. As introduced above, this developing procedure contains 3 steps and we will simply follow these steps. 

- ***Designing the search space.*** In the case of parameter tuning, our search space of the AutoML task, a HyperSpace, is very simple in the sense that there is only one module space which contains only one machine learning model--our KNN model--along with its parameter space. To incorporate these spaces, we first define the ParameterSpace for tunable parameters with different values and then build the whole HyperSpace to include this ParameterSpace so that the search algorithm can search suitable parameters among available ones.  
    ```python
    class Param_space(object):

        def __init__(self, **kwargs):
            super(Param_space, self).__init__()

        @property
        # The following function returns a dictionary containing tunable parameters, where 
        # the avaliable values for each parameter are those provided by the arguments of 
        # Choice(), a class which in fact inherits from the ParameterSpace, one of the three 
        # basic kinds of the HyperSpace. In other words, all values of the returned 
        # dictionary are parts of the parameter space if they are Choice(). 
        def knn(self):
            # cls: the name of the machine learning model
            # other parameters are all parameters of KNN, which is imported from sklearn
            return dict(
                cls=neighbors.KNeighborsClassifier,
                n_neighbors=Choice([2, 3, 5, 6]),
                weights=Choice(['uniform', 'distance']),
                algorithm=Choice(['auto', 'ball_tree', 'kd_tree', 'brute']),
                leaf_size=Choice([20, 30, 40]),
                p=Choice([1, 2]),
                metric='minkowski',
                metric_params=None, 
                n_jobs=None,
            )

        def __call__(self, *args, **kwargs):
            space = HyperSpace()

            with space.as_default():
                hyper_input = HyperInput(name='input1')
                model = self.knn #prepare the KNN model to be inclued into the module space 
                modules = [ModuleSpace(name=f'{model["cls"].__name__}', **model)] #To make a module space containing the KNN model for the HyperSpace using ModuleSpace()
                outputs = ModuleChoice(modules)(hyper_input) #pick a model from the module space if there are mutiple ones as the output of the search space. Here we only have a KNN model.
                space.set_inputs(hyper_input)

            return space
    ```
- ***Constructing the Hypermodel.*** The HyperMdel does not require many modifications for our specific task since many core functionalities of the HyperMdel have already been well defined in ```Hypernets``` and are common across different machine learning models and tasks. We only pay attention to two functions, the ```_get_estimator```, which returns the corresponding KNN model of the sampled search space, and the ```load_estimator```, which loads the configurations of the saved model. The most important method for a HyperModel is the "search" method. By calling the ```search``` method, the search algorithm searches in the search space and returns a sample of the search space to be utilized for the HyperModel. This HyperModel is then evaluated based on the chosen reward metric and updated towards the optimizing direction.    
    ```python
    class KnnModel(HyperModel):
        def __init__(self, searcher, reward_metric=None, task=None):
            
            super(KnnModel, self).__init__(searcher, reward_metric=reward_metric, task=task)
        
        def _get_estimator(self, space_sample):
            return KnnEstimator(space_sample, task=self.task)
        
        def load_estimator(self, model_file):
            return KnnEstimator.load(model_file)
    ```
- ***Building the Estimator.*** Building the Estimator often takes the most effort for developing a new AutoML tool using ```Hypernets```. The ```Estimators``` required by ```Hypernets``` is in fact a more general notion than the frequently used one in ```sklearn```--the machine learning model. Fortunately, for our case of parameter tuning of KNN, the ```Estimator``` is easy to be implemented since the sampled search space only contains one machine learning model which is the only thing that needs to be evaluated by the ```Estimator```. Moreover, we emphasize that the actual abilities of the ```Estimator``` are not restricted to that defined in this section and we refer the readers to the [next section](#sec_eg) for further details. 
    ```python
    class KnnEstimator(Estimator):
        def __init__(self, space_sample, task='binary'):
            # Users can also set the task as None since Hypernets can automatically
            # infer the task type.
            super(KnnEstimator, self).__init__(space_sample, task)

            out = space_sample.get_outputs()[0]# Returns the KNN model
            kwargs = out.param_values
            kwargs = {key: value for key, value in kwargs.items() if not isinstance(value, HyperNode)} # Copy the parameters which will be sent to the KNN model

            cls = kwargs.pop('cls')
            self.model = cls(**kwargs)
            self.cls = cls
            self.model_args = kwargs
        
        def fit(self, X, y, **kwargs):
            # Fit the training data and return the trained model
            self.model.fit(X, y, **kwargs)

            return self
        
        def predict(self, X, **kwargs):
            # Return the label of the given example
            pred = self.model.predict(X, **kwargs)

            return pred

        def evaluate(self, X, y, **kwargs):
            # Evaluate the KNN model on the given dataset (X, y). Here we choose the 
            # mean accuracy of the KNN model on (X, y) as the evaluation score.
            scores = self.model.score(X, y)

            return scores
        
        def save(self, model_file):
            with fs.open(model_file, 'wb') as f:
                pickle.dump(self, f, protocol=4)

        @staticmethod
        def load(model_file):
            with fs.open(model_file, 'rb') as f:
                return pickle.load(f)

        def get_iteration_scores():
            # This function is designed to return the iteration score for each iteration. 
            # It is not mandatory for us to implement this method at first.
            return []
    ```

With the above AutoML tool, we are now ready to perform a complete automatic parameter tuning for KNN. In general, we only need four lines of codes to complete such implementation after we finish designing the required AutoML tools--not for the specific example presented here but a more general routine. This routine is summarized as follows:
1. Define the search space.
    ```python
    search_space = Param_space()
    ```
2. Choose a searcher from those search algorithms provided by ```Hypernets```. One required  argument for the searcher is the search sapce in which the searcher will perform searching.
    ```python
    searcher = GridSearcher(search_space, optimize_direction=optimize_direction)
    ```
3. Construct the HyperMdel which receives the searcher as its required arguments. In our example, the HyperModel is the ```KnnModel```. 
    ```python
    model = KnnModel(searcher=searcher, task='multiclass', reward_metric='accuracy')
    ```
4. The ```search``` method of our HyperModel is called to automatically perform the search process on the dataset (X_train, y_train) and record the current best model parameters. 
    ```python
    model.search(X_train, y_train, X_eval, y_eval, **kwargs)
    ```
5. One can get the best model in the following way:
    ```python
    best_model = model.get_best_trial()
    ```
Now we can celebrate the fine-tuned KNN model! 

The convenience of following this procedure lies in that one needs not to develop anything else to perform parameter tuning of the KNN model for other classification task datasets without categorical features. Instead, simply passing these datasets to the ```search``` method of the ```KnnModel``` will return us the model with suitable parameters.

However, readers will also immediately notice that, before sending the dataset to the model, one has to manually handle the categorical features of some dataset if there exist such things because the KNN model can not treat with categorical features properly. Some users may also want our AutoML tool to be able to perform more things like data cleaning. It is therefore a great idea to extend our AutoML tool for the KNN model to automate the full pipeline of machine learning task once for all. These are exactly the topics of the [next section](#sec_eg).

## Building your full-pipeline AutoML tool for KNN<span id=sec_eg>
Typically, the procedures of a full-pipeline machine learning modeling range from data preprocessing to model ensemble. For the purpse of enabling our AutoML tool to automate such full-pipeline modeling, we need to design a more comprehensive search space, which should at least include transformations of the data, feature engineerings, and the machine learning models along with their tunable parameters. Such AutoML tool will largely relieve us from the headaches of dealing with data and feature issues of datasets. 

The most important part and the primary work we will do is to extend our search space based on the introduction of the basic building blocks of ```Hypernets``` in the last section. For clarity, we still follow the 3 steps of developing our AutoML tools for full-pipeline KNN model with ```Hypernets``` as indicated before.


- ***Designing a search space.*** To enable our AutoML tool to perform things like data    preprocessing, we need to encapsulate these procedures to module spaces to our search space, a ```HyperSpace``` object, and then connect them using the ```connection space``` as introduced above. For this reason, these module spaces are now divided into two kinds: one containing the **preprocessor** and the other for **machine learning model**, i.e. KNN model here. We now devote to wrapping these two kinds of module spaces into our search space respectively for full-pipeline AutoML process. 

    Preprocessors in a search space are connected through ```pipeline```. Since both of them are not closely related to any specific models, fortunately, we can directly borrow them from the ```HyperGBM``` package where they are already well defined and need not be modified much. The module spaces for preprocessors are created and connected by calling the function ```create_preprocessor``` and should be implemented before machine learning models.

    On the other hand, building the module space for our KNN model needs extra effort. We do this by introducing a class ```_HypreEstimatorCreator``` so that one can easily generalize the method presented here to incude other kinds of machine learning models. Then calling the function ```create_estimators```will return the module space of our KNN model. 
    
    We can now define a class ```KnnSearchSpaceGenerator``` as in the last section to obtain the search space which include the ```create_preprocessor``` and ```create_estimators``` as its methods. Moreover, it is fairly easy to manipulate the initializations of the models or even include other machine learning models provided by scikit-learn such as support vector machines into our search space. 
    ```python
    class KnnSearchSpaceGenerator(object):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            
            self.options = kwargs

        @property
        def default_knn_init_kwargs(self):
            return {'n_neighbors': Choice([1, 3, 5]),
                    'weights': Choice(['uniform', 'distance']),
                    'algorithm': Choice(['auto', 'ball_tree', 'kd_tree', 'brute']),
                    'leaf_size': Choice([10, 20 ,30]),
                    'p': Choice([1, 2]),
                    'metric': 'minkowski',
                    'metric_params': None,
                    'n_jobs': None}
        
        @property
        def default_knn_fit_kwargs(self):
            return {}

        @property
        def estimators(self):
            r = {}
            r['knn'] = (kNNEstimator, self.default_knn_init_kwargs, self.default_knn_fit_kwargs)
            return r
        
        def create_preprocessor(self, hyper_input, options):
                    ...
            return preprocessor

        def create_estimators(self, hyper_input, options):
            assert len(self.estimators.keys()) > 0

            creators = [_HyperEstimatorCreator(pairs[0],
                                            init_kwargs=_merge_dict(pairs[1], options.pop(f'{k}_init_kwargs', None)),
                                            fit_kwargs=_merge_dict(pairs[2], options.pop(f'{k}_fit_kwargs', None)))
                        for k, pairs in self.estimators.items()]

            estimators = [c() for c in creators]
            return ModuleChoice(estimators, name='estimator_options')(hyper_input)

        def __call__(self, *args, **kwargs):
            options = _merge_dict(self.options, kwargs)

            space = HyperSpace()
            with space.as_default():
                hyper_input = HyperInput(name='input1')
                self.create_estimators(self.create_preprocessor(hyper_input, options), options)
                space.set_inputs(hyper_input)
            return space

    ```

- ***Constructing the Hypermodel.*** to receive the searcher<span id=sec_model> 
This section needs additional attention for its importance. Here, we devote to constructing the ```HyperYourModel``` of your task, which is an object inherited from the ```Hypermodel``` of the ```Hypernets```. For our example of k-nearest neighbors, this is simply named as ```toy_KNN```. It is not hard for the readers to build ```HyperYourModel``` with models other than the k-nearest neighbors by following steps discussed in this section. 

Basically, to define a class ```HyperYourModel```, one needs to define two functions properly: 
1. A function that returns the estimator of the HyperYourModel from the search space returned by the searcher
    ```python 
    def _get_estimator(space_sample):
        #space_sample, a Hyperspace, is returned by a searcher
        estimator = HyperYourModelEstimator(some_args)
        return estimator 
    ```
    This function overwrites the ```_get_estimator``` method of the ```Hypermodel```, from which the ```HyperYourModel``` is inherited. One may immediately notice that the returned estimators are actually ```HyperYourModelEstimator```, not the typical "estimator" which usually refers to some m 
    
    The uniqueness of each ```HyperYourModel```, e.g. the Hypermodels with k-nearest neighbors or support vector machine, is provided by the class ```HyperYourModelEstimator``` through receiving different search space returned by the searcher. We discuss this uniqueness now and name our special HyperYourModelEstimator as ```toy_KNN_estimator``` since we are taking the k-nearest neighbors as our example. Although a ```HyperYourModelEstimator``` usually includes many arguments and functions to support some advanced features of ```Hypernets```, fortunately, there is nearly nothing that needs to be rewritten from scratch for the reason that many of the arguments and functions have already been provided in the ```HyperGBMEstimator``` of the ```HyperGMB``` package. The more deep reason for this convenience lies in the fact that the ```HyperYourModelEstimator``` should be a more general "estimator" which not only includes the typical estimators, i.e. machine learning models, but also the whole pipeline from the data transformations to the machine learning models, where the steps before introducing these machine learning models to the pipeline are common in different cases. The uniqueness is then due to the new estimators, i.e. machine learning models, that you include into your search space when it is designed and was assumed to be magically provided there. 
    
    We now turn to the implementation details of defining new estimators to explain the reason for such uniqueness. 

2. A function which loads and returns the HyperEstimator for the desired model, for example
    ```python
    def load_estimator(self, model_file):
        #load the details of the model from the model_file
        assert model_file is not None
        return HyperYourModelEstimator.load(model_file)
    ```
    abc
    cde
    fg
- ***Building the estimator.***

Finally, the ```search``` method of the Hypermodel is called to repeat the following procedures: the searcher searches in the search space and samples a full-pipeline model from the search space, the estimator fits the sampled model of the search space, evaluates its performance, and then updating the searcher to get a new sample of the search space until the end. The above process is summarized as follows with 4 lines of codes after loading the data:
```python
#Load the data and suppose that the task is multi-classification
from sklearn.model_selection import train_test_split
X, y = load_your_data()
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.1)

#Design a search space
search_space = get_your_search_space

#Choose a searcher from the Hypernets searchers
searcher = Your_searcher(search_space, other_arguments)

#Pass the searcher as an argument to your model, a Hypermodel object
model = Your_Hypermodel(searcher, task='multiclass', other_arguments)

#Call the 'search' method
model.search(X_train, y_train, X_eval=X_test, y_eval=y_test)
```


### Choosing a searcher
Since many efficient searchers have already been provided in the ```Hypernets```,  it is fairly easy for the readers to simply choose one of them and send the search space you just defined to this searcher. For example,
```python
searcher = RandomSearcher(search_space)
```
One can also take more efforts to design new kinds of searcher by refering to [Searcher](#sec_searcher).
"YourModelSearchSpaceGenerator" and summarized as follows:
    ```python
    class YourModelSearchSpaceGenerator(SearchSpaceGenerator):
        """
        enable_your_model1: bool, set this as True to include model1 in the search space.
        enable_your_model2: bool, set this as True to include model2 in the search space.
        The readers can also add more models
        """
        def __init__(self, enable_your_model1=True, enable_your_model2=True, **kwargs):
            super(YourModelSearchSpaceGenerator, self).__init__(**kwargs)

        #the default initialized parameters for model1, Choice will iterate overe these parameters.
        @property
        def your_model1_init_kwargs(self):
            return {'your_model1_param1': Choice([1, 2, 3]),
                    'your_model1_param2': Choice(['good', 'great']),
                    'your_model1_param3': None}
        
        #the default initialized parameters for model2
        @property
        def your_model2_init_kwargs(self):
            return {'your_model2_param1': Choice([4, 5, 6]),
                    'your_model2_param2': Choice(['trivial', 'nontrivial'])}

        @property
        def your_model1_fit_kwargs(self):
            return {}

        @property
        def your_model2_fit_kwargs(self):
            return {}

        #return the defined estimators along with their initializations, your_model1Estimator and your_model2Estimator are assumed to be magically provided for now.
        @property
        def estimators(self):
            r = {}
            if self.enable_your_model1 = True:
                r['your_model1'] = (your_model1Estimator, self.your_model1_init_kwargs, self.your_model1_fit_kwargs)
            if self.enable_your_model2 = True:
                r['your_model2'] = (your_model2Estimator, self.your_model2_init_kwargs, self.your_model2_fit_kwargs)
            return r
    ```