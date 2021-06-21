```Hypernets``` is an Automated Machine learning(AutoML) framework which allows the users to easily develop various kinds of AutoML and Automated Deep Learning(AutoDL) tools without reinventing some necessary components which are often common to such tools. Before ```Hypernets```, there already existed many AutoML tools. However, these tools are usually designed for some specific purposes thus not convenient to be generalized to other ones. As a result, the AutoML community may have to take a lot of efforts to repeatedly develop some common parts before deploying their AutoML models due to the lack of an underlying AutoML framework. 

```Hypernets``` can save such efforts to a large extent while offer more possibilities. 
- First, it decouples the basic components of a general AutoML procedure (Fig. \ref) as four distinct parts: the ```HyperSpace```, the ```Searcher```, the ```HyperModel```, and the ```Estimation Strategy``` (Fig. \ref). This idea is motivated by allowing users to manipulate each component of an AutoML model with ```Hypernets``` accordingly for different purposes. 
- Second, the ```HyperSpace``` is designed to be a powerful search space. The ```HyperSpace``` consists of three different kinds of space: the **module space**, the **parameter space** and the **connection space** (Fig. \ref), where the module space is designed to contain various machine learning models, data preprocessings or feature engineerings, the parameter space provides the parameters to be searched for machine learning models and the connection space determines the way how different module spaces connect. These connected module spaces and parameter spaces finally give us a highly comprehensive search space. This search space is able to describe the full-pipeline machine learning modelling ranging from data preprocessing to model ensemble. 
- Third, ```Hypernets``` provides many search algorithms including the simple methods, such as Random Search and Grid Search, and the advanced ones such as Monte-Carlo Tree Search. Users can not only simply choose one from these efficient search methods but also design new search algorithms in a similar way. 
- Finally, ```Hypernets``` also supports many advacend techniques to futher improve performances of the trained machine learning models. For example, users can apply early stopping to accelerate the training process and prevent overfitting; data cleaning can be applied to improve data quality; data drift detection can be enabled to improve generalization ability of the model, etc. 

Based on the above brief introduction, using the ```Hypernets``` to implement an AutoML task can now be decomposed as three parts: designing the **search space**, an instance of the ```Hyperspace```, constructing the **Hypermodel** which will be sampled from the search space using a searcher provided by ```Hypernets``` during the search process, and building the **Estimator** which recieves a sampled Hypermodel, evaluates it and then returns the corresponding rewards such that the searcher can update the Hypermodel to be sampled based on the rewards. 

We provide a [toy example](#sec_eg), designing an AutoML task with KNN, for the purpose of helping the readers walk through the full pipeline of implementing the ```Hypernets``` to an AutoML task. 

To reveal the core features and ideas of ```Hypernets```, we first continue to solve the problem defined in the very begining--how to perform parameter tuning of KNN automatically using ```Hyernets```--but with a different manner: we view the parameter tuning problem as a complete AutoML task and constrcut an AutoML tool for this task from scratch using ```Hypernets```. As introduced above, this constructing procedure contains 3 steps and we will follow these steps in the following. 
- ***Designing the search space.*** In the case of parameter tuning, our HyperSpace, the search space of the AutoML task, is very simple in the sense that there is only one module space which contains only one machine learning model--our KNN model--along with its parameter space. To incorporate these spaces, we first define the parameter space for tunable parameters with different values, and then build the whole HyperSpace to include this parameter space so that the search algorithm can search suitable parameters among avaliable ones.  
    ```python
    class Param_space(object):

        def __init__(self, **kwargs):
            super(Param_space, self).__init__()

        @property
        #The following function returns a dictionary containing tunable parameters, where 
        # the avaliable values for each parameter are those provided by the arguments of 
        # Choice(), a class which in fact inherits from the ParameterSpace, one of the three 
        # basic kinds of the HyperSpace. In other words, all values of the returned 
        # dictionary are parts of the parameter space if it is a Choice(). 
        def knn(self):
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
                modules = [ModuleSpace(name=f'{model["cls"].__name__}', **model)] #make a module space of the HyperSpace using ModuleSpace()
                outputs = ModuleChoice(modules)(hyper_input) #pick a model from the module space if there are mutiple ones as the output of the search space. Here we only have a KNN model.
                space.set_inputs(hyper_input)

            return space
    ```
- ***Constructing the Hypermodel.*** The HyperMdel does not reuqire many modifications for our specific task since many core functionalities of the HyperMdel have already been well defined in ```Hypernets``` and are common accross different machine learning models and tasks. We only pay attention to two functions, the ```_get_estimator```, which returns the corresponding KNN model of the sampled search space, and the ```load_estimator```, which loads the configurations of the saved model. The most important method for a HyperModel is the "search" method. By calling the ```search``` method, the search algorithm searches in the search space and returns a sample of the search space to be utilized for the HyperModel. This HyperModel is then evaluated based on the chosen reward metric and updated towards the optimizing direction.    
    ```python
    class KnnModel(HyperModel):
        def __init__(self, searcher, reward_metric=None, task=None):
            
            super(KnnModel, self).__init__(searcher, reward_metric=reward_metric, task=task)
        
        def _get_estimator(self, space_sample):
            return KnnEstimator(space_sample, task=self.task)
        
        def load_estimator(self, model_file):
            return KnnEstimator.load(model_file)
    ```
- ***Building the Estimator.*** Building the Estimator often takes the most efforts for implementing a new AutoML task using ```Hypernets```. These ```Estimators``` required by ```Hypernets``` is in fact a more general notion than the frequently used one--the machine learning models. Fortunately, for our case of parameter tuning of KNN, the ```Estimator``` is easy to be implemented since the sampled search space only contains one machine learning model which is the only thing that needs to be evaluated by the ```Estimator```.  
    ```python
    class KnnEstimator(Estimator):
        def __init__(self, space_sample, task='binary'):
            super(KnnEstimator, self).__init__(space_sample, task)

            out = space_sample.get_outputs()[0]
            kwargs = out.param_values
            kwargs = {key: value for key, value in kwargs.items() if not isinstance(value, HyperNode)}

            cls = kwargs.pop('cls')
            self.model = cls(**kwargs)
            self.cls = cls
            self.model_args = kwargs
        
        def fit(self, X, y, **kwargs):
            self.model.fit(X, y, **kwargs)

            return self
        
        def predict(self, X, **kwargs):
            pred = self.model.predict(X, **kwargs)

            return pred

        def evaluate(self, X, y, **kwargs):
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
            return []
    ```
## Searcher<span id=sec_searcher>

## Easy deploying of your AutoML task<span id=sec_eg>
To apply your end-to-end AutoML models built with the ```Hypernets```, the readers usually first design a search space, which mainly includes transformations of the data, feature engineerings, and the desired estimators, the most important part and the primary work you will do for designing your AutoML model with the ```Hypernets```. With this search space in hand, the readers then choose a searcher from those defined in the ```Hypernets```, such as ```RandomSearcher```, whose functionality is to repeatedly 'search' samples from the search space. This searcher is then passed as an argument to your model, an object inherited from the ```Hypermodel```. Finally, the ```search``` method of the Hypermodel is called to repeat the following procedures: the searcher searches in the search space and samples a full-pipeline model from the search space, the estimator fits the sampled model of the search space, evaluates its performance, and then updating the searcher to get a new sample of the search space until the end. The above process is summarized as follows with 4 lines of codes after loading the data:
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
To help the readers walk through these steps, we provide in the following subsections deploying k-nearest neighbors with the ```Hypernets``` as a simple example to examine details behind each line of the above codes.
### Designing a search space
The search space, an object of the ```Hyperspace``` defined in the ```Hypernets```, is composed of two important components: the ***preprocessor***, which focuses on the data preprocessing and the feature engineerings such that the data and features can be treated by the estimators properly, and the ***estimators***, whose implementation details will be discussed [later](#sec_model) and here we simply assume that the estimators are magically provided. Therefore, to successfully design a search space, we need a ```SearchSpaceGenerator``` to wrap these two components as a whole. 

**Preprocessors** in a search space are connected through ```pipeline```. Since both the preprocessors and ```pipeline``` are not closely related to any specific models, fortunately, we can directly borrow them from the ```HyperGBM``` where they are already well defined and need not to be modified much. The preprocessors are created and connected by calling the function ```create_preprocessor```. Readers can also modify the ```create_preprocessor``` to manipulate the preprocessings of the data. 

Likewise, the **estimators** in the search space are created by calling the function ```create_estimators```, which, on the other hand, needs to be carefully modified for your spcific models, i.e. k-nearest neighbors here. 

Now we can define a class ```SearchSpaceGenerator``` which has the above functions as its methods for the purpose of designing a specific search space. Moreover, to conveniently manipulate the initializations of the models or even inlcude other models defined in scikit-learn such as support vector machines into our search space, we can further define a subclass of ```SearchSpaceGenerator```, which can be named as "YourModelSearchSpaceGenerator" and summarized as follows:
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
Finally, a search space can be created in the following way
```python
get_your_search_space = YourModelSearchSpaceGenerator()
```
Details for the search space of the k-nearest neighbors are presented in ```search_space.py```, where the class ```KNNSearchSpaceGenerator``` is structured as
```python
class KNNSearchSpaceGenerator(SearchSpaceGenerator):
    def __init__(self, **kwargs):
        super(KNNSearchSpaceGenerator, self).__init__(**kwargs)

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
```
 Following the same way, readers can define their own search space by modifying this file or ```search_space.py``` of the ```HyperGMB``` accordingly.

### Choosing a searcher
Since many efficient searchers have already been provided in the ```Hypernets```,  it is fairly easy for the readers to simply choose one of them and send the search space you just defined to this searcher. For example,
```python
searcher = RandomSearcher(search_space)
```
One can also take more efforts to design new kinds of searcher by refering to [Searcher](#sec_searcher).

### Constructing the Hypermodel to receive the searcher<span id=sec_model> 
This section needs additional attentions for its importance. Here, we devotes to constructing the ```HyperYourModel``` of your task, which is an object inherited from the ```Hypermodel``` of the ```Hypernets```. For our example of k-nearest neighbors, this is simply named as ```toy_KNN```. It is not hard for the readers to build ```HyperYourModel``` with models other than the k-nearest neighbors by following steps discusssed in this section. 

Basically, to define a class ```HyperYourModel```, one needs to define two functions properly: 
1. A function which returns the estimator of the HyperYourModel from the search space returned by the searcher
    ```python 
    def _get_estimator(space_sample):
        #space_sample, a Hyperspace, is returned by a searcher
        estimator = HyperYourModelEstimator(some_args)
        return estimator 
    ```
    This function overwrites the ```_get_estimator``` method of the ```Hypermodel```, from which the ```HyperYourModel``` is inherited. One may immediately notice that the returned estimatos are actually ```HyperYourModelEstimator```, not the typical "estimator" which usually refers to some m 
    
    The uniqueness of each ```HyperYourModel```, e.g. the Hypermodels with k-nearest neighbors or support vector machine, is provided by the class ```HyperYourModelEstimator``` through receiving different search space returned by the searcher. We discuss this uniqueness now and name our special HyperYourModelEstimator as ```toy_KNN_estimator``` since we are taking the k-nearest neighbors as our example. Although a ```HyperYourModelEstimator``` usually includes many arguments and functions to support some advanced features of ```Hypernets```, fortunately, there is nearly nothing needs to be rewritten from scratch for the reason that many of the arguments and functions have already been provided in the ```HyperGBMEstimator``` of the ```HyperGMB``` package. The more deep reason for this convenience lies in the fact that the ```HyperYourModelEstimator``` should be a more general "estimator" which not only includes the typical estimators, i.e. machine learning models, but also the whole pipeline from the data transformations to the machine learning models, where the steps before introducing these machine learning models to the pipeliine are common in different cases. The uniqueness is then due to the new estimators, i.e. machine learning models, that you include into your search space when it is designed and was assumed to be magically provided there. 
    
    We now turn to the implementation details of defining new estimators to explain the reason of such uniqueness. 

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
### Calling the ```search``` method
