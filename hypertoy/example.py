from hypertoy.toyknn import KnnModel
from hypertoy.search_space_complicated import search_space_eg
from hypernets.searchers import RandomSearcher
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X[:3]

search_space = search_space_eg
searcher = RandomSearcher(search_space)
sampled_model = KnnModel(searcher, task='multiclass', reward_metric='accuracy', callbacks=[])
sampled_model.search(X_train, y_train, X_evl=None, y_evl=None, cv=False)

best_classifier = sampled_model.load_estimator(sampled_model.get_best_trial().model_file)

