#suppose that we already have the prepared data (X, y), we already split the data into the
#training set and the validation set
#suppose that I also have a search space given by get_search_space()

from toyknn import toy_KNN
from hypernets.searchers import RandomSearcher


def get_search_space():
    return None

searcher = RandomSearcher(get_search_space)
sampled_model = toy_KNN(searcher, task='', reward_metric='accuracy', callbacks=[])
#what is the function of callbacks?
sampled_model.search(X_train, y_train, X_evl=None, y_evl=None, cv=False)

best_classifier = sampled_model.load_estimator(sampled_model.get_best_trial().model_file)

