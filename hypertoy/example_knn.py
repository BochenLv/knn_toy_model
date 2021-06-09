#suppose that we already have the prepared data (X, y), we already split the data into the
#training set and the validation set
#suppose that I also have a search space given by get_search_space()


from hypertoy.toyknn import toy_KNN
from hypertoy.search_space import search_space_eg
from hypernets.searchers import RandomSearcher


search_space = search_space_eg
searcher = RandomSearcher(search_space)
sampled_model = toy_KNN(searcher, task='', reward_metric='accuracy', callbacks=[])
sampled_model.search(X_train, y_train, X_evl=None, y_evl=None, cv=False)

best_classifier = sampled_model.load_estimator(sampled_model.get_best_trial().model_file)

