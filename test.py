from bayes.origin.mulyinomial_native_bayes import MultinomialNB
from utils.data_utils import DataUtils

x, y = DataUtils.get_data_set('data/balloon1.0.txt', split=',')
print(x)
print(y)
nb = MultinomialNB()
nb.feed_data(x, y)