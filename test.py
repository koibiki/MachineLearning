from bayes.origin.mulyinomial_native_bayes import MultinomialNB
from utils.data_utils import DataUtils

x, y = DataUtils.get_data_set('data/balloon1.5.txt', split=',')
print(x)
print(y)
nb = MultinomialNB()
nb.fit(x, y)
nb.evaluate(x, y)
x, y = DataUtils.get_data_set('data/balloon1.5.txt', split=',')
nb.evaluate(x, y)
