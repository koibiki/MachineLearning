from bayes.origin.mulyinomial_native_bayes import MultinomialNB
from bayes.origin.gaussian_native_bayes import GaussianNB
from utils.data_utils import DataUtils

x, y = DataUtils.get_data_set('data/mushroom.txt', split=',')
print(x)
print(y)
nb = GaussianNB()
nb.fit(x, y)


nb.evaluate(x, y)

