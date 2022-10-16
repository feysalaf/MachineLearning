from preprocessing import preprocessor
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from random import seed
from random import randint

seed(1)


class perceptron:
    def __init__(self,file_name):
        self.processed = preprocessor(file_name)
        self.train_data,self.test_data = self.processed.get_train_test_set()
        print("\nLength of training dataset: " + str(len(self.train_data)))
        print("\nLength of testing dataset: " + str(len(self.test_data))+"\n")



    def use_perceptron(self):
        p = Perceptron(random_state=randint(44,1000))
        k = KNeighborsClassifier(n_neighbors=randint(1,10))

        train_labels = self.train_data['Gender']
        test_labels = self.test_data['Gender']
        
        #print(self.train_data)
        print("\nFitting data to perceptron...")
        p.fit(self.train_data,train_labels)
        

        
        predictions_train = p.predict(self.train_data)
        predictions_test = p.predict(self.test_data)
        train_score = accuracy_score(predictions_train, train_labels)
        print("Score on train data using Perceptron: ", train_score)
        test_score = accuracy_score(predictions_test, test_labels)
        print("Score on test data using Perceptron: ", test_score)
        print("\n")

        print("\Fitting data to KNN...")

        k.fit(self.train_data,train_labels)
        predictions_train = k.predict(self.train_data)
        predictions_test = k.predict(self.test_data)
        train_score = accuracy_score(predictions_train, train_labels)
        print("Score on train data using KNN: ", train_score)
        test_score = accuracy_score(predictions_test, test_labels)
        print("Score on test data using KNN: ", test_score)



if __name__ == "__main__":
    obj = perceptron("raw_data.csv")
    obj.use_perceptron()