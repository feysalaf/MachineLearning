from models import perceptron
from preprocessing import preprocessor

print("\n")
print("-"*44)
print("Welcome to the Classification Program")
print("-"*44)


filename = input("\nEnter the filename:(Default:raw_data.csv)")
obj = perceptron(filename)
obj.use_perceptron()