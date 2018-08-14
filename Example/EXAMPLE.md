# NeuralNetwork Example

# You can use the supplied training data file to test the neural network...


# Provided by Bobby Anguelov (I would definitively like to know what this data means!)
# num input : 16
# num output : 3
./trainBPN -d ../Example/ExampleDataSet.csv -in 16 -hidden 16 -out 3

# A single bit in input, same bit in output (for debugging)
# num input : 1
# num output : 1
./trainBPN -d ../Example/bit -in 1 -hidden 2,3,2 -out 1

# 30x30 binary images to classify in three categories : 
# 1,0,0 --> straight line
# 0,1,0 --> rectangle
# 0,0,1 --> random image
# All images contain about the same number of 1s and 0s.

# num input : 900
# num output : 3
./trainBPN -d ../Example/detectShape -in 900 -hidden 20 -out 3
