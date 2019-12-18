PROJECT( NeuralNetwork )

SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" )
SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g" ) 

ADD_EXECUTABLE( trainBPN trainBPN.cpp NeuralNetwork.cpp NeuralNetworkTrainer.cpp
    DataReader.cpp Matrix.cpp ActivationFunctions.cpp )

# Add an executable compiled from evalNN.cpp
ADD_EXECUTABLE( evalNN evalNN.cpp NeuralNetwork.cpp DataReader.cpp ActivationFunctions.cpp Matrix.cpp )





