project( NeuralNetwork )

#find_package (Threads)
#find_package (X11)
#find_package( OpenCV REQUIRED )

set( CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g" ) 

add_executable( trainBPN trainBPN.cpp NeuralNetwork.cpp NeuralNetworkTrainer.cpp TrainingDataReader.cpp Matrix.cpp )

#target_link_libraries( main ${CMAKE_THREAD_LIBS_INIT})
#target_link_libraries( main ${OpenCV_LIBS} )

