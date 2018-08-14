//-------------------------------------------------------------------------
// Simple back-propagation neural network example
// 2017 - Bobby Anguelov
// 2018 - Xavier Provençal
// MIT license: https://opensource.org/licenses/MIT
//-------------------------------------------------------------------------

#include <stdlib.h>
#include <iostream>
#include <sstream>

#include "NeuralNetworkTrainer.h"
#include "TrainingDataReader.h"
#include "Matrix.h"
#include "vectorstream.h"

#if _MSC_VER
#pragma warning(push, 0)
#pragma warning(disable: 4702)
#endif

#include "cmdParser.h"

#if _MSC_VER
#pragma warning(pop)
#endif

// Operators from "vectorstream.h"
using BPN::operator<<;
using BPN::operator>>;

//-------------------------------------------------------------------------

int main( int argc, char* argv[] )
{

  //BPN::Network nin( std::cin );
  //std::cout << nin << std::endl;
  //exit(0);



  cli::Parser cmdParser( argc, argv );
  cmdParser.set_required<std::string>( "d", "dataFile", "Path to training data csv file." );
  cmdParser.set_required<uint32_t>( "in", "numInputs", "Num Input neurons." );
  cmdParser.set_required<std::string>( "hidden", "numHidden", "Comma separated list of the hidden layers sizes (e.g.: 3,2,3 or 1)." );
  cmdParser.set_required<uint32_t>( "out", "numOutputs", "Num Output neurons." );
  cmdParser.set_optional<uint32_t>( "m", "maxEpoch", 100, "Maximum num of iterations." );
  cmdParser.set_optional<double>( "l", "learningRate", 0.01, "Multiplicative coefficient on error gradient" );
  cmdParser.set_optional<double>( "mom", "momentum", 0.9, "Multiplicative coefficient applied on previous error delta when non using batch learning." );
  cmdParser.set_optional<bool>( "b", "batchLearning", false, "Use batch learning (1 : yes, 0 : no)." );
  cmdParser.set_optional<double>( "a", "accuracy", 95, "Desired accuracy. Training stops when the desired accuracy is obtained." );
  cmdParser.set_optional<int32_t>( "v", "verbose", 1, "Verbosity level.\n     Level 0: quiet mode.\n     Level 1: prints training evolution.\n     Level 2: prints NN at initialization and at the end.\n     Level 3: prints NN at every iteration of the learning phase." );

  if ( !cmdParser.run() )
    {
      std::cout << "Invalid command line arguments";
      return 1;
    }

  std::string       trainingDataPath  = cmdParser.get<std::string>( "d" ).c_str();
  uint32_t const    numInputs         = cmdParser.get<uint32_t>( "in" );
  std::string const numsHiddens       = cmdParser.get<std::string>( "hidden" );
  uint32_t const    numOutputs        = cmdParser.get<uint32_t>( "out" );
  uint32_t const    maxEpoch          = cmdParser.get<uint32_t>( "m" );
  double            learningRate      = cmdParser.get<double>( "l" );
  double            momentum          = cmdParser.get<double>( "mom" );
  bool              batchLearning     = cmdParser.get<bool>( "b" );
  double            accuracy          = cmdParser.get<double>( "a" );
  int32_t           verbosity         = cmdParser.get<int32_t>( "v" );

  // Read layers sizes. The first layer is the input layer, the last layer
  // is the output. All other layers are hidden.
  std::vector<int> hiddenLayersSizes;
  std::stringstream ss(numsHiddens);
  ss >> hiddenLayersSizes;

  std::vector<int> layerSizes;
  layerSizes.push_back(numInputs);
  layerSizes.insert( layerSizes.end(), hiddenLayersSizes.begin(), hiddenLayersSizes.end() );
  layerSizes.push_back(numOutputs);



  BPN::TrainingDataReader dataReader( trainingDataPath, numInputs, numOutputs );
  if ( !dataReader.ReadData() )
    {
      return 1;
    }
  if ( verbosity >= 1 )
    {
      std::cout << "Input file: " << trainingDataPath
        << "\nRead complete: " << dataReader.getNumEnties()
        << " inputs loaded" << std::endl;
    }

  // Select activation function
  //
  BPN::ActivationFunction* sigma = new BPN::Sigmoid();
  //BPN::ActivationFunction* sigma = new BPN::Sigmoid(2.0);
  //BPN::ActivationFunction* sigma = new BPN::ReLU();

  // Create neural network

  //BPN::Network::Settings networkSettings{ numInputs, numHidden, numOutputs };
  BPN::Network nn( layerSizes, sigma );
  if (verbosity >= 2)
    {
      std::cout << nn << std::endl;
    }

  // Create neural network trainer
  BPN::NetworkTrainer::Settings trainerSettings;
  trainerSettings.m_learningRate = learningRate;
  trainerSettings.m_momentum = momentum;
  trainerSettings.m_useBatchLearning = batchLearning;
  trainerSettings.m_maxEpochs = maxEpoch;
  trainerSettings.m_desiredAccuracy = accuracy;
  trainerSettings.m_verbosity = verbosity;

  BPN::NetworkTrainer trainer( trainerSettings, &nn );
  trainer.Train( dataReader.getTrainingData() );
  if ( verbosity >= 2) 
    {
      std::cout << nn << std::endl;
    }

  std::cout << nn.serialize() << std::endl;


  return 0;
}

