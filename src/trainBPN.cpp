//-------------------------------------------------------------------------
// Simple back-propagation neural network example
// 2017 - Bobby Anguelov
// 2018 - Xavier Provençal
// MIT license: https://opensource.org/licenses/MIT
//-------------------------------------------------------------------------

#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <assert.h>

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
  cmdParser.set_optional<std::string>( "l", "layers", "[]", "Comma separated list of the layers sizes (e.g. 16,4,4,3 or 1,1,1).\n     First layer is the input neurons.\n     Last layer is the output layer." );
  cmdParser.set_optional<std::string>( "i", "import", "", "Import neural network from file before training. ( \"-\" stands for stdin)" );
  cmdParser.set_optional<std::string>( "e", "export", "", "Export neural network to file after training. ( \"-\" stands for stdout)" );

  cmdParser.set_optional<std::string>( "s", "activation", "Sigmoid(1)", "The actiation function. Available options are:\n     Sigmoid(k), logistic function with stepness ``k``.\n     ReLU,       rectified linear unit.");
  cmdParser.set_optional<uint32_t>( "m", "maxEpoch", 100, "Maximum num of iterations." );
  cmdParser.set_optional<double>( "r", "learningRate", 0.01, "Multiplicative coefficient on error gradient" );
  cmdParser.set_optional<double>( "mom", "momentum", 0.9, "Multiplicative coefficient applied on previous error delta when non using batch learning." );
  cmdParser.set_optional<bool>( "b", "batchLearning", false, "Use batch learning (1 : yes, 0 : no)." );
  cmdParser.set_optional<double>( "a", "accuracy", 95, "Desired accuracy. Training stops when the desired accuracy is obtained." );
  cmdParser.set_optional<int32_t>( "v", "verbose", 1, "Verbosity level.\n     Level 0: quiet mode.\n     Level 1: prints training evolution.\n     Level 2: prints NN at initialization and at the end.\n     Level 3: prints NN at every iteration of the learning phase." );

  if ( !cmdParser.run() )
    {
      std::cout << "Invalid command line arguments";
      return 1;
    }

  std::string const trainingDataPath   = cmdParser.get<std::string>( "d" ).c_str();
  std::string const layers             = cmdParser.get<std::string>( "l" );
  std::string const importFile         = cmdParser.get<std::string>( "i" );
  std::string const exportFile         = cmdParser.get<std::string>( "e" );
  std::string const activationFunction = cmdParser.get<std::string>( "s" );
  uint32_t const    maxEpoch           = cmdParser.get<uint32_t>( "m" );
  double            learningRate       = cmdParser.get<double>( "r" );
  double            momentum           = cmdParser.get<double>( "mom" );
  bool              batchLearning      = cmdParser.get<bool>( "b" );
  double            accuracy           = cmdParser.get<double>( "a" );
  int32_t           verbosity          = cmdParser.get<int32_t>( "v" );

  if ( layers.compare("[]") == 0 && importFile.compare("") == 0 )
    {
      std::cerr << "At least one parameter among -l or -i must be specified. (See help for more impormations)" << std::endl;
      exit(1);
    }

  if ( layers.compare("[]") != 0 && importFile.compare("") != 0 )
    {
      std::cerr << "Only one parameter among -l and -i may be specified. (See help for more impormations)" << std::endl;
      exit(1);
    }

  // Select activation function
  //
  BPN::ActivationFunction* sigma = BPN::ActivationFunction::deserialize( activationFunction );
  //BPN::ActivationFunction* sigma = new BPN::Sigmoid();
  //BPN::ActivationFunction* sigma = new BPN::Sigmoid(2.0);
  //BPN::ActivationFunction* sigma = new BPN::ReLU();

  // Create neural network
  BPN::Network* nn = NULL;;

  if ( layers.compare("[]") != 0 )
    {

      // Read layers sizes. The first layer is the input layer, the last layer
      // is the output. All other layers are hidden.
      std::vector<int> layerSizes;
      std::stringstream ss(layers);
      ss >> layerSizes;

      nn = new BPN::Network( layerSizes, sigma );
    }
  else if ( importFile.compare("") != 0 )
    {
      if ( importFile.compare("-") == 0 )
        {
          nn = new BPN::Network( std::cin );
        }
      else
        {
          std::fstream fs;
          fs.open( importFile, std::fstream::in );
          nn = new BPN::Network( fs );
        }
    }

  assert( nn != NULL );

  if (verbosity >= 2)
    {
      std::cout << *nn << std::endl;
    }


  BPN::TrainingDataReader dataReader( trainingDataPath, nn->getNumInputs(), nn->getNumOutputs() );
  if ( !dataReader.ReadData() )
    {
      return 1;
    }
  if ( verbosity >= 1 )
    {
      std::cout << "Input data file: " << trainingDataPath
        << "\nRead complete: " << dataReader.getNumEnties()
        << " inputs loaded" << std::endl;
    }


  // Create neural network trainer
  BPN::NetworkTrainer::Settings trainerSettings;
  trainerSettings.m_learningRate = learningRate;
  trainerSettings.m_momentum = momentum;
  trainerSettings.m_useBatchLearning = batchLearning;
  trainerSettings.m_maxEpochs = maxEpoch;
  trainerSettings.m_desiredAccuracy = accuracy;
  trainerSettings.m_verbosity = verbosity;

  BPN::NetworkTrainer trainer( trainerSettings, nn );
  trainer.Train( dataReader.getTrainingData() );
  if ( verbosity >= 2) 
    {
      std::cout << *nn << std::endl;
    }

  if ( exportFile.compare("") != 0 )
    {
      if ( exportFile.compare("-") == 0 )
        {
          std::cout << nn->serialize() << std::endl;
        }
      else
        {
          std::fstream fs;
          fs.open( exportFile, std::fstream::out );
          fs << nn->serialize() << std::endl;
        }
    }
  delete nn;
  return 0;
}

