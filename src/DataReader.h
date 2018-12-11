//-------------------------------------------------------------------------
// Simple back-propagation neural network example
// 2017 - Bobby Anguelov
// 2018 - Xavier Provençal
// MIT license: https://opensource.org/licenses/MIT
//-------------------------------------------------------------------------

#pragma once

#include "NeuralNetworkTrainer.h"
#include <string>

//-------------------------------------------------------------------------

namespace BPN
{
  enum InputDataFormat
    {
      numberList,
      text
    };

  class DataReader
    {
  public:

    DataReader( std::string const& filename, 
                int32_t numInputs, 
                int32_t numOutputs,
                InputDataFormat dataType );



    inline int32_t getNumInputs() const { return m_numInputs; }
    inline int32_t getNumOutputs() const { return m_numOutputs; }

    inline int32_t getNumTrainingSets() const { return 0; }
    bool readTraningData( TrainingData& data );

  private:

    static void CreateTrainingData( TrainingData& data, std::vector<TrainingEntry>& entries );

  private:

    std::string                     m_filename;
    int32_t                         m_numInputs;
    int32_t                         m_numOutputs;
    InputDataFormat                 m_dataFormat;
    };
}
