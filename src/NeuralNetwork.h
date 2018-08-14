//-------------------------------------------------------------------------
// Simple back-propagation neural network example
// 2017 - Bobby Anguelov
// MIT license: https://opensource.org/licenses/MIT
//-------------------------------------------------------------------------
// A simple neural network supporting only a single hidden layer

#pragma once
#include "ActivationFunctions.h"
#include <stdint.h>
#include <iostream>
#include <vector>
#include "SafeVector.h"
#include "Matrix.h"

//-------------------------------------------------------------------------

namespace BPN 
{

  //-------------------------------------------------------------------------
  struct Neuron
    {
      Neuron(double a, double v) : activation(a), value(v) 
      {}
      double activation; 
      double value; // = Sigma(activation)
      friend std::ostream& operator<<( std::ostream& os, const BPN::Neuron& n );
    };

  class Network
    {
      friend class NetworkTrainer;

      //-------------------------------------------------------------------------

      //inline static double SigmoidActivationFunction( double x )
      //{
      //    return 1.0 / ( 1.0 + std::exp( -x ) );
      //}

      inline static int32_t ClampOutputValue( double x )
        {
          if ( x < 0.1 ) return 0;
          else if ( x > 0.9 ) return 1;
          else return -1;
        }

  public:

      //struct Settings
      //  {
      //    uint32_t                        m_numInputs;
      //    uint32_t                        m_numHidden;
      //    uint32_t                        m_numOutputs;
      //  };

  public:

      Network(const std::vector<int>& layerSizes, const ActivationFunction* sigma);
      //Network( Settings const& settings, const ActivationFunction* sigma );
      //Network( Settings const& settings, std::vector<double> const& weights, const ActivationFunction* sigma );
      Network( const char* filename);

      std::vector<int32_t> const& Evaluate( std::vector<double> const& input );

      //std::vector<double> const& GetInputHiddenWeights() const { return m_weightsInputHidden; }
      //std::vector<double> const& GetHiddenOutputWeights() const { return m_weightsHiddenOutput; }
      //Matrix const& GetInputHiddenWeights() const { return *m_weightsInputHidden; }
      //Matrix const& GetHiddenOutputWeights() const { return *m_weightsHiddenOutput; }

      void saveToFile(const char* filename) const;

  private:
      void loadFromFile(const char* filename);
      std::string serialize() const;
      void InitializeNetwork();
      void InitializeWeights();
      //void LoadWeights( std::vector<double> const& weights );

      //int32_t GetInputHiddenWeightIndex( int32_t inputIdx, int32_t hiddenIdx ) const 
      //  { 
      //    return inputIdx * m_numHidden + hiddenIdx; 
      //  }
      //int32_t GetHiddenOutputWeightIndex( int32_t hiddenIdx, int32_t outputIdx ) const 
      //  { 
      //    return hiddenIdx * m_numOutputs + outputIdx; 
      //  }


  private:

      int32_t                     m_numLayers;
      int32_t                     m_numInputs;
      int32_t                     m_numOutputs;
      int32_t                     m_numOnLastHidden; // to remove
      std::vector<int>            m_layerSizes;


      //typedef std::vector<Neuron> Layer;
      typedef std::vector<Neuron> Layer;
      std::vector<Layer>          m_neurons;
      Layer*                      m_inputNeurons;
      Layer*                      m_lastHiddenNeurons;
      Layer*                      m_outputNeurons;

      std::vector<int32_t>        m_clampedOutputs;

      // m_wrigntsByLayer[i] is the matrix of weights from layer i to layer i+1
      std::vector<Matrix>         m_weightsByLayer;

      //Matrix*         m_weightsInputHidden;
      //Matrix*         m_weightsHiddenOutput;

      const ActivationFunction*   m_sigma;

  public:

      std::string selfDisplay() const;
      friend std::ostream& operator<<( std::ostream& os, const BPN::Network& n );
    };

}


