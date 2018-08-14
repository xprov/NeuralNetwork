//-------------------------------------------------------------------------
// Simple back-propagation neural network example
// 2017 - Bobby Anguelov
// 2018 - Xavier Provençal
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

      Network(const std::vector<int>& layerSizes, const ActivationFunction* sigma);
      Network( const char* filename);

      std::vector<int32_t> const& Evaluate( std::vector<double> const& input );

      void saveToFile(const char* filename) const;

  private:
      void loadFromFile(const char* filename);
      std::string serialize() const;
      void InitializeNetwork();
      void InitializeWeights();

  private:

      int32_t                     m_numLayers;
      int32_t                     m_numInputs;
      int32_t                     m_numOutputs;
      int32_t                     m_numOnLastHidden; // to remove
      std::vector<int>            m_layerSizes;


      typedef std::vector<Neuron> Layer;
      std::vector<Layer>          m_neurons;
      Layer*                      m_inputNeurons;
      Layer*                      m_lastHiddenNeurons;
      Layer*                      m_outputNeurons;

      std::vector<int32_t>        m_clampedOutputs;

      // m_wrigntsByLayer[i] is the matrix of weights from layer i to layer i+1
      std::vector<Matrix>         m_weightsByLayer;

      const ActivationFunction*   m_sigma;

  public:

      std::string selfDisplay() const;
      friend std::ostream& operator<<( std::ostream& os, const BPN::Network& n );
    };

}


