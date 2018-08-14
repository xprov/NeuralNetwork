//-------------------------------------------------------------------------
// Simple back-propagation neural network example
// 2017 - Bobby Anguelov
// 2018 - Xavier Provençal
// MIT license: https://opensource.org/licenses/MIT
//-------------------------------------------------------------------------
//
#pragma once

#include <cmath>


namespace BPN
{
    enum class ActivationFunctionType
    {
        Sigmoid, ReLU
    };

    class ActivationFunction 
      {
      public:
        /**
         * Let f be the activation function, returns f(x)
         */
        virtual double evaluate( double x ) const 
          {
            (void)x; 
            return 0.0;
          }

        /**
         * Let f be the activation function, returns f'(x)
         *
         * Second parameter is the pre-computed f(x). In some cases, like for
         * the sigmoid activation function, it is faster to compute f'(x) from
         * f(x) then from x.
         */
        virtual double evalDerivative( double x, double fx = 0.0 ) const
          { 
            (void) x;
            (void) fx;
            return 0.0;
          }
      };

    class Sigmoid : public ActivationFunction 
      {
        /**
         *                   1
         * f(x) =    -----------------
         *           1 + exp(lambda*x)
         *
         * f'(x) = lambda * f(x) * (1-f(x))
         */
      public:

        Sigmoid() : lambda(1.0)
          { }

        Sigmoid( double lambda ) : lambda(lambda)
          { }

        inline double evaluate( double x ) const
          {
            return 1.0 / ( 1.0 + std::exp( -lambda * x ) );
          }

        inline double evalDerivative( double x, double fx = 0.0 ) const
          {
            (void) x; // avoid compilation warning for unused variable
            return lambda * fx * (1.0-fx);
          }
        const double lambda;
      };

    class ReLU : public ActivationFunction 
      {
        /**
         *                   1
         * f(x) =    -----------------
         *           1 + exp(lambda*x)
         *
         * f'(x) = lambda * f(x) * (1-f(x))
         */
      public:

        ReLU( ) { }

        inline double evaluate( double x ) const
          {
            //return std::log(1.0 + std::exp(x));
            return (x>0) ? x : 0;
          }

        inline double evalDerivative( double x, double fx = -1.0 ) const
          {
            (void) x; (void) fx;
            return (x > 0) ? -1 : 0;
          }
      };
}
