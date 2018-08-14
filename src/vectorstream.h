//-------------------------------------------------------------------------
// Simple back-propagation neural network example
// 2018 - Xavier Provençal
// MIT license: https://opensource.org/licenses/MIT
//-------------------------------------------------------------------------
//
# pragma once
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>


#include <iomanip>
#include <limits>
#include <exception>

namespace BPN
{


  template <typename T>
    std::ostream& operator<<( std::ostream& os, const std::vector<T>& v )
      {
        os << std::setprecision(std::numeric_limits<long double>::digits10 + 1);
        if (v.size() == 0) 
          {
            os << "[]";
            return os;
          }
        os << "[" << v[0];
        for (unsigned int i=1; i<v.size(); ++i) 
          {
            os << ", " << v[i];
          }
        os << "]";
        return os;
      };



  template <typename T>
    std::istream& operator>>( std::istream& is, std::vector<T>& v )
      {
        std::string s;
        getline(is, s);
        std::stringstream ss(s);
        std::string token;
        T t;
        v.clear();
        while ( std::getline(ss, token, ',') )
          {
            size_t pos = token.find_first_not_of( " [" );
            token = token.substr( pos );
            std::stringstream sss(token);
            sss >> t;
            v.push_back(t);
          }
        return is;
      }
}