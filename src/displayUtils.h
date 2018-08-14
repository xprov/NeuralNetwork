# pragma once
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>

namespace DisplayUtils {

    template <typename T>
      std::string vectorToString(const std::vector<T>& v)
        {
          std::ostringstream ss;
          if (v.size() == 0) return "[]";
          ss << "[" << v[0];
          for (unsigned int i=1; i<v.size(); ++i) 
            {
              ss << ", " << v[i];
            }
          ss << "]";
          return ss.str();
        };

    template <typename T>
      std::string matrixToString(const std::vector<T>& v, int nlines, int ncols, std::string prefix = "" )
        {
          std::ostringstream ss;
          for ( int i=0; i<nlines; ++i ) 
            {
              ss << prefix << "[" << v[i*ncols];
              for ( int j=1; j<ncols; ++j ) 
                {
                  ss << ",\t" << v[i*ncols+j];
                }
              ss << "]\n";
            }
          return ss.str();
        }


}

