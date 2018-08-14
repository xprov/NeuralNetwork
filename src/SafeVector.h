
#pragma once
#include <vector>
#include <iostream>

namespace BPN {
  template<typename T>
  class SafeVector : public std::vector<T>
  {
public:
    T& operator[](int n)
      {
        if (n>=0 && n<(int)this->size())
          {
            return std::vector<T>::operator[](n);
          }
        else
          {
            std::cerr << "\nOut of bound, n=" << n << ", size=" << this->size() << std::endl;
            std::cout << *((int*) NULL) << std::endl;
          }
      }


    T operator[](int n) const
      {
        if (n>=0 && n<(int)this->size())
          {
            return std::vector<T>::operator[](n);
          }
        else
          {
            std::cerr << "\nOut of bound, n=" << n << ", size=" << this->size() << std::endl;
            std::cout << *((int*) NULL) << std::endl;
          }
      }

  };

}
