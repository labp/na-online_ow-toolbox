#ifndef LFARRAYFLOAT2D_H
#define LFARRAYFLOAT2D_H

#include <cstddef>
#include <vector>
using namespace std;

/**
 *2D Array of floats
 */
class LFArrayFloat2d: public vector< float >
{
public:
    /**
     * Returns the number of Elements (float[2])
     */
    size_t GetSize();
    /**
     * Creates the Array of nElements*2 floats
     */
    void Allocate( size_t nElements );
    /**
     * Creates the Array nBytes long
     */
    void AllocateBytes( size_t nBytes );
     /**
      * Returns the Pointer to the Element, or NULL if don't exists
      */
    float* GetElement( size_t index = 0 );
};
#endif
