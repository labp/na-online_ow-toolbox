#ifndef LFARRAYFLOAT3D_H
#define LFARRAYFLOAT3D_H

#include <cstddef>
#include <vector>
using namespace std;

/**
 * 3D Array of floats
 */
class LFArrayFloat3d: public vector< float >
{
public:
    /**
     * Returns the number of Elements (float[3])
     */
    size_t GetSize();
    /**
     * Creates the Array of nElements*3 floats
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
