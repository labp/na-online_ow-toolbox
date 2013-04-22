#include <cstddef>

#include "LFArrayFloat2d.h"

size_t LFArrayFloat2d::GetSize()
{
    return size() / 2;
}

void LFArrayFloat2d::Allocate( size_t nElements )
{
    resize( nElements * 2 );
}

void LFArrayFloat2d::AllocateBytes( size_t nBytes )
{
    resize( nBytes / sizeof(float) );
}

float* LFArrayFloat2d::GetElement( size_t index )
{
    size_t n = index * 2;
    return n < size() ? &( operator[]( n ) ) : NULL;
}
