#include <cstddef>
#include "LFArrayFloat3d.h"

size_t LFArrayFloat3d::GetSize()
{
    return size()/3;
}

void LFArrayFloat3d::Allocate(size_t nElements)
{
    resize(nElements*3);
}

void LFArrayFloat3d::AllocateBytes(size_t nBytes)
{
    resize(nBytes/sizeof(float));
}

float* LFArrayFloat3d::GetElement(size_t index)
{
    size_t n=index*3;
    return n<size()?&(operator[](n)):NULL;
}
