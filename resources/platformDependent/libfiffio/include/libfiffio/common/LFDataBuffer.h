#ifndef LFDATABUFFER_H
#define LFDATABUFFER_H

#include <cstddef>
#include <inttypes.h>
#include <vector>
using namespace std;

/**
 * Buffer containing measurement data (300)
 */
class LFDataBuffer
{
public:
    enum datatype_t/**< Type of the data */
    {
        dt_unknown = -1,/**< Undefined */
        dt_int16 = 2,/**< int16_t */
        dt_int32 = 3,/**< int32_t */
        dt_float = 4/**< float */
    };
protected:
    datatype_t m_DataType;/**< Type of the data in the buffer, default == dt_unknown */
    size_t m_Bytes;/**< Length of the buffer in bytes, default == 0 */
    void* m_Buffer;/**< Pointer to vector, default == NULL */
public:
    LFDataBuffer();
    ~LFDataBuffer();
    /**
     * Clears the buffer, sets data type = undefined and size = 0
     */
    void Init();
    /**
     * Creates the buffer
     */
    void Allocate( datatype_t type, size_t bytes );
    /**
     * Returns the type of variables in buffer
     */
    datatype_t GetDataType();
    /**
     * Returns the size of one element of the array in bytes
     */
    int GetElementSize();
    /**
     * Returns the number of variables in buffer
     */
    size_t GetSize();
    /**
     * Returns the size of the buffer in bytes
     */
    size_t GetSizeBytes();
    /**
     * Returns the array of int16_t or NULL if the data type isn't int16_t
     */
    vector< int16_t >* GetBufferInt16();
    /**
     * Returns the array of int32_t or NULL if the data type isn't int32_t
     */
    vector< int32_t >* GetBufferInt32();
    /**
     * Returns the array of float or NULL if the data type isn't float
     */
    vector< float >* GetBufferFloat();
};

#endif
