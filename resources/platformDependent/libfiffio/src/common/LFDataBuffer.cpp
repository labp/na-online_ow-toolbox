#include <cstddef>
#include "LFDataBuffer.h"

LFDataBuffer::LFDataBuffer() :
    m_DataType( dt_unknown ), m_Bytes( 0 ), m_Buffer( NULL )
{

}

LFDataBuffer::~LFDataBuffer()
{
    if( m_Buffer != NULL )
    {
        switch( m_DataType )
        {
            case dt_int16:
                delete GetBufferInt16();
                break;
            case dt_int32:
                delete GetBufferInt32();
                break;
            case dt_float:
                delete GetBufferFloat();
                break;
            case dt_unknown:
                break;
        }
    }
}

void LFDataBuffer::Init()
{
    if( m_Buffer != NULL )
    {
        switch( m_DataType )
        {
            case dt_int16:
                delete GetBufferInt16();
                break;
            case dt_int32:
                delete GetBufferInt32();
                break;
            case dt_float:
                delete GetBufferFloat();
                break;
            case dt_unknown:
                break;
        }
    }
    m_Buffer = NULL;
    m_Bytes = 0;
    m_DataType = dt_unknown;
}

void LFDataBuffer::Allocate( LFDataBuffer::datatype_t type, size_t bytes )
{
    Init();
    switch( type )
    {
        case dt_int16:
        {
            m_Buffer = new vector< int16_t > ;
            ( ( vector< int16_t >* )m_Buffer )->resize( bytes / type );
            break;
        }
        case dt_int32:
        {
            m_Buffer = new vector< int32_t > ;
            ( ( vector< int32_t >* )m_Buffer )->resize( bytes / type );
            break;
        }
        case dt_float:
        {
            m_Buffer = new vector< float > ;
            ( ( vector< float >* )m_Buffer )->resize( bytes / type );
            break;
        }
        default:
            return;
    }
    m_Bytes = bytes;
    m_DataType = type;
}

LFDataBuffer::datatype_t LFDataBuffer::GetDataType()
{
    return m_DataType;
}

int LFDataBuffer::GetElementSize()
{
    switch(m_DataType){
        case dt_int16: return sizeof(int16_t);
        case dt_int32: return sizeof(int32_t);
        case dt_float:
            return sizeof(float);
        case dt_unknown:
            break;
    }
    return -1;
}

size_t LFDataBuffer::GetSize()
{
    return m_Bytes / GetElementSize();
}

size_t LFDataBuffer::GetSizeBytes()
{
    return m_Bytes;
}

vector< int16_t >* LFDataBuffer::GetBufferInt16()
{
    return m_DataType == dt_int16 ? ( vector< int16_t >* )m_Buffer : NULL;
}

vector< int32_t >* LFDataBuffer::GetBufferInt32()
{
    return m_DataType == dt_int32 ? ( vector< int32_t >* )m_Buffer : NULL;
}

vector< float >* LFDataBuffer::GetBufferFloat()
{
    return m_DataType == dt_float ? ( vector< float >* )m_Buffer : NULL;
}

