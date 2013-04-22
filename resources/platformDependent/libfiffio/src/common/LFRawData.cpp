#include "LFRawData.h"

LFRawData::LFRawData() :
    m_FirstSample( -1 ), m_DataSkip( -1 ), m_DataSkipSamples( -1 )
{

}

LFRawData::~LFRawData()
{

}

void LFRawData::Init()
{
    m_FirstSample = -1;
    m_DataSkip = -1;
    m_DataSkipSamples = -1;
    m_LFDataBuffer.clear();
}

int32_t LFRawData::GetFirstSample()
{
    return m_FirstSample;
}

int32_t LFRawData::GetDataSkip()
{
    return m_DataSkip;
}

int32_t LFRawData::GetDataSkipSamples()
{
    return m_DataSkipSamples;
}

LFArrayPtr<LFDataBuffer>& LFRawData::GetLFDataBuffer()
{
    return m_LFDataBuffer;
}

void LFRawData::SetFirstSample( const int32_t src )
{
    m_FirstSample = src;
}

void LFRawData::SetDataSkip( const int32_t src )
{
    m_DataSkip = src;
}

void LFRawData::SetDataSkipSamples( const int32_t src )
{
    m_DataSkipSamples = src;
}

