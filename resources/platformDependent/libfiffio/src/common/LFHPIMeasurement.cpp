#include <float.h>
#include "LFHPIMeasurement.h"

LFHPIMeasurement::LFHPIMeasurement() :
    m_SamplingFrequency( -FLT_MAX ), m_NumberOfChannels( -1 ), m_NumberOfAverages( -1 ), m_NumberOfHPICoils( -1 ), m_FirstSample(
                    -1 ), m_LastSample( -1 )
{

}

void LFHPIMeasurement::Init()
{
    m_SamplingFrequency = -FLT_MAX;
    m_NumberOfChannels = -1;
    m_NumberOfAverages = -1;
    m_NumberOfHPICoils = -1;
    m_FirstSample = -1;
    m_LastSample = -1;
    m_LFHPICoil.clear();
}

float LFHPIMeasurement::GetSamplingFrequency()
{
    return m_SamplingFrequency;
}

int32_t LFHPIMeasurement::GetNumberOfChannels()
{
    return m_NumberOfChannels;
}

int32_t LFHPIMeasurement::GetNumberOfAverages()
{
    return m_NumberOfAverages;
}

int32_t LFHPIMeasurement::GetNumberOfHPICoils()
{
    return m_NumberOfHPICoils;
}

int32_t LFHPIMeasurement::GetFirstSample()
{
    return m_FirstSample;
}

int32_t LFHPIMeasurement::GetLastSample()
{
    return m_LastSample;
}

LFArrayPtr<LFHPICoil>& LFHPIMeasurement::GetLFHPICoil()
{
    return m_LFHPICoil;
}

void LFHPIMeasurement::SetSamplingFrequency( const float src )
{
    m_SamplingFrequency = src;
}

void LFHPIMeasurement::SetNumberOfChannels( const int32_t src )
{
    m_NumberOfChannels = src;
}

void LFHPIMeasurement::SetNumberOfAverages( const int32_t src )
{
    m_NumberOfAverages = src;
}

void LFHPIMeasurement::SetNumberOfHPICoils( const int32_t src )
{
    m_NumberOfHPICoils = src;
}

void LFHPIMeasurement::SetFirstSample( const int32_t src )
{
    m_FirstSample = src;
}

void LFHPIMeasurement::SetLastSample( const int32_t src )
{
    m_LastSample = src;
}
