#include <float.h>
#include "LFMeasurementInfo.h"

LFMeasurementInfo::LFMeasurementInfo() :
    m_NumberOfChannels( -1 ), m_SamplingFrequency( -FLT_MAX ), m_Lowpass( -FLT_MAX ), m_Highpass( -FLT_MAX ), m_DataPack( -1 ),
                    m_LineFreq( -FLT_MAX ), m_GantryAngle( 0 )
{

}

void LFMeasurementInfo::Init()
{
    m_LFSubject.Init();
    m_LFProject.Init();
    m_LFHPIMeasurement.Init();
    m_LFIsotrak.Init();
    m_LFSSP.Init();
    m_LFEvents.Init();
    m_LFDataAcquisitionParameters.Init();
    m_NumberOfChannels = -1;
    m_SamplingFrequency = -FLT_MAX;
    m_Lowpass = -FLT_MAX;
    m_Highpass = -FLT_MAX;
    m_DataPack = -1;
    m_LineFreq = -FLT_MAX;
    m_GantryAngle = 0;
    m_LFChannelInfo.clear();
    m_BadChannels.clear();
    m_LFHPISubsystem.Init();
}

LFSubject& LFMeasurementInfo::GetLFSubject()
{
    return m_LFSubject;
}

LFProject& LFMeasurementInfo::GetLFProject()
{
    return m_LFProject;
}

LFHPIMeasurement& LFMeasurementInfo::GetLFHPIMeasurement()
{
    return m_LFHPIMeasurement;
}

LFIsotrak& LFMeasurementInfo::GetLFIsotrak()
{
    return m_LFIsotrak;
}

LFSSP& LFMeasurementInfo::GetLFSSP()
{
    return m_LFSSP;
}

LFEvents& LFMeasurementInfo::GetLFEvents()
{
    return m_LFEvents;
}

LFDataAcquisitionParameters& LFMeasurementInfo::GetLFDataAcquisitionParameters()
{
    return m_LFDataAcquisitionParameters;
}

int32_t LFMeasurementInfo::GetNumberOfChannels()
{
    return m_NumberOfChannels;
}

float LFMeasurementInfo::GetSamplingFrequency()
{
    return m_SamplingFrequency;
}

float LFMeasurementInfo::GetLowpass()
{
    return m_Lowpass;
}

float LFMeasurementInfo::GetHighpass()
{
    return m_Highpass;
}

int32_t LFMeasurementInfo::GetDataPack()
{
    return m_DataPack;
}

float LFMeasurementInfo::GetLineFreq()
{
    return m_LineFreq;
}

int32_t LFMeasurementInfo::GetGantryAngle()
{
    return m_GantryAngle;
}

LFArrayPtr<LFChannelInfo>& LFMeasurementInfo::GetLFChannelInfo()
{
    return m_LFChannelInfo;
}

vector< int32_t >& LFMeasurementInfo::GetBadChannels()
{
    return m_BadChannels;
}

LFHPISubsystem& LFMeasurementInfo::GetLFHPISubsystem()
{
    return m_LFHPISubsystem;
}

void LFMeasurementInfo::SetNumberOfChannels( const int32_t src )
{
    m_NumberOfChannels = src;
}

void LFMeasurementInfo::SetSamplingFrequency( const float src )
{
    m_SamplingFrequency = src;
}

void LFMeasurementInfo::SetLowpass( const float src )
{
    m_Lowpass = src;
}

void LFMeasurementInfo::SetHighpass( const float src )
{
    m_Highpass = src;
}

void LFMeasurementInfo::SetDataPack( const int32_t src )
{
    m_DataPack = src;
}

void LFMeasurementInfo::SetLineFreq( const float src )
{
    m_LineFreq = src;
}

void LFMeasurementInfo::SetGantryAngle( const int32_t src )
{
    m_GantryAngle = src;
}

