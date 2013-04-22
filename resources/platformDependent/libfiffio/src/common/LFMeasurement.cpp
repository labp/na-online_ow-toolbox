#include "LFMeasurement.h"

LFMeasurement::LFMeasurement()
{

}

LFMeasurement::~LFMeasurement()
{

}

void LFMeasurement::Init()
{
    m_LFMeasurementInfo.Init();
    m_LFRawData.Init();
    m_LFBem.Init();
}

LFMeasurementInfo& LFMeasurement::GetLFMeasurementInfo()
{
    return m_LFMeasurementInfo;
}

LFRawData& LFMeasurement::GetLFRawData()
{
    return m_LFRawData;
}

LFBem& LFMeasurement::GetLFBem()
{
    return m_LFBem;
}
