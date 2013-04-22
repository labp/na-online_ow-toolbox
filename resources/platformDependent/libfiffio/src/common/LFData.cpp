#include "LFData.h"

LFData::LFData()
{

}

LFData::~LFData()
{

}

void LFData::Init()
{
    m_LFMeasurement.Init();
}

LFMeasurement& LFData::GetLFMeasurement()
{
    return m_LFMeasurement;
}
