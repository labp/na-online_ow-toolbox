#include <float.h>
#include "LFHPICoil.h"

LFHPICoil::LFHPICoil() :
    m_CoilNumber( -1 ), m_CoilFrequency( -FLT_MAX )
{

}

void LFHPICoil::Init()
{
    m_CoilNumber = -1;
    m_Epoch.clear();
    m_Slopes.clear();
    m_CorrelationCoefficient.clear();
    m_CoilFrequency = -FLT_MAX;
}

int32_t LFHPICoil::GetCoilNumber()
{
    return m_CoilNumber;
}

LFArrayFloat2d& LFHPICoil::GetEpoch()
{
    return m_Epoch;
}

vector<float>& LFHPICoil::GetSlopes()
{
    return m_Slopes;
}

vector<float>& LFHPICoil::GetCorrelationCoefficient()
{
    return m_CorrelationCoefficient;
}

float LFHPICoil::GetCoilFrequency()
{
    return m_CoilFrequency;
}

void LFHPICoil::SetCoilNumber( const int32_t src )
{
    m_CoilNumber = src;
}

void LFHPICoil::SetCoilFrequency( const float src )
{
    m_CoilFrequency = src;
}

