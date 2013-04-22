#include "LFHPISubsystem.h"

LFHPISubsystem::LFHPISubsystem() :
    m_NumberOfHPICoils( -1 )
{

}

void LFHPISubsystem::Init()
{
    m_NumberOfHPICoils = -1;
    m_EventChannel.clear();
}

int32_t LFHPISubsystem::GetNumberOfHPICoils()
{
    return m_NumberOfHPICoils;
}

string& LFHPISubsystem::GetEventChannel()
{
    return m_EventChannel;
}

LFArrayPtr<LFHPICoil>& LFHPISubsystem::GetLFHPICoil()
{
    return m_LFHPICoil;
}

void LFHPISubsystem::SetNumberOfHPICoils( const int32_t src )
{
    m_NumberOfHPICoils = src;
}
