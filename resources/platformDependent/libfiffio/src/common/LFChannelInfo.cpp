#include<float.h>
#include<memory.h>
#include "LFChannelInfo.h"

LFChannelInfo::LFChannelInfo() :
    m_ScanNo( -1 ), m_LogNo( -1 ), m_Kind( -1 ), m_Range( -FLT_MAX ), m_Cal( -FLT_MAX ), m_CoilType( -1 ), m_Unit( unit_none ),
                    m_UnitMul( mul_none )
{
    memset( m_R0, 0, sizeof( m_R0 ) );
    memset( m_Ex, 0, sizeof( m_Ex ) );
    memset( m_Ey, 0, sizeof( m_Ey ) );
    memset( m_Ez, 0, sizeof( m_Ez ) );
}

void LFChannelInfo::Init()
{
    m_ScanNo = -1;
    m_LogNo = -1;
    m_Kind = -1;
    m_Range = -FLT_MAX;
    m_Cal = -FLT_MAX;
    m_CoilType = -1;
    memset( m_R0, 0, sizeof( m_R0 ) );
    memset( m_Ex, 0, sizeof( m_Ex ) );
    memset( m_Ey, 0, sizeof( m_Ey ) );
    memset( m_Ez, 0, sizeof( m_Ez ) );
    m_Unit = unit_none;
    m_UnitMul = mul_none;
}

int32_t LFChannelInfo::GetScanNo()
{
    return m_ScanNo;
}

int32_t LFChannelInfo::GetLogNo()
{
    return m_LogNo;
}

int32_t LFChannelInfo::GetKind()
{
    return m_Kind;
}

float LFChannelInfo::GetRange()
{
    return m_Range;
}

float LFChannelInfo::GetCal()
{
    return m_Cal;
}

int32_t LFChannelInfo::GetCoilType()
{
    return m_CoilType;
}

float* LFChannelInfo::GetR0()
{
    return m_R0;
}

float* LFChannelInfo::GetEx()
{
    return m_Ex;
}

float* LFChannelInfo::GetEy()
{
    return m_Ey;
}

float* LFChannelInfo::GetEz()
{
    return m_Ez;
}

fiffunits_t LFChannelInfo::GetUnit()
{
    return m_Unit;
}

fiffmultipliers_t LFChannelInfo::GetUnitMul()
{
    return m_UnitMul;
}

void LFChannelInfo::SetScanNo( const int32_t src )
{
    m_ScanNo = src;
}

void LFChannelInfo::SetLogNo( const int32_t src )
{
    m_LogNo = src;
}

void LFChannelInfo::SetKind( const int32_t src )
{
    m_Kind = src;
}

void LFChannelInfo::SetRange( const float src )
{
    m_Range = src;
}

void LFChannelInfo::SetCal( const float src )
{
    m_Cal = src;
}

void LFChannelInfo::SetCoilType( const int32_t src )
{
    m_CoilType = src;
}

void LFChannelInfo::SetR0( const float* src )
{
    memcpy( m_R0, src, sizeof( m_R0 ) );
}

void LFChannelInfo::SetEx( const float* src )
{
    memcpy( m_Ex, src, sizeof( m_Ex ) );
}

void LFChannelInfo::SetEy( const float* src )
{
    memcpy( m_Ey, src, sizeof( m_Ey ) );
}

void LFChannelInfo::SetEz( const float* src )
{
    memcpy( m_Ez, src, sizeof( m_Ez ) );
}

void LFChannelInfo::SetUnit( const fiffunits_t src )
{
    m_Unit = src;
}

void LFChannelInfo::SetUnitMul( const fiffmultipliers_t src )
{
    m_UnitMul = src;
}

