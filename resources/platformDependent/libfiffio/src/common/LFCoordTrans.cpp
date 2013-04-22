#include <memory.h>
#include "LFCoordTrans.h"

LFCoordTrans::LFCoordTrans() :
    m_From( -1 ), m_To( -1 )
{
    memset( m_Rot, 0, sizeof( m_Rot ) );
    memset( m_Move, 0, sizeof( m_Move ) );
    memset( m_InvRot, 0, sizeof( m_InvRot ) );
    memset( m_InvMove, 0, sizeof( m_InvMove ) );
}

void LFCoordTrans::Init()
{
    m_From = -1;
    m_To = -1;
    memset( m_Rot, 0, sizeof(m_Rot) );
    memset( m_Move, 0, sizeof(m_Move) );
    memset( m_InvRot, 0, sizeof(m_InvRot) );
    memset( m_InvMove, 0, sizeof(m_InvMove) );
}

int32_t LFCoordTrans::GetFrom()
{
    return m_From;
}

int32_t LFCoordTrans::GetTo()
{
    return m_To;
}

float (&LFCoordTrans::GetRot())[3][3]
{
    return m_Rot;
}

float* LFCoordTrans::GetMove()
{
    return m_Move;
}

float (&LFCoordTrans::GetInvRot())[3][3]
{
    return m_InvRot;
}

float* LFCoordTrans::GetInvMove()
{
    return m_InvMove;
}

void LFCoordTrans::SetFrom( const int32_t src )
{
    m_From = src;
}

void LFCoordTrans::SetTo( const int32_t src )
{
    m_To = src;
}

void LFCoordTrans::SetRot( const float src[3][3] )
{
    memcpy( m_Rot, src, sizeof( m_Rot ) );
}

void LFCoordTrans::SetMove( const float* src )
{
    memcpy( m_Move, src, sizeof( m_Rot ) );
}

void LFCoordTrans::SetInvRot( const float src[3][3] )
{
    memcpy( m_InvRot, src, sizeof( m_Rot ) );
}

void LFCoordTrans::SetInvMove( const float* src )
{
    memcpy( m_InvMove, src, sizeof( m_Rot ) );
}

