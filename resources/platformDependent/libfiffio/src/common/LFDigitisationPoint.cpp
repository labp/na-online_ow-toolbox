#include "LFDigitisationPoint.h"

LFDigitisationPoint::LFDigitisationPoint() :
    m_Kind( -1 ), m_Ident( -1 )
{

}

void LFDigitisationPoint::Init()
{
    m_Kind = -1;
    m_Ident = -1;
    m_Rr.clear();
}

int32_t LFDigitisationPoint::GetKind()
{
    return m_Kind;
}

int32_t LFDigitisationPoint::GetIdent()
{
    return m_Ident;
}

LFArrayFloat3d& LFDigitisationPoint::GetRr()
{
    return m_Rr;
}

void LFDigitisationPoint::SetKind( const int32_t src )
{
    m_Kind = src;
}

void LFDigitisationPoint::SetIdent( const int32_t src )
{
    m_Ident = src;
}

