#include <float.h>
#include "LFSubject.h"

LFSubject::LFSubject() :
    m_Birthday( -1 ), m_Sex( sex_unknown ), m_Hand( hand_unknown ), m_Weight( -FLT_MAX ), m_Height( -FLT_MAX )
{

}

void LFSubject::Init()
{
    m_HIS_ID.clear();
    m_LastName.clear();
    m_FirstName.clear();
    m_MiddleName.clear();
    m_Birthday = -1;
    m_Sex = sex_unknown;
    m_Hand = hand_unknown;
    m_Weight = -FLT_MAX;
    m_Height = -FLT_MAX;
    m_Comment.clear();
}

string& LFSubject::GetHIS_ID()
{
    return m_HIS_ID;
}

string& LFSubject::GetLastName()
{
    return m_LastName;
}

string& LFSubject::GetFirstName()
{
    return m_FirstName;
}

string& LFSubject::GetMiddleName()
{
    return m_MiddleName;
}

int32_t LFSubject::GetBirthday()
{
    return m_Birthday;
}

float LFSubject::GetWeight()
{
    return m_Weight;
}

LFSubject::sex_t LFSubject::GetSex()
{
    return m_Sex;
}

LFSubject::hand_t LFSubject::GetHand()
{
    return m_Hand;
}

float LFSubject::GetHeight()
{
    return m_Height;
}

string& LFSubject::GetComment()
{
    return m_Comment;
}

void LFSubject::SetHIS_ID( const char* src )
{
    m_HIS_ID = src;
}

void LFSubject::SetLastName( const char* src )
{
    m_LastName = src;
}

void LFSubject::SetFirstName( const char* src )
{
    m_FirstName = src;
}

void LFSubject::SetMiddleName( const char* src )
{
    m_MiddleName = src;
}

void LFSubject::SetBirthday( const int32_t src )
{
    m_Birthday = src;
}

void LFSubject::SetSex( const LFSubject::sex_t src )
{
    m_Sex = src;
}

void LFSubject::SetHand( const LFSubject::hand_t src )
{
    m_Hand = src;
}

void LFSubject::SetWeight( const float src )
{
    m_Weight = src;
}

void LFSubject::SetHeight( const float src )
{
    m_Height = src;
}

void LFSubject::SetComment( const char* src )
{
    m_Comment = src;
}

