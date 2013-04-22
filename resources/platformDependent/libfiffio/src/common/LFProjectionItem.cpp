#include <float.h>
#include "LFProjectionItem.h"

LFProjectionItem::LFProjectionItem() :
    m_Kind( pi_none ), m_NumberOfChannels( -1 ), m_Time( -FLT_MAX )
{

}

void LFProjectionItem::Init()
{
    m_Description.clear();
    m_Kind = pi_none;
    m_NumberOfChannels = -1;
    m_ChannelNameList.clear();
    m_Time = -FLT_MAX;
    m_ProjectionVectors.clear();
}

string& LFProjectionItem::GetDescription()
{
    return m_Description;
}

LFProjectionItem::proj_item_t LFProjectionItem::GetKind()
{
    return m_Kind;
}

int32_t LFProjectionItem::GetNumberOfChannels()
{
    return m_NumberOfChannels;
}

string& LFProjectionItem::GetChannelNameList()
{
    return m_ChannelNameList;
}

float LFProjectionItem::GetTime()
{
    return m_Time;
}

int32_t LFProjectionItem::GetNumberOfProjectionVectors()
{
    return m_ProjectionVectors.GetSize();
}

LFArrayFloat2d& LFProjectionItem::GetProjectionVectors()
{
    return m_ProjectionVectors;
}

void LFProjectionItem::SetDescription( const char* src )
{
    m_Description = src;
}

void LFProjectionItem::SetKind( const proj_item_t src )
{
    m_Kind = src;
}

void LFProjectionItem::SetNumberOfChannels( const int32_t src )
{
    m_NumberOfChannels = src;
}

void LFProjectionItem::SetChannelNameList( const char* src )
{
    m_ChannelNameList = src;
}

void LFProjectionItem::SetTime( const float src )
{
    m_Time = src;
}

