#include "LFSSP.h"

LFSSP::LFSSP() :
    m_NumberOfChannels( -1 )
{

}

void LFSSP::Init()
{
    m_NumberOfChannels = -1;
    m_LFProjectionItem.clear();
}

int32_t LFSSP::GetNumberOfChannels()
{
    return m_NumberOfChannels;
}

LFArrayPtr<LFProjectionItem>& LFSSP::GetLFProjectionItem()
{
    return m_LFProjectionItem;
}

void LFSSP::SetNumberOfChannels( const int32_t src )
{
    m_NumberOfChannels = src;
}

