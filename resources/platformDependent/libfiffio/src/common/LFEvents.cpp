#include <cstddef>
#include "LFEvents.h"

void LFEvents::Init()
{
    m_EventChannels.clear();
    m_EventList.clear();
}

vector< int32_t >& LFEvents::GetEventChannels()
{
    return m_EventChannels;
}

size_t LFEvents::GetEventListSize()
{
    return m_EventList.size() / 3;
}

void LFEvents::AllocateEventList( size_t nElements )
{
    m_EventList.resize( nElements * 3 );
}

void LFEvents::AllocateEventListBytes( size_t nBytes )
{
    m_EventList.resize( nBytes / sizeof(int32_t) );
}

int32_t* LFEvents::GetEventListElement( size_t index )
{
    size_t n = index * 3;
    return n < m_EventList.size() ? &( m_EventList.operator[]( n ) ) : NULL;
}

