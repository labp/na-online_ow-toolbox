#ifndef LFEVENTS_H
#define LFEVENTS_H

#include <cstddef>
#include <vector>
#include <inttypes.h>
using namespace std;

/*
 * Events block
 */
class LFEvents
{
protected:
    vector< int32_t > m_EventChannels;/**< Event channel numbers (600)*/
    vector< int32_t > m_EventList;/**< List of events, 3 integers per event: [number of samples, before, after] (601)*/
public:
    /**
     * Sets all member variables to default
     */
    void Init();
    /**
     * Returns event channel numbers
     */
    vector< int32_t >& GetEventChannels();
    /**
     * Returns Event List Element number (int32_t[3])
     */
    size_t GetEventListSize();
    /**
     * Creates the Event List of nElements*3 floats
     */
    void AllocateEventList( size_t nElements );
    /**
     * Creates the Event List nBytes long
     */
    void AllocateEventListBytes( size_t nBytes );
    /**
     * Returns the Element Pointer, or NULL if don't exists
     */
    int32_t* GetEventListElement( size_t index=0 );
};

#endif
