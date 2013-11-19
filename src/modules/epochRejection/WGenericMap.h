/*
 * WGenericMap.h
 *
 *  Created on: 08.11.2013
 *      Author: maschke
 */

#ifndef WGENERICMAP_H_
#define WGENERICMAP_H_

#include <map>

#include "WGenericList.h"

using namespace std;

template< class K, class V >
class WGenericMap
{
public:

    /**
     * Inserts a new element to the map collection.
     * @param The element key.
     * @param The element value.
     */
    void addElement( K const&, V const& );

    /**
     * Specified whether or not the map is empty.
     * @return false if the list is empty, else true.
     */
    bool IsEmpty() const;

    /**
     * Removes the element with the given key.
     * @param The key of the map entry.
     */
    void removeAt( const K& key )
    {
        if( IsEmpty() )
            return;

        if( m_map.count( key ) > 0 )
            m_map.erase( key );
    }

    /**
     * Returns the whole map.
     * @return The map.
     */
    const std::map< K, V >& getMap() const;

    /**
     * Returns the key-value-pair for the given key.
     * @param The key.
     * @return The key-value-pair.
     */
    V* getAt( const K& key )
    {
        if( m_map.count( key ) > 0 )
            return &m_map.at( key );

        return NULL;
    }

    /**
     * Sets the value for the given key.
     * @param The key.
     * @param The new value
     */
    void setValueAt( const K& key, const V& value )
    {
        if( m_map.count( key ) > 0 )
            m_map.at( key ) = value;
    }

protected:
    std::map< K, V > m_map;
};

template< class K, class V >
void WGenericMap< K, V >::addElement( K const& key, V const& value )
{
    m_map.insert( std::pair< K, V >( key, value ) );
}

template< class K, class V >
const std::map< K, V >& WGenericMap< K, V >::getMap() const
{
    return m_map;
}

template< class K, class V >
bool WGenericMap< K, V >::IsEmpty() const
{
    return m_map.empty();
}

#endif /* WGENERICMAP_H_ */
