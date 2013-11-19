/*
 * WGenericList.h
 *
 *  Created on: 05.11.2013
 *      Author: maschke
 */

#ifndef WGENERICLIST_H_
#define WGENERICLIST_H_

#include <list>
#include <iterator>

using namespace std;

template< class T >
class WGenericList
{
public:

    /**
     * Add an element at the end of the list.
     * @param
     */
    void addElement( T const& );

    /**
     * Removes the Element from the given position.
     * @param i
     */
    void removeAt( size_t i );

    /**
     * Specified whether or not the list is empty.
     * @return false if the list is empty, else true.
     */
    bool IsEmpty() const;

    /**
     * Returns the whole list.
     * @return
     */
    const std::list< T >& getList() const;

    /**
     * Returns a pointer to the element of the certain position.
     * @param i
     * @return
     */
    T *getAt( size_t i );

    /**
     * This method changes the value at the certain position.
     * @param The position.
     * @param The value.
     */
    void setAt( size_t, T* );

protected:
    std::list< T > m_list;

};

template< class T >
void WGenericList< T >::addElement( T const& element )
{
    m_list.push_back( element );
}

template< class T >
void WGenericList< T >::removeAt( size_t i )
{
    if( this->IsEmpty() )
        return;

    if( i >= m_list.size() )
        return;

    typename list< T >::iterator it = m_list.begin();
    advance( it, i );

    m_list.erase( it );
}

template< class T >
bool WGenericList< T >::IsEmpty() const
{
    return m_list.empty();
}

template< class T >
const std::list< T >& WGenericList< T >::getList() const
{
    return m_list;
}

template< class T >
T *WGenericList< T >::getAt( size_t i )
{
    if( this->IsEmpty() )
        return NULL;

    if( i >= m_list.size() )
        return NULL;

    typename list< T >::iterator it = m_list.begin();
    advance( it, i );

    return &( *it );
}

template< class T >
void WGenericList< T >::setAt( size_t i, T *value )
{
    if( this->IsEmpty() )
        return;

    if( i >= m_list.size() )
        return;

    typename list< T >::iterator it = m_list.begin();
    advance( it, i );

    *it = *value;
}

#endif /* WGENERICLIST_H_ */
