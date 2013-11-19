/*
 * WGenericCircularBuffer.h
 *
 *  Created on: 12.11.2013
 *      Author: maschke
 */

#ifndef WGENERICCIRCULARBUFFER_H_
#define WGENERICCIRCULARBUFFER_H_

#include <boost/circular_buffer.hpp>

template< class T >
class WGenericCircularBuffer
{
public:
    WGenericCircularBuffer(size_t);

private:
    boost::circular_buffer< T > buffer;
};

template< class T >
WGenericCircularBuffer<T>::WGenericCircularBuffer(size_t size) : buffer(size)
{

}

#endif /* WGENERICCIRCULARBUFFER_H_ */
