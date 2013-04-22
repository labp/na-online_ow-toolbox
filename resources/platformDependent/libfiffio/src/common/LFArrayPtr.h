#ifndef LFARRAYPTR_H
#define LFARRAYPTR_H

#include <cstddef>
#include <vector>
using namespace std;

template< class T >
class LFArrayPtr: public vector< T* >
{
public:
    ~LFArrayPtr()
    {
        for( size_t i = 0; i < vector< T* >::size(); i++ )
            delete vector< T* >::operator[]( i );
    }
    void clear()
    {
        for( size_t i = 0; i < vector< T* >::size(); i++ )
            delete vector< T* >::operator[]( i );
        vector< T* >::clear();
    }
    T& AddNew()
    {
        T* p = new T;
        vector< T* >::push_back( p );
        return *p;
    }
    T* operator[]( size_t n )
    {
        return n < 0 || n >= vector< T* >::size() ? NULL : vector< T* >::operator[]( n );
    }
    size_t Attach(const T& src)
    {
        vector< T* >::push_back( &src );
        return vector< T* >::size()-1;
    }
    T* Detach( size_t n )
    {
        if(n < 0 || n >= vector< T* >::size())return NULL;
        T* p=vector< T* >::operator[]( n );
        vector< T* >::erase(n);
        return p;
    }
};

#endif
