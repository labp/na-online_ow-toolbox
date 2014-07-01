//---------------------------------------------------------------------------
//
// Project: NA-Online ( http://www.labp.htwk-leipzig.de )
//
// Copyright 2010 Laboratory for Biosignal Processing, HTWK Leipzig, Germany
//
// This file is part of NA-Online.
//
// NA-Online is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// NA-Online is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with NA-Online. If not, see <http://www.gnu.org/licenses/>.
//
//---------------------------------------------------------------------------

#ifndef WDOWNHILLSIMPLEXMETHOD_IMPL_HPP_
#define WDOWNHILLSIMPLEXMETHOD_IMPL_HPP_

#include <cassert>
#include <cmath> // pow()

#include <iostream> // std::cerr

#include "WDownhillSimplexMethod.hpp"

template< size_t DIM >
WDownhillSimplexMethod< DIM >::WDownhillSimplexMethod()
{
    m_alpha = 1.0;
    m_beta = 0.5;
    m_gamma = 2.0;
    m_maxIterations = 128;
    m_iterations = 0;
    m_epsilon = 1e-9;
    m_initialFactor = 2.0;
    m_x.resize( DIM + 1 );
}

template< size_t DIM >
WDownhillSimplexMethod< DIM >::~WDownhillSimplexMethod()
{
}

template< size_t DIM >
bool WDownhillSimplexMethod< DIM >::isConverged() const
{
    return ( m_iterations >= m_maxIterations ) || ( func( m_x[0] ) <= m_epsilon );
}

template< size_t DIM >
double WDownhillSimplexMethod< DIM >::getReflectionCoeff() const
{
    return m_alpha;
}

template< size_t DIM >
void WDownhillSimplexMethod< DIM >::setReflectionCoeff( double alpha )
{
    assert( 0.0 <= alpha );
    m_alpha = alpha;
}

template< size_t DIM >
double WDownhillSimplexMethod< DIM >::getContractionCoeff() const
{
    return m_beta;
}

template< size_t DIM >
void WDownhillSimplexMethod< DIM >::setContractionCoeff( double beta )
{
    assert( 0.0 <= beta && beta <= 1.0 );
    m_beta = beta;
}

template< size_t DIM >
double WDownhillSimplexMethod< DIM >::getExpansionCoeff() const
{
    return m_gamma;
}

template< size_t DIM >
void WDownhillSimplexMethod< DIM >::setExpansionCoeff( double gamma )
{
    m_gamma = gamma;
}

template< size_t DIM >
size_t WDownhillSimplexMethod< DIM >::getMaximumIterations() const
{
    return m_maxIterations;
}

template< size_t DIM >
void WDownhillSimplexMethod< DIM >::setMaximumIterations( size_t iterations )
{
    m_maxIterations = iterations;
}

template< size_t DIM >
size_t WDownhillSimplexMethod< DIM >::getIterations() const
{
    return m_iterations;
}

template< size_t DIM >
double WDownhillSimplexMethod< DIM >::getEpsilon() const
{
    return m_epsilon;
}

template< size_t DIM >
void WDownhillSimplexMethod< DIM >::setEpsilon( double eps )
{
    m_epsilon = eps;
}

template< size_t DIM >
double WDownhillSimplexMethod< DIM >::getInitialFactor() const
{
    return m_initialFactor;
}

template< size_t DIM >
void WDownhillSimplexMethod< DIM >::setInitialFactor( double factor )
{
    m_initialFactor = factor;
}

template< size_t DIM >
typename WDownhillSimplexMethod< DIM >::PointT WDownhillSimplexMethod< DIM >::getBestVariable() const
{
    return m_x[0];
}

template< size_t DIM >
void WDownhillSimplexMethod< DIM >::optimize( const PointT& initial )
{
    m_x[0] = initial;
    typename PointT::Index dim = 0;
    for( typename PointVector::size_type i = 1; i < m_x.size(); ++i )
    {
        PointT p = initial;
        p( dim ) = p( dim ) * m_initialFactor;
        m_x[i] = p;
        ++dim;
    }

    Step next = STEP_START;
    while( next != STEP_EXIT )
    {
        switch( next )
        {
            case STEP_START:
                order();
                if( isConverged() )
                {
                    next = STEP_EXIT;
                    break;
                }
                ++m_iterations;
                centroid();
                next = STEP_REFLECTION;
                break;
            case STEP_REFLECTION:
                next = reflection();
                break;
            case STEP_EXPANSION:
                next = expansion();
                break;
            case STEP_CONTRACTION:
                next = contraction();
                break;
            default:
                std::cerr << "Undefined control flow!";
                next = STEP_EXIT;
                break;
        }
    }
}

template< size_t DIM >
void WDownhillSimplexMethod< DIM >::order()
{
    // The ordering is used in reflection() and for min/max.
    // An alternative is to  store y-values and the indices of min/max.
    // Insertionsort
    for( typename PointVector::size_type i = 1; i < m_x.size(); ++i )
    {
        const PointT insert = m_x[i];
        const double fInsert = func( insert );
        typename PointVector::size_type j = i;
        while( j > 0 && func( m_x[j - 1] ) > fInsert )
        {
            m_x[j] = m_x[j - 1];
            --j;
        }
        m_x[j] = insert;
    }
}

template< size_t DIM >
void WDownhillSimplexMethod< DIM >::centroid()
{
    PointT xo = PointT::Zero();
    for( typename PointVector::size_type i = 0; i < DIM; ++i )
    {
        xo += m_x[i];
    }
    m_xo = xo / DIM;
}

template< size_t DIM >
typename WDownhillSimplexMethod< DIM >::Step WDownhillSimplexMethod< DIM >::reflection()
{
    m_xr = m_xo + m_alpha * ( m_xo - m_x[DIM] );
    const double yr = func( m_xr );
    const double yl = func( m_x[0] );

    if( yr < yl )
    {
        return STEP_EXPANSION;
    }
    // was sorted so
    const double yi = func( m_x[DIM - 1] );
    if( yr > yi )
    {
        const double yh = func( m_x[DIM] );
        if( yr <= yh )
        {
            m_x[DIM] = m_xr;
        }
        return STEP_CONTRACTION;
    }
    else
    {
        m_x[DIM] = m_xr;
        return STEP_START;
    }
}

template< size_t DIM >
typename WDownhillSimplexMethod< DIM >::Step WDownhillSimplexMethod< DIM >::expansion()
{
    const PointT xe = m_xo + m_gamma * ( m_xr - m_xo );
    const double ye = func( xe );
    const double yl = func( m_x[0] );

    if( ye < yl )
    {
        m_x[DIM] = xe;
    }
    else
    {
        m_x[DIM] = m_xr;

    }
    return STEP_START;
}

template< size_t DIM >
typename WDownhillSimplexMethod< DIM >::Step WDownhillSimplexMethod< DIM >::contraction()
{
    const PointT xc = m_xo + m_beta * ( m_x[DIM] - m_xo );
    const double yc = func( xc );
    const double yh = func( m_x[DIM] );

    if( yc > yh )
    {
        const PointT xl = m_x[0];
        for( typename PointVector::size_type i = 0; i < DIM + 1; ++i )
        {
            m_x[i] = 0.5 * ( m_x[i] + xl );
        }
    }
    else
    {
        m_x[DIM] = xc;
    }
    return STEP_START;
}

#endif  // WDOWNHILLSIMPLEXMETHOD_IMPL_HPP_
