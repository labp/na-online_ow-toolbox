#ifndef DOWNHILLSIMPLEXMETHOD_IMPL_HPP_
#define DOWNHILLSIMPLEXMETHOD_IMPL_HPP_

#include <cassert>
#include <cmath> // pow()

#include <iostream> // std::cerr

#include "DownhillSimplexMethod.hpp"

namespace cppmath
{
    template< size_t DIM >
    DownhillSimplexMethod< DIM >::DownhillSimplexMethod() :
                    DIMENSION( DIM ), VALUES( DIM + 1 )
    {
        m_alpha = 1.0;
        m_beta = 0.5;
        m_gamma = 2.0;
        m_maxIterations = 128;
        m_iterations = 0;
        m_epsilon = 1e-9;
        m_initFactor = 2.0;
        m_yr = 0.0;
    }

    template< size_t DIM >
    DownhillSimplexMethod< DIM >::~DownhillSimplexMethod()
    {
    }

    template< size_t DIM >
    typename DownhillSimplexMethod< DIM >::Converged DownhillSimplexMethod< DIM >::converged() const
    {
        if( m_iterations >= m_maxIterations )
        {
            return CONVERGED_ITERATIONS;
        }
        if( func( m_x[0] ) <= m_epsilon )
        {
            return CONVERGED_EPSILON;
        }
        return CONVERGED_NO;
    }

    template< size_t DIM >
    double DownhillSimplexMethod< DIM >::getReflectionCoeff() const
    {
        return m_alpha;
    }

    template< size_t DIM >
    void DownhillSimplexMethod< DIM >::setReflectionCoeff( double alpha )
    {
        assert( 0.0 <= alpha );
        m_alpha = alpha;
    }

    template< size_t DIM >
    double DownhillSimplexMethod< DIM >::getContractionCoeff() const
    {
        return m_beta;
    }

    template< size_t DIM >
    void DownhillSimplexMethod< DIM >::setContractionCoeff( double beta )
    {
        assert( 0.0 <= beta && beta <= 1.0 );
        m_beta = beta;
    }

    template< size_t DIM >
    double DownhillSimplexMethod< DIM >::getExpansionCoeff() const
    {
        return m_gamma;
    }

    template< size_t DIM >
    void DownhillSimplexMethod< DIM >::setExpansionCoeff( double gamma )
    {
        m_gamma = gamma;
    }

    template< size_t DIM >
    size_t DownhillSimplexMethod< DIM >::getMaximumIterations() const
    {
        return m_maxIterations;
    }

    template< size_t DIM >
    void DownhillSimplexMethod< DIM >::setMaximumIterations( size_t iterations )
    {
        m_maxIterations = iterations;
    }

    template< size_t DIM >
    double DownhillSimplexMethod< DIM >::getEpsilon() const
    {
        return m_epsilon;
    }

    template< size_t DIM >
    void DownhillSimplexMethod< DIM >::setEpsilon( double eps )
    {
        m_epsilon = eps;
    }

    template< size_t DIM >
    double DownhillSimplexMethod< DIM >::getInitialFactor() const
    {
        return m_initFactor;
    }

    template< size_t DIM >
    void DownhillSimplexMethod< DIM >::setInitialFactor( double factor )
    {
        m_initFactor = factor;
    }

    template< size_t DIM >
    size_t DownhillSimplexMethod< DIM >::getResultIterations() const
    {
        return m_iterations;
    }

    template< size_t DIM >
    typename DownhillSimplexMethod< DIM >::ParamsT DownhillSimplexMethod< DIM >::getResultParams() const
    {
        return m_x[0];
    }

    template< size_t DIM >
    double DownhillSimplexMethod< DIM >::getResultError() const
    {
        return m_y[0];
    }

    template< size_t DIM >
    void DownhillSimplexMethod< DIM >::createInitials( const ParamsT& initial )
    {
        m_x[0] = initial;
        typename ParamsT::Index dim = 0;
        for( size_t i = 1; i < VALUES; ++i )
        {
            ParamsT p = initial;
            p( dim ) = p( dim ) * m_initFactor;
            m_x[i] = p;
            ++dim;
        }

        for( size_t i = 0; i < VALUES; ++i )
        {
            m_y[i] = func( m_x[i] );
        }
    }

    template< size_t DIM >
    void DownhillSimplexMethod< DIM >::optimize( const ParamsT& initial )
    {
        // Prepare optimization
        m_iterations = 0;
        createInitials( initial );

        Step next = STEP_START;
        while( next != STEP_EXIT )
        {
            switch( next )
            {
                case STEP_START:
                    order();
                    if( converged() != CONVERGED_NO )
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
    void DownhillSimplexMethod< DIM >::order()
    {
        // The ordering is used in reflection() and for min/max.
        // Insertionsort
        for( size_t i = 1; i < VALUES; ++i )
        {
            const ParamsT x_insert = m_x[i];
            const double y_insert = m_y[i];
            size_t j = i;
            while( j > 0 && m_y[j - 1] > y_insert )
            {
                m_x[j] = m_x[j - 1];
                m_y[j] = m_y[j - 1];
                --j;
            }
            m_x[j] = x_insert;
            m_y[j] = y_insert;
        }
    }

    template< size_t DIM >
    void DownhillSimplexMethod< DIM >::centroid()
    {
        ParamsT xo = ParamsT::Zero();
        for( size_t i = 0; i < DIM; ++i )
        {
            xo += m_x[i];
        }
        m_xo = xo / DIM;
    }

    template< size_t DIM >
    typename DownhillSimplexMethod< DIM >::Step DownhillSimplexMethod< DIM >::reflection()
    {
        m_xr = m_xo + m_alpha * ( m_xo - m_x[DIM] );
        m_yr = func( m_xr );
        const double yr = m_yr;
        const double yl = m_y[0];

        if( yr < yl )
        {
            return STEP_EXPANSION;
        }
        // was sorted so
        const double yi = m_y[DIM - 1];
        if( yr > yi )
        {
            const double yh = m_y[DIM];
            if( yr <= yh )
            {
                m_x[DIM] = m_xr;
                m_y[DIM] = yr;
            }
            return STEP_CONTRACTION;
        }
        else
        {
            m_x[DIM] = m_xr;
            m_y[DIM] = yr;
            return STEP_START;
        }
    }

    template< size_t DIM >
    typename DownhillSimplexMethod< DIM >::Step DownhillSimplexMethod< DIM >::expansion()
    {
        const ParamsT xe = m_xo + m_gamma * ( m_xr - m_xo );
        const double ye = func( xe );
        const double yl = m_y[0];

        if( ye < yl )
        {
            m_x[DIM] = xe;
            m_y[DIM] = ye;
        }
        else
        {
            m_x[DIM] = m_xr;
            m_y[DIM] = m_yr;
        }
        return STEP_START;
    }

    template< size_t DIM >
    typename DownhillSimplexMethod< DIM >::Step DownhillSimplexMethod< DIM >::contraction()
    {
        const ParamsT xc = m_xo + m_beta * ( m_x[DIM] - m_xo );
        const double yc = func( xc );
        const double yh = m_y[DIM];

        if( yc > yh )
        {
            const ParamsT xl = m_x[0];
            for( size_t i = 0; i < DIM + 1; ++i )
            {
                m_x[i] = 0.5 * ( m_x[i] + xl );
                m_y[i] = func( m_x[i] );
            }
        }
        else
        {
            m_x[DIM] = xc;
            m_y[DIM] = yc;
        }
        return STEP_START;
    }
} /* namespace cppmath */

#endif  // DOWNHILLSIMPLEXMETHOD_IMPL_HPP_
