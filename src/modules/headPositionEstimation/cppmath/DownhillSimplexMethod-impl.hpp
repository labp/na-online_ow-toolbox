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
                    N( DIM - 1 ), N1( DIM ), DIMENSION( DIM ), VALUES( DIM + 1 )
    {
        m_refl = 1.0;
        m_contr = 0.5;
        m_exp = 2.0;
        m_shri = 0.5;

        m_initFactor = 2.0;

        m_maxIterations = 200 * DIM;
        m_iterations = 0;
        m_epsilon = 1e-4;
        m_fr = 0.0;
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
        return m_refl;
    }

    template< size_t DIM >
    void DownhillSimplexMethod< DIM >::setReflectionCoeff( double coeff )
    {
        m_refl = coeff;
    }

    template< size_t DIM >
    double DownhillSimplexMethod< DIM >::getContractionCoeff() const
    {
        return m_contr;
    }

    template< size_t DIM >
    void DownhillSimplexMethod< DIM >::setContractionCoeff( double coeff )
    {
        m_contr = coeff;
    }

    template< size_t DIM >
    double DownhillSimplexMethod< DIM >::getExpansionCoeff() const
    {
        return m_exp;
    }

    template< size_t DIM >
    void DownhillSimplexMethod< DIM >::setExpansionCoeff( double coeff )
    {
        m_exp = coeff;
    }

    template< size_t DIM >
    double DownhillSimplexMethod< DIM >::getShrinkageCoeff() const
    {
        return m_shri;
    }

    template< size_t DIM >
    void DownhillSimplexMethod< DIM >::setShrinkageCoeff( double coeff )
    {
        m_shri = coeff;
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
        return m_f[0];
    }

    template< size_t DIM >
    void DownhillSimplexMethod< DIM >::createInitials( const ParamsT& initial )
    {
        const double zero_term_delta = 0.00025; // delta for zero elements of x

        m_x[0] = initial;
        typename ParamsT::Index dim = 0;
        for( size_t i = 1; i < VALUES; ++i )
        {
            ParamsT p = initial;
            if( p( dim ) != 0 )
            {
                p( dim ) = m_initFactor * p( dim );
            }
            else
            {
                p( dim ) = zero_term_delta;
            }
            m_x[i] = p;
            ++dim;
        }

        for( size_t i = 0; i < VALUES; ++i )
        {
            m_f[i] = func( m_x[i] );
        }
    }

    template< size_t DIM >
    typename DownhillSimplexMethod< DIM >::Converged DownhillSimplexMethod< DIM >::optimize( const ParamsT& initial )
    {
        assert( m_refl > 0.0 );
        assert( m_exp > 1.0 );
        assert( m_exp > m_refl );
        assert( 0 < m_contr && m_contr < 1 );
        assert( 0 < m_shri && m_shri < 1 );
        assert( m_initFactor > 0.0 );

        // Prepare optimization
        m_iterations = 0;
        createInitials( initial );

        Converged conv = CONVERGED_NO;
        Step next = STEP_START;
        while( next != STEP_EXIT )
        {
            switch( next )
            {
                case STEP_START:
                    order();
                    conv = converged();
                    if( conv != CONVERGED_NO )
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
                case STEP_SHRINKAGE:
                    next = shrinkage();
                    break;
                default:
                    std::cerr << "Undefined control flow!";
                    next = STEP_EXIT;
                    break;
            }
        }

        return conv;
    }

    template< size_t DIM >
    void DownhillSimplexMethod< DIM >::order()
    {
        // The ordering is used in reflection() and for min/max.
        // Insertionsort
        for( size_t i = 1; i < VALUES; ++i )
        {
            const ParamsT x_insert = m_x[i];
            const double y_insert = m_f[i];
            size_t j = i;
            while( j > 0 && m_f[j - 1] > y_insert )
            {
                m_x[j] = m_x[j - 1];
                m_f[j] = m_f[j - 1];
                --j;
            }
            m_x[j] = x_insert;
            m_f[j] = y_insert;
        }
    }

    template< size_t DIM >
    void DownhillSimplexMethod< DIM >::centroid()
    {
        ParamsT xo = ParamsT::Zero();
        for( size_t i = 0; i <= N; ++i )
        {
            xo += m_x[i];
        }
        m_xo = xo / DIM;
    }

    template< size_t DIM >
    void DownhillSimplexMethod< DIM >::accept( const ParamsT& x, double f )
    {
        m_x[N1] = x;
        m_f[N1] = f;
    }

    template< size_t DIM >
    typename DownhillSimplexMethod< DIM >::Step DownhillSimplexMethod< DIM >::reflection()
    {
        m_xr = m_xo + m_refl * ( m_xo - m_x[N1] );
        m_fr = func( m_xr );
        const double f_r = m_fr;
        const double f_1 = m_f[0];
        const double f_n = m_f[N];

        if( f_1 <= f_r && f_r < f_n )
        {
            const ParamsT& x_r = m_xr;
            accept( x_r, f_r );
            return STEP_START;
        }

        if( f_r < f_1 )
        {
            return STEP_EXPANSION;
        }

        if( f_r >= f_n )
        {
            return STEP_CONTRACTION;
        }

        return STEP_EXIT;
    }

    template< size_t DIM >
    typename DownhillSimplexMethod< DIM >::Step DownhillSimplexMethod< DIM >::expansion()
    {
        const ParamsT x_e = m_xo + m_exp * ( m_xr - m_xo );
        const double f_e = func( x_e );
        const double f_r = m_fr;

        if( f_e < f_r )
        {
            accept( x_e, f_e );
        }
        else
        {
            const ParamsT& x_r = m_xr;
            accept( x_r, f_r );
        }
        return STEP_START;
    }

    template< size_t DIM >
    typename DownhillSimplexMethod< DIM >::Step DownhillSimplexMethod< DIM >::contraction()
    {
        const double f_r = m_fr;
        const double f_n = m_f[N];
        const double f_n1 = m_f[N1];

        // outside contraction
        if( f_n <= f_r && f_r < f_n1 )
        {
            const ParamsT& x_r = m_xr;
            const ParamsT x_c = m_xo + m_contr * ( x_r - m_xo );
            const double y_c = func( x_c );

            if( y_c <= f_r )
            {
                accept( x_c, y_c );
                return STEP_START;
            }
            else
            {
                return STEP_SHRINKAGE;
            }
        }

        // inside contraction
        if( f_r >= f_n1 )
        {
            const ParamsT& x_n1 = m_x[N1];
            const ParamsT x_cc = m_xo - m_contr * ( m_xo - x_n1 );
            const double fcc = func( x_cc );

            if( fcc <= f_r )
            {
                accept( x_cc, fcc );
                return STEP_START;
            }
            else
            {
                return STEP_SHRINKAGE;
            }
        }

        return STEP_EXIT;
    }

    template< size_t DIM >
    typename DownhillSimplexMethod< DIM >::Step DownhillSimplexMethod< DIM >::shrinkage()
    {
        const ParamsT& x_1 = m_x[0];
        for( size_t i = 1; i <= N1; ++i )
        {
            m_x[i] = x_1 + m_shri * ( m_x[i] - x_1 );
            m_f[i] = func( m_x[i] );
        }

        // TODO(cpieloth): nonshrink ordering rule, shrink ordering rule
        return STEP_START;
    }
} /* namespace cppmath */

#endif  // DOWNHILLSIMPLEXMETHOD_IMPL_HPP_
