#ifndef DOWNHILLSIMPLEXMETHOD_HPP_
#define DOWNHILLSIMPLEXMETHOD_HPP_

#include <cstddef> // size_t

#include <Eigen/Dense>

namespace cppmath
{
    /**
     * Implementation of Downhill Simplex or Nelder-Mead method for nonlinear optimization from 1998.
     * J. Lagarias, J. Reeds, M. Wright, P. Wright,
     * "Convergence Properties of the Nelder-Mead Simplex Method in Low Dimensions,"
     * SIAM Journal of Optimization, 1998, 9, 112-147
     *
     * \author cpieloth
     * \copyright Copyright 2014 Christof Pieloth, Licensed under the Apache License, Version 2.0
     */
    template< size_t DIM >
    class DownhillSimplexMethod
    {
    public:
        typedef Eigen::Matrix< double, DIM, 1 > ParamsT; /**< Abbreviation for a vector of parameters. */

        /**
         * Enum to indicate how the optimization was converged.
         */
        enum Converged
        {
            CONVERGED_NO, /**< Optimization was not started. */
            CONVERGED_EPSILON, /**< Optimization is smaller than epsilon/threshold. */
            CONVERGED_ITERATIONS, /**< Maximum iterations was reached. */
            CONVERGED_YES /**< Optimization is converged, but not specified how. */
        };

        DownhillSimplexMethod();
        virtual ~DownhillSimplexMethod();

        /**
         * Implementation of the function to minimize.
         *
         * \param x  n-dimensional parameter vector.
         * \return function value for vector x.
         */
        virtual double func( const ParamsT& x ) const = 0;

        double getReflectionCoeff() const;

        void setReflectionCoeff( double coeff );

        double getContractionCoeff() const;

        void setContractionCoeff( double coeff );

        double getExpansionCoeff() const;

        void setExpansionCoeff( double coeff );

        double getShrinkageCoeff() const;

        void setShrinkageCoeff( double coeff );

        size_t getMaximumIterations() const;

        void setMaximumIterations( size_t iterations );

        double getEpsilon() const;

        void setEpsilon( double eps );

        double getInitialFactor() const;

        void setInitialFactor( double factor );

        size_t getResultIterations() const;

        ParamsT getResultParams() const;

        double getResultError() const;

        /**
         * Starts the optimization.
         *
         * \param initial Initial start point.
         * \return Enum::Converged
         */
        Converged optimize( const ParamsT& initial );

    protected:
        /**
         * Orders the parameters x and f(x) from min to max.
         */
        virtual void order();

        /**
         * Checks if the optimization has been converged.
         *
         * \return Enum::Converged
         */
        virtual Converged converged() const;

        /**
         * Creates the initial parameter set and their function values.
         *
         * \param initial Start parameter used to calculate initials.
         */
        virtual void createInitials( const ParamsT& initial );

        double m_initFactor; /**< Factor to create the initial parameter set. */

        ParamsT m_x[DIM + 1]; /**< Vector of all n+1 points. */
        double m_f[DIM + 1]; /**< Stores the function values to reduce re-calculation. */

        double m_epsilon; /**< Threshold or deviation for convergence. */
        size_t m_maxIterations; /**< Maximum iterations until the algorithm is canceled. */
        size_t m_iterations; /**< Iteration counter used for break condition. */

        const size_t DIMENSION; /**< Constant for dimension. */
        const size_t VALUES; /**< Constant for DIM+1. */

    private:
        enum Step
        {
            STEP_START, STEP_EXIT, STEP_REFLECTION, STEP_EXPANSION, STEP_CONTRACTION, STEP_SHRINKAGE
        };

        const size_t N; /**< Index n=DIM-1 */
        const size_t N1; /**< Index n+1=DIM */

        double m_refl; /**< Reflection coefficient. */
        double m_contr; /**< Contraction coefficient. */
        double m_exp; /**< Expansion coefficient. */
        double m_shri; /**< Shrinkage coefficient. */

        void centroid();
        void accept( const ParamsT& x, double f );

        Step reflection();
        Step expansion();
        Step contraction();
        Step shrinkage();

        ParamsT m_xo;
        ParamsT m_xr;
        double m_fr;
    };
} /* namespace cppmath */

// Load the implementation
#include "DownhillSimplexMethod-impl.hpp"

#endif  // DOWNHILLSIMPLEXMETHOD_HPP_
