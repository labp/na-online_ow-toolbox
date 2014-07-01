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

#ifndef WDOWNHILLSIMPLEXMETHOD_HPP_
#define WDOWNHILLSIMPLEXMETHOD_HPP_

#include <cstddef> // size_t
#include <vector>

#include <Eigen/Dense>

/**
 * Implementation of the Downhill Simplex or Nelder-Mead method for nonlinear optimization.
 * Nelder, John Ashworth; Mead, Roger: A Simplex Method for Function Minimization. Computer Journal, 1965, 7, 308-313
 *
 * \author cpieloth
 * \copyright Copyright 2014 Christof Pieloth, Licensed under the Apache License, Version 2.0
 */
template< size_t DIM >
class WDownhillSimplexMethod
{
public:
    WDownhillSimplexMethod();
    virtual ~WDownhillSimplexMethod();

    typedef Eigen::Matrix< double, 1, DIM > PointT;

    /**
     * Implementation of the function to minimize.
     *
     * \param x  n-dimensional point
     * \return function value for point x.
     */
    virtual double func( const PointT& x ) const = 0;

    /**
     * Checks if the optimization is converged.
     *
     * \return True, if optimization is converged.
     */
    virtual bool isConverged() const;

    double getReflectionCoeff() const;

    void setReflectionCoeff( double alpha );

    double getContractionCoeff() const;

    void setContractionCoeff( double beta );

    double getExpansionCoeff() const;

    void setExpansionCoeff( double gamma );

    size_t getMaximumIterations() const;

    void setMaximumIterations( size_t iterations );

    double getEpsilon() const;

    void setEpsilon( double eps );

    double getInitialFactor() const;

    void setInitialFactor( double factor );

    size_t getIterations() const;

    PointT getBestVariable() const;

    /**
     * Starts the optimization.
     *
     * \param initial Initial start point.
     */
    void optimize( const PointT& initial );

protected:
    typedef std::vector< PointT > PointVector;
    PointVector m_x; /**< Vector of all n+1 points. */

    double m_epsilon; /**< Threshold or deviation for convergence. */

    size_t m_maxIterations; /**< Maximum iterations until the algorithm is canceled. */
    size_t m_iterations; /**< Iteration counter used for break condition. */

private:
    enum Step
    {
        STEP_START, STEP_EXIT, STEP_REFLECTION, STEP_EXPANSION, STEP_CONTRACTION
    };

    double m_alpha; /**< Reflection coefficient. */
    double m_beta; /**< Contraction coefficient. */
    double m_gamma; /**< Expansion coefficient. */
    double m_initialFactor; /**< Factor to create the initial point set. */

    void order();
    void centroid();
    Step reflection();
    Step expansion();
    Step contraction();

    PointT m_xo;
    PointT m_xr;
};

// Load the implementation
#include "WDownhillSimplexMethod-impl.hpp"

#endif  // WDOWNHILLSIMPLEXMETHOD_HPP_
