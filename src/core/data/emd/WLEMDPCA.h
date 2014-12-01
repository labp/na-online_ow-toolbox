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

#ifndef WLEMDPCA_H
#define WLEMDPCA_H

#include <boost/shared_ptr.hpp>
#include <Eigen/Core>

#include <core/common/math/linearAlgebra/WVectorFixed.h>

#include "WLEMData.h"

/**
 * Data of a principle component analysis.
 *
 * \author jones
 * \ingroup data
 */
class WLEMDPCA: public WLEMData
{
public:
    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WLEMDPCA > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WLEMDPCA > ConstSPtr;

    typedef Eigen::MatrixXd MatrixT;

    typedef Eigen::VectorXd VectorT;

    WLEMDPCA();

    explicit WLEMDPCA( const WLEMDPCA& pca );

    explicit WLEMDPCA( const WLEMData& emd );

    virtual ~WLEMDPCA();

    virtual WLEMData::SPtr clone() const;

    virtual WLEModality::Enum getModalityType() const;

    void setTransformationMatrix( boost::shared_ptr< MatrixT > );
    MatrixT& getTransformationMatrix();
    void setChannelMeans( boost::shared_ptr< VectorT > );
    VectorT& getChannelMeans();
    void setPreprocessedData( WLEMData::SPtr new_preprocessed_data );
    WLEMData::SPtr getPreprocessedData();

private:
    boost::shared_ptr< MatrixT > m_transformation_matrix;

    boost::shared_ptr< VectorT > m_channel_means;

    WLEMData::SPtr m_preprocessed_data;
};
#endif  // WLEMDPCA_H
