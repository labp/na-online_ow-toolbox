//---------------------------------------------------------------------------
//
// Project: OpenWalnut ( http://www.openwalnut.org )
//
// Copyright 2009 OpenWalnut Community, BSV@Uni-Leipzig and CNCF@MPI-CBS
// For more information see http://www.openwalnut.org/copying
//
// This file is part of OpenWalnut.
//
// OpenWalnut is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// OpenWalnut is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with OpenWalnut. If not, see <http://www.gnu.org/licenses/>.
//
//---------------------------------------------------------------------------

#ifndef WLEMDPCA_H
#define WLEMDPCA_H

#include <boost/shared_ptr.hpp>
#include <Eigen/Core>

#include <core/common/math/linearAlgebra/WVectorFixed.h>

#include "core/data/WLEMMEnumTypes.h"

#include "WLEMData.h"

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

    virtual LaBP::WEModalityType::Enum getModalityType() const;

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
