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

#ifndef WPCA_H
#define WPCA_H

#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>
#include <Eigen/Core>

#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDPCA.h"

/**
 * \author jones
 */

class WPCA
{
public:
    typedef boost::shared_ptr< WPCA > SPtr;
    typedef boost::shared_ptr< const WPCA > ConstSPtr;

    typedef WLEMDPCA::MatrixT MatrixT;
    typedef WLEMDPCA::VectorT VectorT;

    static const std::string CLASS;

    explicit WPCA();
    explicit WPCA( int, bool );
    void setParams( int, bool );

    virtual ~WPCA();

    WLEMData::SPtr processData( WLEMData::SPtr );

private:
    Eigen::MatrixXd getMeanCentered( Eigen::MatrixXd );
    Eigen::MatrixXd getCovarianceMatrix( Eigen::MatrixXd );
    boost::shared_ptr< WLEMData::DataT > computePCA( WLEMData::DataT& );

    WLEMData::SPtr convertPCAToModality( WLEMDPCA::SPtr pcaIn );
    WLEMDPCA::SPtr createPCAContainer( WLEMData::SPtr emdIn,
                    boost::shared_ptr< WLEMDPCA::DataT > pcaData );

    boost::shared_ptr< std::vector< std::vector< double > > > eigenMatrixTo2DVector( Eigen::MatrixXd& );
    boost::shared_ptr< std::vector< double > > eigenMatrixTo1DVector( Eigen::MatrixXd& );

    boost::shared_ptr< MatrixT > m_transformationMatrix;
    boost::shared_ptr< VectorT > m_channelMeans;

    int m_num_dims;
    bool m_reverse;

};

#endif  // WPCA_H
