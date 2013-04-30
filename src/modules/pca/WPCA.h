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

#ifndef WPCA_H
#define WPCA_H

#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>
#include <Eigen/Core>

#include "core/data/emd/WLEMD.h"
#include "core/data/emd/WLEMDPCA.h"

class WPCA
{
public:

    typedef LaBP::WLEMDPCA::MatrixT MatrixT;
    typedef LaBP::WLEMDPCA::VectorT VectorT;

    static const std::string CLASS;

    explicit WPCA();
    explicit WPCA( int, bool );
    void setParams( int, bool );

    virtual ~WPCA();

    boost::shared_ptr< LaBP::WLEMD > processData( LaBP::WLEMD::SPtr );

private:
    Eigen::MatrixXd getMeanCentered( Eigen::MatrixXd );
    Eigen::MatrixXd getCovarianceMatrix( Eigen::MatrixXd );
    boost::shared_ptr< LaBP::WLEMD::DataT > computePCA( LaBP::WLEMD::DataT& );

    LaBP::WLEMD::SPtr convertPCAToModality( LaBP::WLEMDPCA::SPtr pcaIn );
    LaBP::WLEMDPCA::SPtr createPCAContainer( LaBP::WLEMD::SPtr emdIn,
                    boost::shared_ptr< LaBP::WLEMDPCA::DataT > pcaData );

    boost::shared_ptr< std::vector< std::vector< double > > > eigenMatrixTo2DVector( Eigen::MatrixXd& );
    boost::shared_ptr< std::vector< double > > eigenMatrixTo1DVector( Eigen::MatrixXd& );

    boost::shared_ptr< MatrixT > m_transformationMatrix;
    boost::shared_ptr< VectorT > m_channelMeans;

    int m_num_dims;
    bool m_reverse;

};

#endif  // WPCA_H
