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

#include <string>
#include <vector>

#include <core/common/math/linearAlgebra/WVectorFixed.h>

#include "core/data/WLEMMEnumTypes.h"

#include "WLEMData.h"
#include "WLEMDPCA.h"

WLEMDPCA::WLEMDPCA() :
                WLEMData()
{
}

WLEMDPCA::WLEMDPCA( const WLEMDPCA& pca ) :
                WLEMData( pca )
{
}

WLEMDPCA::WLEMDPCA( const WLEMData& emd ) :
                WLEMData( emd )
{
}

WLEMDPCA::~WLEMDPCA()
{
}

WLEMData::SPtr WLEMDPCA::clone() const
{
    WLEMDPCA::SPtr pca( new WLEMDPCA( *this ) );
    return pca;
}

LaBP::WEModalityType::Enum WLEMDPCA::getModalityType() const
{
    return LaBP::WEModalityType::PCA;
}

void WLEMDPCA::setTransformationMatrix( boost::shared_ptr< MatrixT > new_trans )
{
    m_transformation_matrix = new_trans;
}

WLEMDPCA::MatrixT& WLEMDPCA::getTransformationMatrix()
{
    return *m_transformation_matrix;
}

void WLEMDPCA::setChannelMeans( boost::shared_ptr< VectorT > new_chan_means )
{
    m_channel_means = new_chan_means;
}

WLEMDPCA::VectorT& WLEMDPCA::getChannelMeans()
{
    return *m_channel_means;
}

void WLEMDPCA::setPreprocessedData( WLEMData::SPtr new_preprocessed_data )
{
    m_preprocessed_data = new_preprocessed_data;
}

WLEMData::SPtr WLEMDPCA::getPreprocessedData()
{
    return m_preprocessed_data;
}
