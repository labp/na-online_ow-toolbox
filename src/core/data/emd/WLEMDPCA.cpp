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

#include <string>
#include <vector>

#include <core/common/math/linearAlgebra/WVectorFixed.h>

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

WLEModality::Enum WLEMDPCA::getModalityType() const
{
    return WLEModality::PCA;
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
