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

#include "WLEMD.h"
#include "WLEMDPCA.h"

LaBP::WLEMDPCA::WLEMDPCA() :
                WLEMD()
{
    m_chanNames.reset( new std::vector< std::string >() );
}

LaBP::WLEMDPCA::WLEMDPCA( const WLEMDPCA& pca ) :
                WLEMD( pca )
{
}

LaBP::WLEMDPCA::WLEMDPCA( const WLEMD& emd ) :
                WLEMD( emd )
{
    // C++11 supports "delegating constructors". So default initialization could be moved to default constructor.
    m_chanNames.reset( new std::vector< std::string >() );
}

LaBP::WLEMDPCA::~WLEMDPCA()
{
}

LaBP::WLEMD::SPtr LaBP::WLEMDPCA::clone() const
{
    LaBP::WLEMDPCA::SPtr pca( new LaBP::WLEMDPCA( *this ) );
    return pca;
}

LaBP::WEModalityType::Enum LaBP::WLEMDPCA::getModalityType() const
{
    return LaBP::WEModalityType::PCA;
}

void LaBP::WLEMDPCA::setTransformationMatrix( boost::shared_ptr< MatrixT > new_trans )
{
    m_transformation_matrix = new_trans;
}

LaBP::WLEMDPCA::MatrixT& LaBP::WLEMDPCA::getTransformationMatrix()
{
    return *m_transformation_matrix;
}

void LaBP::WLEMDPCA::setChannelMeans( boost::shared_ptr< VectorT > new_chan_means )
{
    m_channel_means = new_chan_means;
}

LaBP::WLEMDPCA::VectorT& LaBP::WLEMDPCA::getChannelMeans()
{
    return *m_channel_means;
}

void LaBP::WLEMDPCA::setPreprocessedData( WLEMD::SPtr new_preprocessed_data )
{
    m_preprocessed_data = new_preprocessed_data;
}

LaBP::WLEMD::SPtr LaBP::WLEMDPCA::getPreprocessedData()
{
    return m_preprocessed_data;
}
