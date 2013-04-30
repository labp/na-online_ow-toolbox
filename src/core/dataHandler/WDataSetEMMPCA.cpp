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

#include "core/data/emd/WLEMD.h"
#include "WDataSetEMMPCA.h"
#include "WDataSetEMMEnumTypes.h"

LaBP::WDataSetEMMPCA::WDataSetEMMPCA() :
                WLEMD()
{
    m_chanNames.reset( new std::vector< std::string >() );
}

LaBP::WDataSetEMMPCA::WDataSetEMMPCA( const WDataSetEMMPCA& pca ) :
                WLEMD( pca )
{
}

LaBP::WDataSetEMMPCA::WDataSetEMMPCA( const WLEMD& emd ) :
                WLEMD( emd )
{
    // C++11 supports "delegating constructors". So default initialization could be moved to default constructor.
    m_chanNames.reset( new std::vector< std::string >() );
}

LaBP::WDataSetEMMPCA::~WDataSetEMMPCA()
{
}

LaBP::WLEMD::SPtr LaBP::WDataSetEMMPCA::clone() const
{
    LaBP::WDataSetEMMPCA::SPtr pca( new LaBP::WDataSetEMMPCA( *this ) );
    return pca;
}

LaBP::WEModalityType::Enum LaBP::WDataSetEMMPCA::getModalityType() const
{
    return LaBP::WEModalityType::PCA;
}

void LaBP::WDataSetEMMPCA::setTransformationMatrix( boost::shared_ptr< MatrixT > new_trans )
{
    m_transformation_matrix = new_trans;
}

LaBP::WDataSetEMMPCA::MatrixT& LaBP::WDataSetEMMPCA::getTransformationMatrix()
{
    return *m_transformation_matrix;
}

void LaBP::WDataSetEMMPCA::setChannelMeans( boost::shared_ptr< VectorT > new_chan_means )
{
    m_channel_means = new_chan_means;
}

LaBP::WDataSetEMMPCA::VectorT& LaBP::WDataSetEMMPCA::getChannelMeans()
{
    return *m_channel_means;
}

void LaBP::WDataSetEMMPCA::setPreprocessedData( WLEMD::SPtr new_preprocessed_data )
{
    m_preprocessed_data = new_preprocessed_data;
}

LaBP::WLEMD::SPtr LaBP::WDataSetEMMPCA::getPreprocessedData()
{
    return m_preprocessed_data;
}
