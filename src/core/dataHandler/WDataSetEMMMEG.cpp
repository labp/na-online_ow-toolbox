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

#include <boost/shared_ptr.hpp>

#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>

#include "WDataSetEMMEMD.h"
#include "WDataSetEMMMEG.h"
#include "WDataSetEMMEnumTypes.h"

LaBP::WDataSetEMMMEG::WDataSetEMMMEG() :
                WDataSetEMMEMD()
{
    m_chanPos3d = boost::shared_ptr< std::vector< WPosition > >( new std::vector< WPosition >() );
    m_faces = boost::shared_ptr< std::vector< WVector3i > >( new std::vector< WVector3i >() );

    m_eX.reset( new std::vector< WVector3f >() );
    m_eY.reset( new std::vector< WVector3f >() );
    m_eZ.reset( new std::vector< WVector3f >() );
}

LaBP::WDataSetEMMMEG::WDataSetEMMMEG( const WDataSetEMMMEG& meg ) :
                WDataSetEMMEMD( meg )
{
    m_chanPos3d = meg.m_chanPos3d;
    m_faces = meg.m_faces;
    m_eX = meg.m_eX;
    m_eY = meg.m_eY;
    m_eZ = meg.m_eZ;
}

LaBP::WDataSetEMMMEG::~WDataSetEMMMEG()
{
}

LaBP::WDataSetEMMEMD::SPtr LaBP::WDataSetEMMMEG::clone() const
{
    LaBP::WDataSetEMMMEG::SPtr meg( new WDataSetEMMMEG( *this ) );
    return meg;
}

LaBP::WEModalityType::Enum LaBP::WDataSetEMMMEG::getModalityType() const
{
    return LaBP::WEModalityType::MEG;
}

boost::shared_ptr< std::vector< WPosition > > LaBP::WDataSetEMMMEG::getChannelPositions3d() const
{
    return m_chanPos3d;
}

void LaBP::WDataSetEMMMEG::setChannelPositions3d( boost::shared_ptr< std::vector< WPosition > > chanPos3d )
{
    this->m_chanPos3d = chanPos3d;
}

std::vector< WVector3i >& LaBP::WDataSetEMMMEG::getFaces() const
{
    return *m_faces;
}

void LaBP::WDataSetEMMMEG::setFaces( boost::shared_ptr< std::vector< WVector3i > > faces )
{
    m_faces = faces;
}

std::vector< WVector3f >& LaBP::WDataSetEMMMEG::getEx() const
{
    return *m_eX;
}

void LaBP::WDataSetEMMMEG::setEx( boost::shared_ptr< std::vector< WVector3f > > vec )
{
    m_eX = vec;
}

std::vector< WVector3f >& LaBP::WDataSetEMMMEG::getEy() const
{
    return *m_eY;
}

void LaBP::WDataSetEMMMEG::setEy( boost::shared_ptr< std::vector< WVector3f > > vec )
{
    m_eY = vec;
}

std::vector< WVector3f >& LaBP::WDataSetEMMMEG::getEz() const
{
    return *m_eZ;
}

void LaBP::WDataSetEMMMEG::setEz( boost::shared_ptr< std::vector< WVector3f > > vec )
{
    m_eZ = vec;
}
