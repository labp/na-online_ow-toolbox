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

#include <vector>

#include <boost/shared_ptr.hpp>

#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>

#include "core/data/WLEMMEnumTypes.h"

#include "WLEMD.h"
#include "WLEMDEEG.h"

LaBP::WLEMDEEG::WLEMDEEG() :
                WLEMD()
{
    m_chanPos3d = boost::shared_ptr< std::vector< WPosition > >( new std::vector< WPosition >() );
    m_faces = boost::shared_ptr< std::vector< WVector3i > >( new std::vector< WVector3i >() );
}

LaBP::WLEMDEEG::WLEMDEEG( const WLEMDEEG& eeg ) :
                WLEMD( eeg )
{
    m_chanPos3d = eeg.m_chanPos3d;
    m_faces = eeg.m_faces;
    m_polarityType = eeg.getPolarityType();
}

LaBP::WLEMDEEG::~WLEMDEEG()
{
}

LaBP::WLEMD::SPtr LaBP::WLEMDEEG::clone() const
{
    LaBP::WLEMDEEG::SPtr eeg( new LaBP::WLEMDEEG( *this ) );
    return eeg;
}

LaBP::WEPolarityType::Enum LaBP::WLEMDEEG::getPolarityType() const
{
    return m_polarityType;
}

LaBP::WEModalityType::Enum LaBP::WLEMDEEG::getModalityType() const
{
    return LaBP::WEModalityType::EEG;
}

boost::shared_ptr< std::vector< WPosition > > LaBP::WLEMDEEG::getChannelPositions3d() const
{
    return m_chanPos3d;
}

void LaBP::WLEMDEEG::setChannelPositions3d( boost::shared_ptr< std::vector< WPosition > > chanPos3d )
{
    m_chanPos3d = chanPos3d;
}

std::vector< WVector3i >& LaBP::WLEMDEEG::getFaces() const
{
    return *m_faces;
}

void LaBP::WLEMDEEG::setFaces( boost::shared_ptr< std::vector< WVector3i > > faces )
{
    m_faces = faces;
}
