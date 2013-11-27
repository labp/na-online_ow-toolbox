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

#include "WLEMData.h"
#include "WLEMDEEG.h"

WLEMDEEG::WLEMDEEG() :
                WLEMData()
{
    m_chanPos3d = WLArrayList< WPosition >::instance();
    m_faces = WLArrayList< WVector3i >::instance();
}

WLEMDEEG::WLEMDEEG( const WLEMDEEG& eeg ) :
                WLEMData( eeg )
{
    m_chanPos3d = eeg.m_chanPos3d;
    m_faces = eeg.m_faces;
    m_polarityType = eeg.getPolarityType();
}

WLEMDEEG::~WLEMDEEG()
{
}

WLEMData::SPtr WLEMDEEG::clone() const
{
    WLEMDEEG::SPtr eeg( new WLEMDEEG( *this ) );
    return eeg;
}

LaBP::WEPolarityType::Enum WLEMDEEG::getPolarityType() const
{
    return m_polarityType;
}

LaBP::WEModalityType::Enum WLEMDEEG::getModalityType() const
{
    return LaBP::WEModalityType::EEG;
}

WLArrayList< WPosition >::SPtr WLEMDEEG::getChannelPositions3d()
{
    return m_chanPos3d;
}

WLArrayList< WPosition >::ConstSPtr WLEMDEEG::getChannelPositions3d() const
{
    return m_chanPos3d;
}

void WLEMDEEG::setChannelPositions3d( WLArrayList< WPosition >::SPtr chanPos3d )
{
    m_chanPos3d = chanPos3d;
}

void WLEMDEEG::setChannelPositions3d( boost::shared_ptr< std::vector< WPosition > > chanPos3d )
{
    m_chanPos3d = WLArrayList< WPosition >::instance( *chanPos3d );
}

WLArrayList< WVector3i >::SPtr WLEMDEEG::getFaces()
{
    return m_faces;
}

WLArrayList< WVector3i >::ConstSPtr WLEMDEEG::getFaces() const
{
    return m_faces;
}

void WLEMDEEG::setFaces( boost::shared_ptr< std::vector< WVector3i > > faces )
{
    m_faces = WLArrayList< WVector3i >::instance( *faces );
}

void WLEMDEEG::setFaces( WLArrayList< WVector3i >::SPtr faces )
{
    m_faces = faces;
}
