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

#include <vector>

#include <boost/shared_ptr.hpp>

#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>

#include "WLEMData.h"
#include "WLEMDEEG.h"

WLEMDEEG::WLEMDEEG() :
                WLEMData()
{
    m_positions = WLPositions::instance();
    m_faces = WLArrayList< WVector3i >::instance();
}

WLEMDEEG::WLEMDEEG( const WLEMDEEG& eeg ) :
                WLEMData( eeg )
{
    m_positions = eeg.m_positions;
    m_faces = eeg.m_faces;
}

WLEMDEEG::~WLEMDEEG()
{
}

WLEMData::SPtr WLEMDEEG::clone() const
{
    WLEMDEEG::SPtr eeg( new WLEMDEEG( *this ) );
    return eeg;
}

WLEModality::Enum WLEMDEEG::getModalityType() const
{
    return WLEModality::EEG;
}

WLEMDEEG::PositionsT::SPtr WLEMDEEG::getChannelPositions3d()
{
    return m_positions;
}

WLEMDEEG::PositionsT::ConstSPtr WLEMDEEG::getChannelPositions3d() const
{
    return m_positions;
}

void WLEMDEEG::setChannelPositions3d( PositionsT::SPtr positions )
{
    m_positions = positions;
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
