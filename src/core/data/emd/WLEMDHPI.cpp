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

#include "core/common/WLogger.h"

#include "WLEMDHPI.h"

const std::string WLEMDHPI::CLASS = "WLEMDHPI";

WLEMDHPI::WLEMDHPI() :
                WLEMData()
{
    m_chanNames->reserve( 5 );
    m_nrHpiCoils = 0;

    m_chanPos3d = WLArrayList< WPosition >::instance();
    m_chanPos3d->reserve( 5 );
    m_transformations = WLArrayList< TransformationT >::instance();
}

WLEMDHPI::WLEMDHPI( const WLEMDHPI& hpi )
{
    m_nrHpiCoils = hpi.m_nrHpiCoils;
    m_chanPos3d = hpi.m_chanPos3d;
}

WLEMDHPI::~WLEMDHPI()
{
}

WLEMData::SPtr WLEMDHPI::clone() const
{
    WLEMDHPI::SPtr hpi( new WLEMDHPI( *this ) );
    return hpi;
}

WLEModality::Enum WLEMDHPI::getModalityType() const
{
    return WLEModality::HPI;
}

WLArrayList< WPosition >::SPtr WLEMDHPI::getChannelPositions3d()
{
    return m_chanPos3d;
}

WLArrayList< WPosition >::ConstSPtr WLEMDHPI::getChannelPositions3d() const
{
    return m_chanPos3d;
}

void WLEMDHPI::setChannelPositions3d( WLArrayList< WPosition >::SPtr chanPos3d )
{
    m_chanPos3d = chanPos3d;
}

void WLEMDHPI::setChannelPositions3d( boost::shared_ptr< std::vector< WPosition > > chanPos3d )
{
    m_chanPos3d = WLArrayList< WPosition >::instance( *chanPos3d );
}

bool WLEMDHPI::setChannelPositions3d( WLList< WLDigPoint >::ConstSPtr digPoints )
{
    if( m_chanPos3d->size() > 0 )
    {
        wlog::warn( CLASS ) << "Overwriting channel positions.";
        m_chanPos3d->clear();
    }

    WLList< WLDigPoint >::const_iterator it;
    for( it = digPoints->begin(); it != digPoints->end(); ++it )
    {
        if( it->getKind() == WLEPointType::HPI )
        {
            m_chanPos3d->push_back( it->getPoint() );
        }
    }

    if( getNrChans() > 0 && ( getNrChans() % m_chanPos3d->size() == 0 ) )
    {
        return true;
    }
    if( getNrChans() > 0 && ( getNrChans() % m_chanPos3d->size() != 0 ) )
    {
        wlog::error( CLASS ) << "Found positions do not match channel count!";
        m_chanPos3d->clear();
        return false;
    }
    return m_chanPos3d->size() > 0;
}

WLChanNrT WLEMDHPI::getNrHpiCoils() const
{
    return m_nrHpiCoils;
}

bool WLEMDHPI::setNrHpiCoils( WLChanNrT count )
{
    const WLChanNrT n_chans = getNrChans();
    if( n_chans > 0 && ( n_chans % count != 0 ) )
    {
        wlog::error( CLASS ) << "Number of HPI coils not set! Count does not match data channels.";
        return false;
    }
    const size_t n_pos = getChannelPositions3d()->size();
    if( n_pos > 0 && ( n_pos != count ) )
    {
        wlog::error( CLASS ) << "Number of HPI coils not set! Count does not match positions.";
        return false;
    }

    m_nrHpiCoils = count;
    return true;
}

WLArrayList< WLEMDHPI::TransformationT >::SPtr WLEMDHPI::getTransformations()
{
    return m_transformations;
}

WLArrayList< WLEMDHPI::TransformationT >::ConstSPtr WLEMDHPI::getTransformations() const
{
    return m_transformations;
}

void WLEMDHPI::setTransformations( WLArrayList< TransformationT >::SPtr trans )
{
    m_transformations = trans;
}
