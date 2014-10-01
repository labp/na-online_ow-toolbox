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

#include <boost/foreach.hpp>
#include <boost/shared_ptr.hpp>

#include <core/common/WAssert.h>
#include <core/common/WLogger.h>
#include <core/common/exceptions/WPreconditionNotMet.h>
#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>

#include "WLEMData.h"
#include "WLEMDMEG.h"

const std::string WLEMDMEG::CLASS = "WLEMDMEG";

WLEMDMEG::WLEMDMEG() :
                WLEMData()
{
    m_modality = WLEModality::MEG;
    m_chanPos3d = WLArrayList< WPosition >::instance();
    m_faces = WLArrayList< WVector3i >::instance();

    m_eX = WLArrayList< WVector3f >::instance();
    m_eY = WLArrayList< WVector3f >::instance();
    m_eZ = WLArrayList< WVector3f >::instance();
}

WLEMDMEG::WLEMDMEG( WLEModality::Enum modality )
{
    if( !WLEModality::isMEG( modality ) )
    {
        throw WPreconditionNotMet( "Modality must be MEG, gradiometer or magnetometer!" );
    }

    m_modality = modality;
    m_chanPos3d = WLArrayList< WPosition >::instance();
    m_faces = WLArrayList< WVector3i >::instance();

    m_eX = WLArrayList< WVector3f >::instance();
    m_eY = WLArrayList< WVector3f >::instance();
    m_eZ = WLArrayList< WVector3f >::instance();
}

WLEMDMEG::WLEMDMEG( const WLEMDMEG& meg ) :
                WLEMData( meg )
{
    m_modality = meg.m_modality;
    m_chanPos3d = meg.m_chanPos3d;
    m_faces = meg.m_faces;
    m_eX = meg.m_eX;
    m_eY = meg.m_eY;
    m_eZ = meg.m_eZ;
}

WLEMDMEG::~WLEMDMEG()
{
}

WLEMData::SPtr WLEMDMEG::clone() const
{
    WLEMDMEG::SPtr meg( new WLEMDMEG( *this ) );
    return meg;
}

WLEModality::Enum WLEMDMEG::getModalityType() const
{
    return m_modality;
}

WLArrayList< WPosition >::SPtr WLEMDMEG::getChannelPositions3d()
{
    return m_chanPos3d;
}

WLArrayList< WPosition >::ConstSPtr WLEMDMEG::getChannelPositions3d() const
{
    return m_chanPos3d;
}

WLArrayList< WPosition >::ConstSPtr WLEMDMEG::getChannelPositions3d( WLEMEGGeneralCoilType::Enum type ) const
{
    if( m_chanPos3d->size() % 3 != 0 || m_chanPos3d->empty() )
    {
        return WLArrayList< WPosition >::ConstSPtr( new WLArrayList< WPosition > );
    }

    std::vector< size_t > picks = getPicks( type );
    WLArrayList< WPosition >::SPtr posPtr( new WLArrayList< WPosition > );
    WLArrayList< WPosition >& positions = *posPtr;
    positions.reserve( picks.size() );

    std::vector< size_t >::const_iterator it;
    for( it = picks.begin(); it != picks.end(); ++it )
    {
        positions.push_back( m_chanPos3d->at( *it ) );
    }

    return posPtr;
}

void WLEMDMEG::setChannelPositions3d( WLArrayList< WPosition >::SPtr chanPos3d )
{
    m_chanPos3d = chanPos3d;
}

void WLEMDMEG::setChannelPositions3d( boost::shared_ptr< std::vector< WPosition > > chanPos3d )
{
    m_chanPos3d = WLArrayList< WPosition >::instance( *chanPos3d );
}

WLArrayList< WVector3i >::SPtr WLEMDMEG::getFaces()
{
    return m_faces;
}

WLArrayList< WVector3i >::ConstSPtr WLEMDMEG::getFaces() const
{
    return m_faces;
}

WLArrayList< WVector3i >::ConstSPtr WLEMDMEG::getFaces( WLEMEGGeneralCoilType::Enum type ) const
{
    if( m_faces->size() % 3 != 0 || m_faces->empty() )
    {
        return WLArrayList< WVector3i >::ConstSPtr( new WLArrayList< WVector3i > );
    }

    std::vector< size_t > picks = getPicks( type );
    WLArrayList< WVector3i >::SPtr facesPtr( new WLArrayList< WVector3i > );
    WLArrayList< WVector3i >& faces = *facesPtr;
    faces.reserve( picks.size() );

    size_t row = 0;
    std::vector< size_t >::const_iterator it;
    for( it = picks.begin(); it != picks.end(); ++it )
    {
        faces.push_back( m_faces->at( *it ) );
    }

    return facesPtr;
}

void WLEMDMEG::setFaces( boost::shared_ptr< std::vector< WVector3i > > faces )
{
    m_faces = WLArrayList< WVector3i >::instance( *faces );
}

void WLEMDMEG::setFaces( WLArrayList< WVector3i >::SPtr faces )
{
    m_faces = faces;
}

WLArrayList< WVector3f >::SPtr WLEMDMEG::getEx()
{
    return m_eX;
}

WLArrayList< WVector3f >::ConstSPtr WLEMDMEG::getEx() const
{
    return m_eX;
}

void WLEMDMEG::setEx( WLArrayList< WVector3f >::SPtr vec )
{
    m_eX = vec;
}

WLArrayList< WVector3f >::SPtr WLEMDMEG::getEy()
{
    return m_eY;
}

WLArrayList< WVector3f >::ConstSPtr WLEMDMEG::getEy() const
{
    return m_eY;
}

void WLEMDMEG::setEy( WLArrayList< WVector3f >::SPtr vec )
{
    m_eY = vec;
}

WLArrayList< WVector3f >::SPtr WLEMDMEG::getEz()
{
    return m_eZ;
}

WLArrayList< WVector3f >::ConstSPtr WLEMDMEG::getEz() const
{
    return m_eZ;
}

void WLEMDMEG::setEz( WLArrayList< WVector3f >::SPtr vec )
{
    m_eZ = vec;
}

WLEMDMEG::DataSPtr WLEMDMEG::getData( WLEMEGGeneralCoilType::Enum type ) const
{
    if( getNrChans() % 3 != 0 )
    {
        return WLEMDMEG::DataSPtr( new WLEMDMEG::DataT );
    }

    std::vector< size_t > picks = getPicks( type );
    WLEMDMEG::DataSPtr dataPtr( new WLEMDMEG::DataT( picks.size(), getSamplesPerChan() ) );
    WLEMDMEG::DataT& data = *dataPtr;

    size_t row = 0;
    std::vector< size_t >::const_iterator it;
    for( it = picks.begin(); it != picks.end(); ++it )
    {
        data.row( row++ ) = m_data->row( *it );
    }

    return dataPtr;
}

WLEMDMEG::DataSPtr WLEMDMEG::getDataBadChannels( WLEMEGGeneralCoilType::Enum type ) const
{
    if( getNrChans() % 3 != 0 )
    {
        return WLEMDMEG::DataSPtr( new WLEMDMEG::DataT );
    }

    std::vector< size_t > picks = getPicks( type );
    WLEMDMEG::DataSPtr dataPtr( new WLEMDMEG::DataT( picks.size() - getNrBadChans( type ), getSamplesPerChan() ) );
    WLEMDMEG::DataT& data = *dataPtr;

    size_t row = 0;

    BOOST_FOREACH( size_t it, picks )
    {
        if( isBadChannel( it ) )
        {
            continue;
        }

        data.row( row ) = m_data->row( it );

        ++row;
    }

    return dataPtr;
}

WLEMDMEG::DataSPtr WLEMDMEG::getDataBadChannels( WLEMEGGeneralCoilType::Enum type, ChannelListSPtr badChans ) const
{
    if( getNrChans() % 3 != 0 )
    {
        return WLEMDMEG::DataSPtr( new WLEMDMEG::DataT );
    }

    if( badChans == 0 )
    {
        return getDataBadChannels( type );
    }

    std::vector< size_t > picks = getPicks( type );
    WLEMDMEG::DataSPtr dataPtr( new WLEMDMEG::DataT( picks.size() - getNrBadChans( type ), getSamplesPerChan() ) );
    WLEMDMEG::DataT& data = *dataPtr;

    size_t row = 0;

    BOOST_FOREACH( size_t it, picks )
    {
        if( isBadChannel( it ) || std::find( badChans->begin(), badChans->end(), it ) != badChans->end() )
        {
            continue;
        }

        data.row( row ) = m_data->row( it );

        ++row;
    }

    return dataPtr;
}

std::vector< size_t > WLEMDMEG::getPicks( WLEMEGGeneralCoilType::Enum type ) const
{
    if( m_picksMag.size() + m_picksGrad.size() != getNrChans() )
    {
        m_picksGrad.clear();
        m_picksMag.clear();
        const size_t rows = getNrChans();
        if( rows % 3 != 0 )
        {
            return std::vector< size_t >(); // empty vector
        }

        m_picksMag.reserve( rows / 3 );
        m_picksMag.reserve( ( rows / 3 ) * 2 );

        for( size_t ch = 0; ch < rows; ++ch )
        {
            switch( getChannelType( ch ) )
            {
                case WLEMEGGeneralCoilType::MAGNETOMETER:
                    m_picksMag.push_back( ch );
                    break;
                case WLEMEGGeneralCoilType::GRADIOMETER:
                    m_picksGrad.push_back( ch );
                    break;
            }
        }
    }

    switch( type )
    {
        case WLEMEGGeneralCoilType::MAGNETOMETER:
            return m_picksMag;
        case WLEMEGGeneralCoilType::GRADIOMETER:
            return m_picksGrad;
        default:
            return std::vector< size_t >(); // empty vector
    }
}

WLEMDMEG::CoilPicksT WLEMDMEG::coilPicks( const WLEMDMEG& meg, WLEMEGGeneralCoilType::Enum type )
{
    CoilPicksT picks;
    const size_t rows = meg.getNrChans();
    if( rows % 3 != 0 )
    {
        wlog::error( CLASS ) << "channels % 3 != 0";
        return picks; // empty vector
    }

    switch( type )
    {
        case WLEMEGGeneralCoilType::MAGNETOMETER:
            picks.reserve( rows / 3 );
            break;
        case WLEMEGGeneralCoilType::GRADIOMETER:
            picks.reserve( ( rows / 3 ) * 2 );
            break;
    }

    for( size_t ch = 0; ch < rows; ++ch )
    {
        if( meg.getChannelType( ch ) == type )
        {
            picks.push_back( ch );
        }
    }

    return picks;
}

bool WLEMDMEG::extractCoilModality( WLEMDMEG::SPtr& megOut, WLEMDMEG::ConstSPtr megIn, WLEModality::Enum type, bool dataOnly )
{
    if( !WLEModality::isMEG( type ) )
    {
        wlog::error( CLASS ) << "Requested extraction into a non-MEG type!";
        return false;
    }

    CoilPicksT picksAll;
    switch( type )
    {
        case WLEModality::MEG:
        {
            megOut.reset( new WLEMDMEG( *megIn ) );
            WLEMDMEG::DataSPtr data( new WLEMDMEG::DataT( megIn->getData() ) );
            megOut->setData( data );
            return true;
        }
        case WLEModality::MEG_MAG:
            picksAll = coilPicks( *megIn, WLEMEGGeneralCoilType::MAGNETOMETER );
            break;
        case WLEModality::MEG_GRAD:
            picksAll = coilPicks( *megIn, WLEMEGGeneralCoilType::GRADIOMETER );
            break;
        case WLEModality::MEG_GRAD_MERGED:
            picksAll = coilPicks( *megIn, WLEMEGGeneralCoilType::GRADIOMETER );
            break;
        default:
            wlog::error( CLASS ) << "Requested formation into a non-MEG type!";
            return false;
    }

    megOut.reset( new WLEMDMEG( type ) );
    if( picksAll.empty() )
    {
        wlog::warn( CLASS ) << "Requested extraction into the same type!";
        return false;
    }

    CoilPicksT picksFiltered;
    if( type != WLEModality::MEG_GRAD_MERGED )
    {
        picksFiltered = picksAll;
    }
    else
    {
        picksFiltered.reserve( picksAll.size() / 2 );
        for( size_t i = 0; i < picksAll.size(); i += 2 )
        {
            picksFiltered.push_back( picksAll[i] );
        }
    }

    CoilPicksT::const_iterator it;

    WLEMDMEG::DataSPtr dataPtr( new WLEMDMEG::DataT( picksFiltered.size(), megIn->getSamplesPerChan() ) );
    WLEMDMEG::DataT& data = *dataPtr;

    const WLEMDMEG::DataT& data_from = megIn->getData();
    WLEMDMEG::DataT::Index row = 0;
    WLEMDMEG::ChannelT chan1, chan2;
    for( it = picksAll.begin(); it != picksAll.end(); ++it )
    {
        if( type != WLEModality::MEG_GRAD_MERGED )
        {
            data.row( row++ ) = data_from.row( *it );
        }
        else
        {
            chan1 = data_from.row( *it );
            chan2 = data_from.row( *( ++it ) );

            chan1 = 0.5 * ( chan1.cwiseProduct( chan1 ) + chan2.cwiseProduct( chan2 ) );
            data.row( row++ ) = chan1.cwiseSqrt();
        }
    }

    megOut->setData( dataPtr );
    megOut->setSampFreq( megIn->getSampFreq() );
    if( dataOnly )
    {
        return true;
    }

    const WLArrayList< std::string >& chNames_from = *megIn->getChanNames();
    WLArrayList< std::string >& chNames = *megOut->getChanNames();
    chNames.reserve( picksFiltered.size() );
    if( picksFiltered.size() <= chNames_from.size() )
    {
        for( it = picksFiltered.begin(); it != picksFiltered.end(); ++it )
        {
            chNames.push_back( chNames_from[*it] );
        }
    }

    const WLArrayList< WPosition >& chPos_from = *megIn->getChannelPositions3d();
    WLArrayList< WPosition >& chPos = *megOut->getChannelPositions3d();
    chPos.reserve( picksFiltered.size() );
    if( picksFiltered.size() <= chPos_from.size() )
    {
        for( it = picksFiltered.begin(); it != picksFiltered.end(); ++it )
        {
            chPos.push_back( chPos_from[*it] );
        }
    }

    const WLArrayList< WVector3i >& chFaces_from = *megIn->getFaces();
    WLArrayList< WVector3i >& chFaces = *megOut->getFaces();
    chFaces.reserve( picksFiltered.size() );
    if( picksFiltered.size() <= chFaces_from.size() )
    {
        for( it = picksFiltered.begin(); it != picksFiltered.end(); ++it )
        {
            chFaces.push_back( chFaces_from[*it] );
        }
    }

    const WLArrayList< WVector3f >& eX_from = *megIn->getEx();
    const WLArrayList< WVector3f >& eY_from = *megIn->getEy();
    const WLArrayList< WVector3f >& eZ_from = *megIn->getEz();
    WLArrayList< WVector3f >& eX = *megOut->getEx();
    WLArrayList< WVector3f >& eY = *megOut->getEy();
    WLArrayList< WVector3f >& eZ = *megOut->getEz();
    eX.reserve( picksFiltered.size() );
    eY.reserve( picksFiltered.size() );
    eZ.reserve( picksFiltered.size() );
    const bool eq = eX_from.size() == eY_from.size() && eX_from.size() == eZ_from.size();
    if( picksFiltered.size() <= eX_from.size() && eq )
    {
        for( it = picksFiltered.begin(); it != picksFiltered.end(); ++it )
        {
            eX.push_back( eX_from[*it] );
            eY.push_back( eY_from[*it] );
            eZ.push_back( eZ_from[*it] );
        }
    }

    return true;
}

size_t WLEMDMEG::getNrBadChans( WLEMEGGeneralCoilType::Enum type ) const
{
    size_t count = 0;
    std::vector< size_t > picks = getPicks( type );

    for( size_t i = 0; i < picks.size(); ++i )
    {
        if( isBadChannel( picks.at( i ) ) )
        {
            ++count;
        }
    }

    return count;
}
