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

#include <boost/shared_ptr.hpp>

#include "WLEMData.h"
#include "WLEMDSource.h"

const std::string WLEMDSource::CLASS = "WDataSetEMMSource";

WLEMDSource::WLEMDSource() :
                WLEMData()
{
}

WLEMDSource::WLEMDSource( const WLEMDSource& source ) :
                WLEMData( source )
{
    m_originModalityType = source.m_originModalityType;
}

WLEMDSource::WLEMDSource( const WLEMData& emd ) :
                WLEMData( emd )
{
    // C++11 supports "delegating constructors". So default initialization could be moved to default constructor.
    m_originModalityType = emd.getModalityType();
}

WLEMDSource::~WLEMDSource()
{
}

WLEModality::Enum WLEMDSource::getModalityType() const
{
    return WLEModality::SOURCE;
}

WLEMData::SPtr WLEMDSource::clone() const
{
    WLEMDSource::SPtr emd( new WLEMDSource( *this ) );
    return emd;
}

WLEModality::Enum WLEMDSource::getOriginModalityType() const
{
    return m_originModalityType;
}

void WLEMDSource::setOriginModalityType( WLEModality::Enum modality )
{
    m_originModalityType = modality;
}
