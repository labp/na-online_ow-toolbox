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

#include <core/common/WLogger.h>

#include "WLEMMHpiInfo.h"

const std::string WLEMMHpiInfo::CLASS = "WLEMMHpiInfo";

WLEMMHpiInfo::WLEMMHpiInfo()
{
    m_devToHead.setIdentity();
}

WLEMMHpiInfo::~WLEMMHpiInfo()
{
}

WLEMMHpiInfo::TransformationT WLEMMHpiInfo::getDevToHead() const
{
    return m_devToHead;
}

void WLEMMHpiInfo::setDevToHead( const TransformationT& t )
{
    m_devToHead = t;
}

WLEMMHpiInfo::DigPointsT WLEMMHpiInfo::getDigPoints() const
{
    return m_digPoints;
}

bool WLEMMHpiInfo::setDigPoints( const DigPointsT& digPoints )
{
    m_digPoints.clear();
    DigPointsT::const_iterator it;
    for( it = digPoints.begin(); it != digPoints.end(); ++it )
    {
        if( it->getKind() == WLEPointType::HPI )
        {
            m_digPoints.push_back( *it );
        }
    }

    if( !m_hpiFrequencies.empty() && m_hpiFrequencies.size() != m_digPoints.size() )
    {
        wlog::warn( CLASS ) << "Attention, count of digitization points and frequencies is not equal!";
    }
    return !m_digPoints.empty();
}

bool WLEMMHpiInfo::addDigPoint( const WLDigPoint& digPoint )
{
    if( digPoint.getKind() == WLEPointType::HPI )
    {
        m_digPoints.push_back( digPoint );
        return true;
    }
    else
    {
        return false;
    }
}

void WLEMMHpiInfo::clearDigPoints()
{
    m_digPoints.clear();
}

WLEMMHpiInfo::HpiFrequenciesT WLEMMHpiInfo::getHpiFrequencies() const
{
    return m_hpiFrequencies;
}

void WLEMMHpiInfo::setHpiFrequencies( const HpiFrequenciesT& freqs )
{
    if( !m_digPoints.empty() && m_digPoints.size() != freqs.size() )
    {
        wlog::warn( CLASS ) << "Attention, count of digitization points and frequencies is not equal!";
    }
    m_hpiFrequencies = freqs;
}

void WLEMMHpiInfo::addHpiFrequency( WLFreqT freq )
{
    if( freq < 100 )
    {
        wlog::warn( CLASS ) << "Frequency of " << freq << " is very low!";
    }
    m_hpiFrequencies.push_back( freq );
}

void WLEMMHpiInfo::clearHpiFrequencies()
{
    m_hpiFrequencies.clear();
}
