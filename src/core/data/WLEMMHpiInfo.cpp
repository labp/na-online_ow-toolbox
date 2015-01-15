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
    m_devToHead.setZero(); // Init with zero due to indicate it was not set, because transformation matrix has at least one 1.
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

WLEMMHpiInfo::DigPointsT WLEMMHpiInfo::getDigPointsResult() const
{
    return m_digPointsResult;
}

bool WLEMMHpiInfo::setDigPointsResult( const DigPointsT& digPoints )
{
    m_digPointsResult.clear();
    DigPointsT::const_iterator it;
    for( it = digPoints.begin(); it != digPoints.end(); ++it )
    {
        if( it->getKind() == WLEPointType::HPI )
        {
            m_digPointsResult.push_back( *it );
        }
    }

    if( !m_hpiFrequencies.empty() && m_hpiFrequencies.size() != m_digPointsResult.size() )
    {
        wlog::warn( CLASS ) << "Attention, count of digitization points and frequencies is not equal!";
    }
    return !m_digPointsResult.empty();
}

bool WLEMMHpiInfo::addDigPointResult( const WLDigPoint& digPoint )
{
    if( digPoint.getKind() == WLEPointType::HPI )
    {
        m_digPointsResult.push_back( digPoint );
        return true;
    }
    else
    {
        return false;
    }
}

void WLEMMHpiInfo::clearDigPointsResult()
{
    m_digPointsResult.clear();
}

WLEMMHpiInfo::DigPointsT WLEMMHpiInfo::getDigPointsHead() const
{
    return m_digPointsHead;
}

bool WLEMMHpiInfo::setDigPointsHead( const DigPointsT& digPoints )
{
    m_digPointsHead.clear();
    DigPointsT::const_iterator it;
    for( it = digPoints.begin(); it != digPoints.end(); ++it )
    {
        if( it->getKind() == WLEPointType::HPI )
        {
            m_digPointsHead.push_back( *it );
        }
    }

    if( !m_hpiFrequencies.empty() && m_hpiFrequencies.size() != m_digPointsHead.size() )
    {
        wlog::warn( CLASS ) << "Attention, count of digitization points and frequencies is not equal!";
    }
    return !m_digPointsHead.empty();
}

bool WLEMMHpiInfo::addDigPointHead( const WLDigPoint& digPoint )
{
    if( digPoint.getKind() == WLEPointType::HPI )
    {
        m_digPointsHead.push_back( digPoint );
        return true;
    }
    else
    {
        return false;
    }
}

void WLEMMHpiInfo::clearDigPointsHead()
{
    m_digPointsHead.clear();
}

WLEMMHpiInfo::HpiFrequenciesT WLEMMHpiInfo::getHpiFrequencies() const
{
    return m_hpiFrequencies;
}

void WLEMMHpiInfo::setHpiFrequencies( const HpiFrequenciesT& freqs )
{
    if( !m_digPointsResult.empty() && m_digPointsResult.size() != freqs.size() )
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
