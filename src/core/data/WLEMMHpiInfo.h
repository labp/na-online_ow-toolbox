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

#ifndef CORE_DATA_WLEMMHPIINFO_H_
#define CORE_DATA_WLEMMHPIINFO_H_

#include <list>
#include <ostream>
#include <string>

#include <boost/shared_ptr.hpp>

#include "WLDataTypes.h"
#include "WLDigPoint.h"

/**
 * HPI information.
 *
 * \author pieloth
 * \ingroup data
 */
class WLEMMHpiInfo
{
public:
    typedef boost::shared_ptr< WLEMMHpiInfo > SPtr;
    typedef boost::shared_ptr< const WLEMMHpiInfo > ConstSPtr;

    typedef std::list< WLDigPoint > DigPointsT;
    typedef std::list< WLFreqT > HpiFrequenciesT;
    typedef WLMatrix4::Matrix4T TransformationT;

    static const std::string CLASS;

    WLEMMHpiInfo();
    virtual ~WLEMMHpiInfo();

    /**
     * Gets the transformation for device to head.
     *
     * \return Transformation matrix, default identity matrix.
     */
    TransformationT getDevToHead() const;

    /**
     * Sets the transformation for device to head.
     *
     * \param t Transformation matrix.
     */
    void setDevToHead( const TransformationT& t );

    /**
     * Gets the positions of the HPI coils from hpi_result.
     *
     * \return Digitization of HPI coils.
     */
    DigPointsT getDigPointsResult() const;

    /**
     * Sets digitization of HPI coils for hpi_result. Skips non HPI points.
     *
     * \param digPoints Digitization points.
     * \return True if one or more HPI coils were set.
     */
    bool setDigPointsResult( const DigPointsT& digPoints );

    /**
     * Adds a digitization point for a HPI coil for hpi_result.
     *
     * \param digPoint Digitization point.
     * \return True if point was added.
     */
    bool addDigPointResult( const WLDigPoint& digPoint );

    /**
     * Deletes all digitization points from hpi_result.
     */
    void clearDigPointsResult();

    /**
     * Gets the positions of the HPI coils in head coords.
     *
     * \return Digitization of HPI coils.
     */
    DigPointsT getDigPointsHead() const;

    /**
     * Sets digitization of HPI coils in head coords. Skips non HPI points.
     *
     * \param digPoints Digitization points.
     * \return True if one or more HPI coils were set.
     */
    bool setDigPointsHead( const DigPointsT& digPoints );

    /**
     * Adds a digitization point for a HPI coil in head coords.
     *
     * \param digPoint Digitization point.
     * \return True if point was added.
     */
    bool addDigPointHead( const WLDigPoint& digPoint );

    /**
     * Deletes all digitization points in head coords.
     */
    void clearDigPointsHead();

    /**
     * Gets the frequencies of HPI coils.
     *
     * \return Frequencies of HPI coils in Hz.
     */
    HpiFrequenciesT getHpiFrequencies() const;

    /**
     * Sets the frequencies for HPI coils.
     *
     * \param freqs Frequency in Hz.
     */
    void setHpiFrequencies( const HpiFrequenciesT& freqs );

    /**
     * Adds a frequency for a HPI coil.
     *
     * \param freq Frequency in Hz.
     */
    void addHpiFrequency( WLFreqT freq );

    /**
     * Deletes all frequencies.
     */
    void clearHpiFrequencies();

private:
    TransformationT m_devToHead; //!< Transformation from device to head. Zero if not set/initialized.
    DigPointsT m_digPointsResult;
    DigPointsT m_digPointsHead;
    HpiFrequenciesT m_hpiFrequencies;
};

/**
 * Overload for streamed output.
 */
inline std::ostream& operator<<( std::ostream &strm, const WLEMMHpiInfo& obj )
{

    strm << obj.CLASS;
    strm << ": digPointsResult=" << obj.getDigPointsResult().size();
    strm << "; digPointsHead=" << obj.getDigPointsHead().size();
    strm << "; frequencies=" << obj.getHpiFrequencies().size();
    strm << "; devToHead=";
    WLMatrix4::Matrix4T t = obj.getDevToHead();
    if( t.isIdentity() )
    {
        strm << "IDENTITY";
    }
    else
    {
        strm << "[" << t.row( 0 ) << ", " << t.row( 1 ) << ", " << t.row( 2 ) << ", " << t.row( 3 ) << "]";
    }
    return strm;
}

#endif  // CORE_DATA_WLEMMHPIINFO_H_
