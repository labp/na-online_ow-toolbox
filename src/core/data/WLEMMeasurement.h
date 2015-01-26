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

#ifndef WLEMMEASUREMENT_H
#define WLEMMEASUREMENT_H

#include <ostream>
#include <string>
#include <utility>
#include <vector>
#include <set>

#include <boost/shared_ptr.hpp>

#include <core/common/WDefines.h>
#include <core/common/exceptions/WNotFound.h>

#include "core/container/WLList.h"
#include "core/data/WLDataTypes.h"
#include "core/data/WLDigPoint.h"
#include "core/data/WLEMMHpiInfo.h"
#include "core/data/WLEMMSubject.h"
#include "core/data/WLTransformation.h"
#include "core/data/emd/WLEMData.h"
#include "core/data/enum/WLEModality.h"
#include "core/data/enum/WLEPointType.h"
#include "core/util/profiler/WLLifetimeProfiler.h"

/**
 * \brief Electromagnetic measurement contains all data and information about a measurement.
 * Electromagnetic measurement contains all data and information about a measurement,
 * e.g. EEG/MEG data, subject information, surfaces and more.
 *
 * \authors kaehler, pieloth
 * \ingroup data
 */
class WLEMMeasurement
{
public:
    typedef boost::shared_ptr< WLEMMeasurement > SPtr; //!< Abbreviation for a shared pointer.

    typedef boost::shared_ptr< const WLEMMeasurement > ConstSPtr; //!< Abbreviation for const shared pointer.

    typedef int EventT; //!< Data type for events.

    typedef std::vector< EventT > EChannelT; //!< An event channel.

    typedef std::vector< EChannelT > EDataT; //!< Event data.

    static const std::string CLASS; //!< Class name for logging purpose.

    /**
     * Constructor.
     */
    WLEMMeasurement();

    /**
     * Constructor.
     *
     * \param subject Subject to use.
     */
    explicit WLEMMeasurement( WLEMMSubject::SPtr subject );

    /**
     * Copy constructor, creates a shallow copy from object.
     *
     * \note Modalities are not copied.
     * \param emm The object to copy from.
     */
    explicit WLEMMeasurement( const WLEMMeasurement& emm );

    /**
     * Creates a shallow copy from the instance.
     *
     * \note Modalities are not copied.
     * \return A new instance without the modalities.
     */
    WLEMMeasurement::SPtr clone() const;

    /**
     * Destructor.
     */
    virtual ~WLEMMeasurement();

    /**
     * Adds a modality to the modality list.
     * NOTE: MEG_MAG, MEG_GRAD and MEG_GRAD_MERGE are not added!
     *
     * \param modality modality to add to list
     * \return True, if modality was added.
     */
    bool addModality( WLEMData::SPtr modality );

    /**
     * getter for vector of modalities
     *
     * \return vector of modalities
     */
    std::vector< WLEMData::SPtr > getModalityList();

    /**
     * setter for Modality list
     *
     * \param list the new modality list
     * \return Number of modalities which are added.
     */
    size_t setModalityList( const std::vector< WLEMData::SPtr >& list );

    /**
     * Returns the number modalities.
     */
    size_t getModalityCount() const;

    /**
     * Returns the modality or an empty shared pointer. Throws an exception if i >= size.
     */
    WLEMData::SPtr getModality( size_t i );

    /**
     * Returns the modality or an empty shared pointer. Throws an exception if i >= size.
     */
    WLEMData::ConstSPtr getModality( size_t i ) const;

    /**
     * Returns the first occurrence of EMMEMD  with the given type or an empty shared pointer.
     *
     * \throws WNotFound if requested type is not available.
     */
    WLEMData::SPtr getModality( WLEModality::Enum type );

    /**
     * Returns the first occurrence of EMMEMD  with the given type or an empty shared pointer.
     *
     * \throws WNotFound if requested type is not available.
     */
    WLEMData::ConstSPtr getModality( WLEModality::Enum type ) const;

    /**
     * Returns the first occurrence of EMMEMD  with the given type or an empty shared pointer.
     *
     * \throws WNotFound if requested type is not available.
     */
    template< typename EMD >
    boost::shared_ptr< EMD > getModality( WLEModality::Enum type )
    {
        WLEMData::SPtr emd = getModality( type );
        if( !emd )
        {
            throw WNotFound( "Modality type not available!" );
        }
        return emd->getAs< EMD >();
    }

    /**
     * Returns the first occurrence of EMMEMD  with the given type or an empty shared pointer.
     *
     * \throws WNotFound if requested type is not available.
     */
    template< typename EMD >
    boost::shared_ptr< const EMD > getModality( WLEModality::Enum type ) const
    {
        WLEMData::ConstSPtr emd = getModality( type );
        if( !emd )
        {
            throw WNotFound( "Modality type not available!" );
        }
        return emd->getAs< EMD >();
    }

    /**
     * Returns a set with available modality types.
     */
    std::set< WLEModality::Enum > getModalityTypes() const;

    /**
     * Checks if a modality type is available.
     *
     * \return true if modality type is available, false if not.
     */
    bool hasModality( WLEModality::Enum type ) const;

    /**
     * getter for experimenter
     *
     * \return Experimenter as string
     */
    std::string getExperimenter() const;

    /**
     * getter for optional experiment description
     *
     * \return experiment description as string
     */
    std::string getExpDescription() const;

    /**
     * getter for subject
     *
     * \return Experimenter as string
     */
    WLEMMSubject::SPtr getSubject();

    /**
     * getter for subject
     *
     * \return Experimenter as string
     */
    WLEMMSubject::ConstSPtr getSubject() const;

    /**
     * setter for experimenter
     *
     * \param experimenter Experimenter as string
     */
    void setExperimenter( std::string experimenter );

    /**
     * setter for optional experiment description
     *
     * \param expDescription experiment description as string
     */
    void setExpDescription( std::string expDescription );

    /**
     * setter for subject
     *
     * \param subject Experimenter as string
     */
    void setSubject( WLEMMSubject::SPtr subject );

    /**
     * Returns the event/stimuli channels.
     */
    boost::shared_ptr< EDataT > getEventChannels() const;

    /**
     * Sets the event/stimuli channel/data.
     */
    void setEventChannels( boost::shared_ptr< EDataT > data );

    /**
     * Adds an event channels.
     */
    void addEventChannel( EChannelT& data );

    /**
     * Returns the event/stimuli channel.
     */
    EChannelT& getEventChannel( WLChanIdxT i ) const;

    /**
     * Returns number of event channels.
     */
    WLChanNrT getEventChannelCount() const;

    /**
     * Gets profiler for lifetime and clone counter.
     *
     * \return profiler
     */
    WLLifetimeProfiler::SPtr getProfiler();

    /**
     * Gets profiler for lifetime and clone counter.
     *
     * \return profiler
     */
    WLLifetimeProfiler::ConstSPtr getProfiler() const;

    /**
     * Sets profiler for lifetime and clone counter.
     *
     * \param profiler
     */
    void setProfiler( WLLifetimeProfiler::SPtr profiler );

    /**
     * Gets the digitized points, i.e. EEG and HPI.
     *
     * \return digitized points
     */
    WLList< WLDigPoint >::SPtr getDigPoints();

    /**
     * Gets the digitized points, i.e. EEG and HPI.
     *
     * \return digitized points
     */
    WLList< WLDigPoint >::ConstSPtr getDigPoints() const;

    /**
     * Gets the digitized points of a specified kind, e.g. EEG or HPI.
     *
     * \return A new list containing all points of the requested kind, maybe empty.
     */
    WLList< WLDigPoint >::SPtr getDigPoints( WLEPointType::Enum kind ) const;

    /**
     * Sets the digitized points, i.e. EEG and HPI.
     *
     * \param digPoints
     */
    void setDigPoints( WLList< WLDigPoint >::SPtr digPoints );

    /**
     * Gets the transformation matrix: device to fiducial.
     *
     * \return %transformation matrix
     */
    WLTransformation::SPtr getDevToFidTransformation();

    /**
     * Gets the transformation matrix: device to fiducial.
     *
     * \return %transformation matrix
     */
    WLTransformation::ConstSPtr getDevToFidTransformation() const;

    /**
     * Sets the transformation matrix: device to fiducial.
     *
     * \param mat %Transformation matrix.
     */
    void setDevToFidTransformation( WLTransformation::SPtr mat );

    /**
     * Gets the transformation matrix: fiducial to ACPC.
     *
     * \return %transformation matrix
     */
    WLTransformation::SPtr getFidToACPCTransformation();

    /**
     * Gets the transformation matrix: fiducial to ACPC.
     *
     * \return %transformation matrix
     */
    WLTransformation::ConstSPtr getFidToACPCTransformation() const;

    /**
     * Sets the transformation matrix: fiducial to ACPC.
     *
     * \param mat %Transformation matrix.
     */
    void setFidToACPCTransformation( WLTransformation::SPtr mat );

    /**
     * Gets the HPI information.
     *
     * \return HPI information
     */
    WLEMMHpiInfo::SPtr getHpiInfo();

    /**
     * Gets the HPI information.
     *
     * \return HPI information
     */
    WLEMMHpiInfo::ConstSPtr getHpiInfo() const;

    /**
     * Sets the HPI information.
     *
     * \param hpiInfo HPI information to set.
     */
    void setHpiInfo( WLEMMHpiInfo::SPtr hpiInfo );

private:
    WLLifetimeProfiler::SPtr m_profiler;

    std::string m_experimenter; //!< experiment supervisor.

    std::string m_expDescription; //!< description of experiment.

    std::vector< WLEMData::SPtr > m_modalityList; //!< Container for EMDs.

    WLEMMSubject::SPtr m_subject; //!< Subject information.

    boost::shared_ptr< std::vector< EChannelT > > m_eventChannels; //!< Event/Stimuli channels

    WLList< WLDigPoint >::SPtr m_digPoints; //!< Digitized points.

    WLTransformation::SPtr m_transDevToFid; //!< %Transformation matrix: device to fiducial.

    WLTransformation::SPtr m_transFidToACPC; //!< %Transformation matrix: fiducial to ACPC.

    WLEMMHpiInfo::SPtr m_hpiInfo; //!< HPI info containing coil position and frequencies.
};

inline std::ostream& operator<<( std::ostream &strm, const WLEMMeasurement& obj )
{
    strm << WLEMMeasurement::CLASS << ": modalities=[";
    for( size_t m = 0; m < obj.getModalityCount(); ++m )
    {
        strm << "{" << *obj.getModality( m ) << "}, ";
    }
    strm << "]";
    strm << ", digPoints=" << obj.getDigPoints()->size();
    strm << ", eventChannels=" << obj.getEventChannelCount();
    strm << ", hpiInfo={" << *obj.getHpiInfo() << "}";
    return strm;
}

#endif  // WLEMMEASUREMENT_H
