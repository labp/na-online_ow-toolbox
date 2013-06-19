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

#ifndef WLEMMEASUREMENT_H
#define WLEMMEASUREMENT_H

#include <string>
#include <utility>
#include <vector>
#include <set>

#include <boost/shared_ptr.hpp>

#include "core/util/WLLifetimeProfiler.h"

#include "core/data/emd/WLEMData.h"
#include "WLEMMEnumTypes.h"
#include "WLEMMSubject.h"
/**
 * TODO(kaehler): Comments
 */
class WLEMMeasurement
{
public:
    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WLEMMeasurement > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WLEMMeasurement > ConstSPtr;

    typedef int EventT;

    typedef std::vector< EventT > EChannelT;

    typedef std::vector< EChannelT > EDataT;

    static const std::string CLASS;

    /**
     * TODO(kaehler): Comments
     */
    WLEMMeasurement();

    /**
     * TODO(kaehler): Comments
     */
    explicit WLEMMeasurement( LaBP::WLEMMSubject::SPtr subject );

    /**
     * copy constructor, makes a shallow copy from object except the data vector
     *
     * \param m_WDataSetEMMObject the object to copy from
     */
    explicit WLEMMeasurement( const WLEMMeasurement& emm );

    WLEMMeasurement::SPtr clone() const;

    /**
     * TODO(kaehler): Comments
     */
    virtual ~WLEMMeasurement();

    /**
     * add a Modality to the modality list
     *
     * \param modality modality to add to list
     */
    void addModality( WLEMData::SPtr modality );

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
     */
    void setModalityList( std::vector< WLEMData::SPtr > list );

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
     * Returns the first occurrence of EMMEMD  with the given type or an empty shared pointer. Throws an exception if requested type is not available.
     */
    WLEMData::SPtr getModality( LaBP::WEModalityType::Enum type );

    /**
     * Returns the first occurrence of EMMEMD  with the given type or an empty shared pointer. Throws an exception if requested type is not available.
     */
    WLEMData::ConstSPtr getModality( LaBP::WEModalityType::Enum type ) const;

    /**
     * Returns the first occurrence of EMMEMD  with the given type or an empty shared pointer. Throws an exception if requested type is not available.
     */
    template< typename EMD >
    boost::shared_ptr< EMD > getModality( LaBP::WEModalityType::Enum type )
    {
        WLEMData::SPtr emd = getModality( type );
        if( !emd )
        {
            throw "Modality type not available!";
        }
        return emd->getAs< EMD >();
    }

    /**
     * Returns the first occurrence of EMMEMD  with the given type or an empty shared pointer. Throws an exception if requested type is not available.
     */
    template< typename EMD >
    boost::shared_ptr< const EMD > getModality( LaBP::WEModalityType::Enum type ) const
    {
        WLEMData::ConstSPtr emd = getModality( type );
        if( !emd )
        {
            throw "Modality type not available!";
        }
        return emd->getAs< EMD >();
    }

    /**
     * Returns a set with available modality types.
     */
    std::set< LaBP::WEModalityType::Enum > getModalityTypes() const;

    /**
     * Checks if a modality type is available.
     *
     * \return true if modality type is available, false if not.
     */
    bool hasModality( LaBP::WEModalityType::Enum type ) const;

    /**
     * swaps given modality with modality of the same modality type in list, there is only one list per modality at the time
     * TODO (pieloth): why returning WDataSetEMM
     *
     * \param modality modality to swap with
     */
    WLEMMeasurement::SPtr newModalityData( WLEMData::SPtr modality );

    // -----------getter and setter-----------------------------------------------------------------------------

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
    LaBP::WLEMMSubject::SPtr getSubject();

    /**
     * getter for subject
     *
     * \return Experimenter as string
     */
    LaBP::WLEMMSubject::ConstSPtr getSubject() const;

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
    void setSubject( LaBP::WLEMMSubject::SPtr subject );

    /**
     * Returns the event/stimuli channels.
     */
    boost::shared_ptr< std::vector< EChannelT > > getEventChannels() const;

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
    EChannelT& getEventChannel( int i ) const;

    /**
     * Returns number of event channels.
     */
    size_t getEventChannelCount() const;

    WLLifetimeProfiler::SPtr getProfiler();
    WLLifetimeProfiler::ConstSPtr getProfiler() const;
    void setProfiler( const WLLifetimeProfiler& profiler );

private:
    WLLifetimeProfiler::SPtr m_profiler;

    /**
     * experiment supervisor
     */
    std::string m_experimenter;

    /**
     * optional description of experiment
     */
    std::string m_expDescription;

    /**
     * list with modality specific measurements \ref WDataSetEMMEMD
     */
    std::vector< WLEMData::SPtr > m_modalityList;

    /**
     * subject information
     */
    LaBP::WLEMMSubject::SPtr m_subject;

    /**
     * Event/Stimuli channels
     */
    boost::shared_ptr< std::vector< EChannelT > > m_eventChannels;
};

#endif  // WLEMMEASUREMENT_H
