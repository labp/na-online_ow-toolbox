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

#ifndef WBADCHANNELMANAGER_H_
#define WBADCHANNELMANAGER_H_

#include <list>
#include <map>

#include <boost/shared_ptr.hpp>

#include "core/data/emd/WLEMData.h"
#include "core/data/WLEMMeasurement.h"
#include "core/data/enum/WLEModality.h"

/**
 * The WBadChannelManager is used to store channel numbers distinguished by their modality types.
 * The class is designed as singleton pattern for accessing them from all of the module.
 */
class WBadChannelManager
{
public:
    /**
     * A std::map<K,V> with a WLEModality::Enum as key and boost shared pointer on an ChannelList as value.
     */
    typedef std::map< WLEModality::Enum, WLEMData::ChannelListSPtr > ChannelMap;

    /**
     * A constant ChannelMap.
     */
    typedef const ChannelMap ConstChannelMap;

    /**
     * A boost shared pointer on an ChannelMap.
     */
    typedef boost::shared_ptr< ChannelMap > ChannelMap_SPtr;

    /**
     * A boost shared pointer on an constant ChannelMap.
     */
    typedef boost::shared_ptr< ConstChannelMap > ChannelMap_ConstSPtr;

    /**
     * Returns a static pointer on a WBadChannelManager object. This pointer can be used to do operations on the inherited map.
     *
     * \return A pointer on a WBadChannelManager object.
     */
    static WBadChannelManager *instance();

    /**
     * This method inserts a certain channel number to the channel collection.
     *
     * \param The modality as key.
     * \param The channel number to insert.
     */
    void addChannel( const WLEModality::Enum&, const size_t& );

    /**
     * This method removes a certain channel number form the collection.
     *
     * @param The modality as key.
     * @param The channel number to remove.
     */
    void removeChannel( const WLEModality::Enum&, const size_t& );

    /**
     * Gets true if the ChannelMap is empty, else it returns false.
     *
     * \return True / false.
     */
    bool isMapEmpty() const;

    /**
     * Gets true if the channel is a bad channel, else it returns false.
     *
     * \param The modality.
     * \param The channel number.
     * \return True / false.
     */
    bool isChannelBad( const WLEModality::Enum&, const size_t ) const;

    /**
     * Gets true if the modality has bad channels, else it returns false.
     *
     * \param The Modality.
     * \return True / false.
     */
    bool hasBadChannels( const WLEModality::Enum& ) const;

    /**
     * Returns the number of all bad channels.
     *
     * \return Number of all bad channels.
     */
    size_t countChannels() const;

    /**
     * Returns the number of bad channels for the modality.
     *
     * \param The modality.
     * \return The number of bad channels.
     */
    size_t countChannels( const WLEModality::Enum& ) const;

    /**
     * This method returns a boost shared pointer on a ChannelList for the certain modality.
     * If there was not ChannelList in the collection, the method returns an empty pointer
     * (equals '0').
     * Note: The list is only a copy.
     *
     * \param The modality.
     * \return The ChannelList-pointer.
     */
    WLEMData::ChannelListSPtr getChannelList( const WLEModality::Enum& );

    /**
     * This method returns the whole bad channel map as shared pointer.
     * Note: The map is only a copy.
     *
     * \return The ChannelMap pointer.
     */
    WBadChannelManager::ChannelMap_SPtr getChannelMap();

    /**
     * This method merges the given ChannelMap on the managers current ChannelMap.
     * In doing so the manager maps internal the ChannelLists on the maps.
     *
     * \param The ChannelMap
     */
    void merge( ChannelMap_SPtr );

    /**
     * This method merges the given ChannelList on the managers current ChannelMap for the modality.
     * In doing so the manager maps internal the ChannelLists on the maps.
     *
     * \param The modality.
     * \param The channel list.
     */
    void merge( const WLEModality::Enum&, WLEMData::ChannelListSPtr );

    /**
     * This method clears the collections of the channel map.
     */
    void reset();

protected:
    /**
     * A boost shared pointer on a std::map with a LaBP::WEModalityType::Enum as key and ChannelList as value type.
     */
    ChannelMap_SPtr m_map;

private:
    /**
     * The private constructor initialize the ChannelMap member variable. It is hidden to realize the singleton pattern.
     */
    WBadChannelManager();

    /**
     * The private copy constructor is hidden to realize the singleton pattern.
     *
     * \param Class instance.
     */
    WBadChannelManager( const WBadChannelManager& );

    /**
     * The private destructor.
     */
    ~WBadChannelManager();

    /**
     * The static class instance for singleton.
     */
    static WBadChannelManager *m_instance;
};

#endif  // WBADCHANNELMANAGER_H_
