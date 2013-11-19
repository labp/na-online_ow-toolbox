/*
 * WBadChannelManager.h
 *
 *  Created on: 08.11.2013
 *      Author: maschke
 */

#ifndef WBADCHANNELMANAGER_H_
#define WBADCHANNELMANAGER_H_

#include "core/data/WLEMMeasurement.h"
#include "core/data/WLEMMEnumTypes.h"

#include "WGenericList.h"
#include "WGenericMap.h"

class WBadChannelManager: public WGenericMap< LaBP::WEModalityType::Enum, WGenericList< size_t > >
{
public:

    /**
     * Returns a static pointer on a WBadChannelManager object. This pointer can be used to do operations on the inherited map.
     * @return A pointer on a WBadChannelManager object.
     */
    static WBadChannelManager *instance();

    /**
     * This method inserts a certain channel number to the channel collection.
     * @param The modality as key.
     * @param The channel number to insert.
     */
    void addElement( const LaBP::WEModalityType::Enum&, const size_t& );

    /**
     * This method removes a certain channel number form the collection.
     * @param The modality as key.
     * @param The channel number to remove.
     */
    void removeAt( const LaBP::WEModalityType::Enum&, const size_t& );

private:
    WBadChannelManager();
    WBadChannelManager( const WBadChannelManager& );
    ~WBadChannelManager();
};

#endif /* WBADCHANNELMANAGER_H_ */
