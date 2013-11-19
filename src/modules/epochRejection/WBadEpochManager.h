/*
 * WBadEpochManager.h
 *
 *  Created on: 08.11.2013
 *      Author: maschke
 */

#ifndef WBADEPOCHMANAGER_H_
#define WBADEPOCHMANAGER_H_

#include "boost/circular_buffer.hpp"

#include "WBadEpoch.h"

class WBadEpochManager
{
public:
    static WBadEpochManager *instance();

    boost::circular_buffer< WBadEpoch >& getBuffer();

private:
    WBadEpochManager();
    WBadEpochManager( const WBadEpochManager& );
    virtual ~WBadEpochManager();

    boost::circular_buffer< WBadEpoch > buffer;
};

#endif /* WBADEPOCHMANAGER_H_ */
