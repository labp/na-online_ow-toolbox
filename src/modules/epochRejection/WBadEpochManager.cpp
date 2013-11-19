/*
 * WBadEpochManager.cpp
 *
 *  Created on: 08.11.2013
 *      Author: maschke
 */

#include "WBadEpochManager.h"

WBadEpochManager::WBadEpochManager() :
                buffer( 5 )
{

}

WBadEpochManager::~WBadEpochManager()
{

}

WBadEpochManager *WBadEpochManager::instance()
{
    static WBadEpochManager _instance;

    return &_instance;
}

boost::circular_buffer< WBadEpoch >& WBadEpochManager::getBuffer()
{
    return buffer;
}
