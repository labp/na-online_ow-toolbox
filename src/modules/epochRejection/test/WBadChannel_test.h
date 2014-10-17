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

#ifndef WBADCHANNEL_TEST_H_
#define WBADCHANNEL_TEST_H_

#include <cxxtest/TestSuite.h>

#include "core/common/WLogger.h"
#include "core/data/enum/WLEModality.h"

#include "modules/epochRejection/WBadChannelManager.h"

#define CLASS "WBadChannelTest"

class WBadChannelTest: public CxxTest::TestSuite
{
public:
    void setUp( void )
    {
        WLogger::startup();

        wlog::debug( CLASS ) << "startup";
    }

    void test_WBadChannelAdd()
    {
        TS_WARN( "Starting BadChannelTest -> Add-Test" );

        WBadChannelManager::instance()->reset(); // clear the manager

        // test manager on empty collection
        TS_ASSERT_EQUALS( 0, WBadChannelManager::instance()->countChannels() );

        // add some channels
        WBadChannelManager::instance()->addChannel( WLEModality::EEG, 2 );
        WBadChannelManager::instance()->addChannel( WLEModality::EEG, 1 );
        WBadChannelManager::instance()->addChannel( WLEModality::EEG, 20 );
        WBadChannelManager::instance()->addChannel( WLEModality::MEG, 5 );
        WBadChannelManager::instance()->addChannel( WLEModality::MEG, 6 );
        WBadChannelManager::instance()->addChannel( WLEModality::MEG, 1 );
        WBadChannelManager::instance()->addChannel( WLEModality::EOG, 2 );
        WBadChannelManager::instance()->addChannel( WLEModality::EOG, 8 );
        WBadChannelManager::instance()->addChannel( WLEModality::EOG, 5 );
        WBadChannelManager::instance()->addChannel( WLEModality::EOG, 15 );

        // test simple filled manager
        TS_ASSERT_EQUALS( 10, WBadChannelManager::instance()->countChannels() );
        TS_ASSERT_EQUALS( 3, WBadChannelManager::instance()->countChannels( WLEModality::EEG ) );
        TS_ASSERT_EQUALS( 3, WBadChannelManager::instance()->countChannels( WLEModality::MEG ) );
        TS_ASSERT_EQUALS( 4, WBadChannelManager::instance()->countChannels( WLEModality::EOG ) );

        // add a channel duplicate -> nothing should happen with the channel quantity.
        WBadChannelManager::instance()->addChannel( WLEModality::MEG, 5 );
        TS_ASSERT_EQUALS( 10, WBadChannelManager::instance()->countChannels() );
    }

    void test_WBadChannelRemove()
    {
        // test channel removing
        // channel not included -> nothing should happen with the channel quantity.
        WBadChannelManager::instance()->removeChannel( WLEModality::EOG, 100 );
        TS_ASSERT_EQUALS( 10, WBadChannelManager::instance()->countChannels() );

        // remove existing channel(s) and test new channel quantities
        WBadChannelManager::instance()->removeChannel( WLEModality::EEG, 2 );
        WBadChannelManager::instance()->removeChannel( WLEModality::EOG, 2 );
        TS_ASSERT_EQUALS( 8, WBadChannelManager::instance()->countChannels() );
        TS_ASSERT_EQUALS( 2, WBadChannelManager::instance()->countChannels( WLEModality::EEG ) );
        TS_ASSERT_EQUALS( 3, WBadChannelManager::instance()->countChannels( WLEModality::MEG ) );
        TS_ASSERT_EQUALS( 3, WBadChannelManager::instance()->countChannels( WLEModality::EOG ) );

        // test clearing the manager
        WBadChannelManager::instance()->reset();
        TS_ASSERT_EQUALS( 0, WBadChannelManager::instance()->countChannels() );
        TS_ASSERT_EQUALS( 0, WBadChannelManager::instance()->countChannels( WLEModality::EEG ) );
        TS_ASSERT_EQUALS( 0, WBadChannelManager::instance()->countChannels( WLEModality::MEG ) );
        TS_ASSERT_EQUALS( 0, WBadChannelManager::instance()->countChannels( WLEModality::EOG ) );
    }

    void test_WBadChannelMerge()
    {
        // test merging of various channel collections
        WBadChannelManager::ChannelMap_SPtr mapSPtr( new WBadChannelManager::ChannelMap() );
        WLEMData::ChannelListSPtr listEEG( new WLEMData::ChannelList() );
        listEEG->push_back( 1 );
        listEEG->push_back( 2 );
        listEEG->push_back( 3 );
        listEEG->push_back( 5 );
        mapSPtr->insert( WBadChannelManager::ChannelMap::value_type( WLEModality::EEG, listEEG ) );

        WBadChannelManager::instance()->merge( mapSPtr );

        listEEG->clear();
        mapSPtr->clear();

        TS_ASSERT_EQUALS( 4, WBadChannelManager::instance()->countChannels() );
        TS_ASSERT_EQUALS( 4, WBadChannelManager::instance()->countChannels( WLEModality::EEG ) );
        TS_ASSERT_EQUALS( 0, WBadChannelManager::instance()->countChannels( WLEModality::MEG ) );

        // add channels for one more modality type
        mapSPtr->clear();
        listEEG->clear();
        WLEMData::ChannelListSPtr listMEG( new WLEMData::ChannelList() );
        listMEG->push_back( 8 );
        listMEG->push_back( 13 );
        listEEG->push_back( 3 ); // add a duplicate channel
        mapSPtr->insert( WBadChannelManager::ChannelMap::value_type( WLEModality::EEG, listEEG ) );
        mapSPtr->insert( WBadChannelManager::ChannelMap::value_type( WLEModality::MEG, listMEG ) );

        WBadChannelManager::instance()->merge( mapSPtr );
        listEEG->clear();
        mapSPtr->clear();
        TS_ASSERT_EQUALS( 6, WBadChannelManager::instance()->countChannels() );
        TS_ASSERT_EQUALS( 4, WBadChannelManager::instance()->countChannels( WLEModality::EEG ) );
        TS_ASSERT_EQUALS( 2, WBadChannelManager::instance()->countChannels( WLEModality::MEG ) );

        // merge channel lists on the manager
        WLEMData::ChannelListSPtr listEOG( new WLEMData::ChannelList() );
        listEOG->push_back( 2 );
        listEOG->push_back( 3 );
        listEOG->push_back( 5 );
        listEEG->push_back( 500 );
        listEEG->push_back( 3 ); // add a duplicate channel
        listEEG->push_back( 50 );
        WBadChannelManager::instance()->merge( WLEModality::EOG, listEOG );
        WBadChannelManager::instance()->merge( WLEModality::EEG, listEEG );

        listEOG->clear();
        listEEG->clear();
        TS_ASSERT_EQUALS( 11, WBadChannelManager::instance()->countChannels() );
        TS_ASSERT_EQUALS( 3, WBadChannelManager::instance()->countChannels( WLEModality::EOG ) );
    }

    void test_WBadChannelGet()
    {
        WBadChannelManager::instance()->reset(); // clear collection

        // add a new channel list to the manager
        WLEMData::ChannelListSPtr listEEG( new WLEMData::ChannelList() );
        listEEG->push_back( 1 );
        listEEG->push_back( 2 );
        listEEG->push_back( 3 );
        listEEG->push_back( 5 );
        listEEG->push_back( 8 );
        WBadChannelManager::instance()->merge( WLEModality::EEG, listEEG );

        listEEG->clear(); // remove all elements from the list.

        WLEMData::ChannelListSPtr listToTest = WBadChannelManager::instance()->getChannelList( WLEModality::EEG ); // store the channels in a new list
        WBadChannelManager::ChannelMap_SPtr mapToTest = WBadChannelManager::instance()->getChannelMap();

        WBadChannelManager::instance()->reset(); // clear the manager

        TS_ASSERT_EQUALS( 5, listToTest->size() );
        TS_ASSERT_EQUALS( false, mapToTest->empty() );
        TS_ASSERT_EQUALS( 0, WBadChannelManager::instance()->countChannels() );
        TS_ASSERT_EQUALS( true, WBadChannelManager::instance()->isMapEmpty() );
    }
};

#endif  // WBADCHANNEL_TEST_H_
