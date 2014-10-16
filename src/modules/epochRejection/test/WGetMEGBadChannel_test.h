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

#ifndef WGETMEGBADCHANNEL_TEST_H_
#define WGETMEGBADCHANNEL_TEST_H_

#include <fstream>

#include <cxxtest/TestSuite.h>
#include <boost/foreach.hpp>

#include "core/common/WLogger.h"
#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDMEG.h"
#include "core/data/enum/WLEModality.h"
#include "core/data/WLEMMeasurement.h"

#include "WEpochRejectionTestHelper.h"

class WGetMEGBadChannel: public CxxTest::TestSuite
{
public:
    WGetMEGBadChannel()
    {
        this->m_helper.reset( new WEpochRejectionTestHelper() ); // create helper class
    }

    void setUp( void )
    {
        WLogger::startup();
    }

    void test_GetMEGNoBadChannels()
    {
        WLEMMeasurement::SPtr emm( new WLEMMeasurement() );
        WLEMDMEG::SPtr modalityMEG( new WLEMDMEG() );

        emm->addModality( m_helper->createModality( 300, 350, 0, modalityMEG ) );

        WLEMData::SPtr modalityFromEmm = emm->getModality( WLEModality::MEG );

        TS_ASSERT_EQUALS( modalityFromEmm->getData().rows(), modalityMEG->getData().rows() );
    }

    void test_GetMEGWithBadChannels()
    {
        WLEMMeasurement::SPtr emm( new WLEMMeasurement() );
        WLEMDMEG::SPtr modalityMEG( new WLEMDMEG() );
        WLEMData::ChannelListSPtr list( new WLEMData::ChannelList() );

        // G G M | G G M ...
        // 0 1 2 | 3 4 5 | 6 7 8 | 9 10 11 | 12 13 14
        // mark some channels as bad channels, in this test only magnetometer channels
        list->push_back( 2 );
        list->push_back( 5 );
        list->push_back( 8 );
        list->push_back( 11 );
        list->push_back( 14 );

        emm->addModality( m_helper->createModality( 300, 350, 0, modalityMEG ) );
        emm->getModality( WLEModality::MEG )->setBadChannels( list );

        WLEMData::SPtr modalityFromEmm = emm->getModality( WLEModality::MEG );
        WLEMDMEG::ConstSPtr meg = boost::static_pointer_cast< const WLEMDMEG >( modalityFromEmm );

        TS_ASSERT_EQUALS( meg->getBadChannels()->size(), 5 );
        TS_ASSERT_EQUALS( modalityFromEmm->getDataBadChannels()->rows(), 295 );
        TS_ASSERT_EQUALS( meg->getData().rows(), 300 );
        TS_ASSERT_EQUALS( meg->getData( WLEMEGGeneralCoilType::GRADIOMETER )->rows(), 200 );
        TS_ASSERT_EQUALS( meg->getData( WLEMEGGeneralCoilType::MAGNETOMETER )->rows(), 100 );

        TS_ASSERT_EQUALS( meg->isBadChannel( 2 ), true );
        TS_ASSERT_EQUALS( meg->isBadChannel( 5 ), true );
        TS_ASSERT_EQUALS( meg->isBadChannel( 12 ), false );
        TS_ASSERT_EQUALS( meg->isBadChannel( 14 ), true );
        TS_ASSERT_EQUALS( meg->isBadChannel( 9 ), false );
        TS_ASSERT_EQUALS( meg->getDataBadChannels( WLEMEGGeneralCoilType::GRADIOMETER )->rows(), 200 );
        TS_ASSERT_EQUALS( meg->getDataBadChannels( WLEMEGGeneralCoilType::MAGNETOMETER )->rows(), 95 );
    }

private:
    WEpochRejectionTestHelper::SPtr m_helper;
};

#endif  // WGETMEGBADCHANNEL_TEST_H_
