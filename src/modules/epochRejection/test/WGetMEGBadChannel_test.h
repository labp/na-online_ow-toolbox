/*
 * WGetMEGBadChannel_test.h
 *
 *  Created on: 31.01.2014
 *      Author: maschke
 */

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
#include "core/data/WLEMMEnumTypes.h"

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
        TS_ASSERT_EQUALS( meg->getData( LaBP::WEGeneralCoilType::GRADIOMETER )->rows(), 200 );
        TS_ASSERT_EQUALS( meg->getData( LaBP::WEGeneralCoilType::MAGNETOMETER )->rows(), 100 );

        TS_ASSERT_EQUALS( meg->isBadChannel(2), true );
        TS_ASSERT_EQUALS( meg->isBadChannel(5), true );
        TS_ASSERT_EQUALS( meg->isBadChannel(12), false );
        TS_ASSERT_EQUALS( meg->isBadChannel(14), true );
        TS_ASSERT_EQUALS( meg->isBadChannel(9), false );
        TS_ASSERT_EQUALS( meg->getDataBadChannels( LaBP::WEGeneralCoilType::GRADIOMETER )->rows(), 200 );
        TS_ASSERT_EQUALS( meg->getDataBadChannels( LaBP::WEGeneralCoilType::MAGNETOMETER )->rows(), 95 );
    }

private:
    WEpochRejectionTestHelper::SPtr m_helper;
};

#endif /* WGETMEGBADCHANNEL_TEST_H_ */
