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

#ifndef WEPOCHRECJECTIONSINGLE_TEST_H_
#define WEPOCHRECJECTIONSINGLE_TEST_H_

#include <iostream>

#include <cxxtest/TestSuite.h>

#include "core/common/WLogger.h"
#include "core/data/enum/WLEModality.h"
#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDEEG.h"
#include "core/data/emd/WLEMDEOG.h"
#include "core/data/emd/WLEMDMEG.h"

#include "../WEpochRejectionSingle.h"
#include "../WEpochRejectionTotal.h"
#include "../WBadChannelManager.h"
#include "WEpochRejectionTestHelper.h"

#define THRESHOLD_EEG 150e-6
#define THRESHOLD_EOG 120e-6
#define THRESHOLD_MEG_GRAD 200e-12
#define THRESHOLD_MEG_MAG 5e-12

#define CLASS "WEpochRejectionSingleTest"

class WEpochRejectionSingleTest: public CxxTest::TestSuite
{

public:

    void setUp( void )
    {
        WLogger::startup();

        wlog::debug( CLASS ) << "startup";
    }

    WEpochRejectionSingleTest()
    {
        this->m_helper.reset( new WEpochRejectionTestHelper() ); // create helper class
    }

    void test_SingleChannelWithRejection()
    {
        WLEMMeasurement::SPtr emm( new WLEMMeasurement() );
        WEpochRejectionSingle::SPtr rejection( new WEpochRejectionSingle() );

        //rejection->setThresholds( THRESHOLD_EEG, THRESHOLD_EOG, THRESHOLD_MEG_GRAD, THRESHOLD_MEG_MAG );

        // prepare modalities for processing
        WLEMDEEG::SPtr modalityEEG( new WLEMDEEG() );
        WLEMDEOG::SPtr modalityEOG( new WLEMDEOG() );
        WLEMDMEG::SPtr modalityMEG( new WLEMDMEG() );
        emm->addModality( m_helper->createModality( 400, 350, 10, THRESHOLD_EEG, modalityEEG ) );
        emm->addModality( m_helper->createModality( 400, 350, 0, THRESHOLD_EOG, modalityEOG ) );
        emm->addModality( m_helper->createModality( 300, 350, 15, modalityMEG ) );

        // start rejection
        bool result = rejection->doRejection( emm );

        TS_ASSERT_EQUALS( result, true );
        TS_ASSERT_EQUALS( rejection->getCount(), 25 );
    }

    void test_SingleAndTotalRejection()
    {
        WEpochRejectionSingle::SPtr rejectSingle( new WEpochRejectionSingle() );
        WEpochRejectionTotal::SPtr rejectTotal( new WEpochRejectionTotal() );

        //rejectSingle->setThresholds( THRESHOLD_EEG, THRESHOLD_EOG, THRESHOLD_MEG_GRAD, THRESHOLD_MEG_MAG );
        //rejectTotal->setThresholds( THRESHOLD_EEG, THRESHOLD_EOG, THRESHOLD_MEG_GRAD, THRESHOLD_MEG_MAG );

        WLEMMeasurement::SPtr emm( new WLEMMeasurement() );

        // prepare modalities for processing
        WLEMDEEG::SPtr modalityEEG( new WLEMDEEG() );
        WLEMDEOG::SPtr modalityEOG( new WLEMDEOG() );
        WLEMDMEG::SPtr modalityMEG( new WLEMDMEG() );
        emm->addModality( m_helper->createModality( 400, 350, 10, THRESHOLD_EEG, modalityEEG ) );
        emm->addModality( m_helper->createModality( 400, 350, 0, THRESHOLD_EOG, modalityEOG ) );
        emm->addModality( m_helper->createModality( 300, 350, 15, modalityMEG ) );

        TS_ASSERT_EQUALS( rejectTotal->doRejection(emm), true ); // total rejection first
        TS_ASSERT_EQUALS( rejectSingle->doRejection(emm), true ); // single rejection with bad channel detection second

        // update the bad channels in the EMM object
        WBadChannelManager::instance()->reset();
        //WBadChannelManager::instance()->merge(rejectSingle->getRejectedMap());
        emm->getModality(WLEModality::EEG)->setBadChannels(WBadChannelManager::instance()->getChannelList(WLEModality::EEG));
        emm->getModality(WLEModality::MEG)->setBadChannels(WBadChannelManager::instance()->getChannelList(WLEModality::MEG));

        // test the number of bad channels after updating
        TS_ASSERT_EQUALS(emm->getModality(WLEModality::EEG)->getBadChannels()->size(), 10);
        TS_ASSERT_EQUALS(emm->getModality(WLEModality::EOG)->getBadChannels()->size(), 0);
        TS_ASSERT_EQUALS(emm->getModality(WLEModality::MEG)->getBadChannels()->size(), 15);

        // test total rejection again. Now their should be no problem, because all bad channels were rejected.
        TS_ASSERT_EQUALS( rejectTotal->doRejection(emm), false );
    }

private:
    WEpochRejectionTestHelper::SPtr m_helper;

};

#endif /* WEPOCHRECJECTIONSINGLE_TEST_H_ */
