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

#ifndef WEPOCHREJECTION_TEST_H_
#define WEPOCHREJECTION_TEST_H_

#include <iostream>
#include <algorithm>

#include <cxxtest/TestSuite.h>

#include "core/common/WLogger.h"
#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDEEG.h"
#include "core/data/emd/WLEMDEOG.h"
#include "core/data/emd/WLEMDMEG.h"

#include "../WEpochRejectionTotal.h"
#include "WEpochRejectionTestHelper.h"

#define THRESHOLD_EEG 150e-6
#define THRESHOLD_EOG 120e-6
#define THRESHOLD_MEG_GRAD 200e-12
#define THRESHOLD_MEG_MAG 5e-12

#define REJECTION_FAKTOR 1.5

#define CLASS "WEpochRejectionTest"

typedef boost::shared_ptr< Eigen::MatrixXd > MatrixSPtr;

class WEpochRejectionTest: public CxxTest::TestSuite
{
public:

    WEpochRejectionTest()
    {
        this->m_helper.reset(new WEpochRejectionTestHelper()); // create helper class
    }

    void setUp( void )
    {
        WLogger::startup();

        wlog::debug( CLASS ) << "startup";
    }

    /**
     * Test a data set with one modality and some failures.
     */
    void test_totalChannelWithRejection( void )
    {
        WLEMMeasurement::SPtr emm( new WLEMMeasurement() );
        WEpochRejectionTotal::SPtr reject( new WEpochRejectionTotal() );
        reject->setThresholds( THRESHOLD_EEG, THRESHOLD_EOG, THRESHOLD_MEG_GRAD, THRESHOLD_MEG_MAG );

        WLEMDEEG::SPtr modality( new WLEMDEEG() );
        emm->addModality( m_helper->createModality( 10, 10, 2, THRESHOLD_EEG, modality ) );

        TS_ASSERT_EQUALS( true, reject->doRejection( emm ) );
        TS_ASSERT_EQUALS( 1, reject->getCount() );
    }

    /**
     * Test a data set with one modality and no failure.
     */
    void test_rejcetSingleModalityNoRejection( void )
    {
        WLEMMeasurement::SPtr emd( new WLEMMeasurement() );
        WEpochRejectionTotal *reject( new WEpochRejectionTotal() );

        // define thresholds
        reject->setThresholds( THRESHOLD_EEG, THRESHOLD_EOG, THRESHOLD_MEG_GRAD, THRESHOLD_MEG_MAG );

        // add a single modality to the data set
        WLEMDEEG::SPtr modality( new WLEMDEEG() );
        emd->addModality( m_helper->createModality( 10, 10, 0, THRESHOLD_EEG, modality ) );

        TS_ASSERT_EQUALS( false, reject->doRejection( emd ) );
        TS_ASSERT_EQUALS( 0, reject->getCount() );

    }

    /**
     * Test a data set with many modalities and some failures.
     */
    void test_rejcetMutipleModalitiesWithRejection( void )
    {
        WLEMMeasurement::SPtr emd( new WLEMMeasurement() );
        WEpochRejectionTotal *reject( new WEpochRejectionTotal() );

        // define thresholds
        reject->setThresholds( THRESHOLD_EEG, THRESHOLD_EOG, THRESHOLD_MEG_GRAD, THRESHOLD_MEG_MAG );

        WLEMDEEG::SPtr eegMod( new WLEMDEEG() );
        WLEMDEOG::SPtr eogMod( new WLEMDEOG() );
        WLEMDMEG::SPtr megMod( new WLEMDMEG() );
        emd->addModality( m_helper->createModality( 10, 10, 5, THRESHOLD_EEG, eegMod ) );
        emd->addModality( m_helper->createModality( 10, 10, 0, THRESHOLD_EOG, eogMod ) );
        emd->addModality( m_helper->createModality( 10, 10, 5, megMod ) );

        TS_ASSERT_EQUALS( true, reject->doRejection( emd ) );
        TS_ASSERT_EQUALS( 1, reject->getCount() );
    }

    /**
     * Test a data set with many modalities and no failure.
     */
    void test_rejcetMutipleModalitiesNoRejection( void )
    {
        WLEMMeasurement::SPtr emm( new WLEMMeasurement() );
        WEpochRejectionTotal *reject( new WEpochRejectionTotal() );

        // define thresholds
        reject->setThresholds( THRESHOLD_EEG, THRESHOLD_EOG, THRESHOLD_MEG_GRAD, THRESHOLD_MEG_MAG );

        WLEMDEEG::SPtr eegMod( new WLEMDEEG() );
        WLEMDEOG::SPtr eogMod( new WLEMDEOG() );
        WLEMDMEG::SPtr megMod( new WLEMDMEG() );
        emm->addModality( m_helper->createModality( 10, 10, 0, THRESHOLD_EEG, eegMod ) );
        emm->addModality( m_helper->createModality( 10, 10, 0, THRESHOLD_EOG, eogMod ) );
        emm->addModality( m_helper->createModality( 9, 10, 0, megMod ) ); // always a multiple of 3 -> else exception!!!

        TS_ASSERT_EQUALS( false, reject->doRejection( emm ) );
        TS_ASSERT_EQUALS( 0, reject->getCount() );
    }

private:

    WEpochRejectionTestHelper::SPtr m_helper;
};

#endif /* WEPOCHREJECTION_TEST_H_ */
