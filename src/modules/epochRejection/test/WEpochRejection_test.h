/*
 * WEpochRejection_test.h
 *
 *  Created on: 24.07.2013
 *      Author: maschke
 */

#ifndef WEPOCHREJECTION_TEST_H_
#define WEPOCHREJECTION_TEST_H_

#include <cxxtest/TestSuite.h>

#include "core/common/WLogger.h"
#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDEEG.h"
#include "core/data/emd/WLEMDEOG.h"
#include "core/data/emd/WLEMDMEG.h"

#include "../WEpochRejectionSingle.h"

#define THRESHOLD_EEG 150e-6
#define THRESHOLD_EOG 120e-6
#define THRESHOLD_MEG_GRAD 200e-12
#define THRESHOLD_MEG_MAG 5e-12

#define CLASS "WEpochRejectionTest"

typedef boost::shared_ptr< Eigen::MatrixXd > MatrixSPtr;

class WEpochRejectionTest: public CxxTest::TestSuite
{
public:
    void setUp( void )
    {
        WLogger::startup();
    }

    /**
     * Test a data set with one modality and some failures.
     */
    void test_rejcetSingleModalityRejection( void )
    {
        WLEMMeasurement::SPtr emd( new WLEMMeasurement() );
        WEpochRejectionSingle *reject( new WEpochRejectionSingle() );
        reject->setThresholds( THRESHOLD_EEG, THRESHOLD_EOG, THRESHOLD_MEG_GRAD, THRESHOLD_MEG_MAG );

        WLEMDEEG::SPtr modality( new WLEMDEEG() );
        emd->addModality( createModality( 10, 10, 2, modality ) );

        TS_ASSERT_EQUALS( true, reject->doRejection( emd ) );
        TS_ASSERT_EQUALS( 1, reject->getCount() );
    }

    /**
     * Test a data set with one modality and no failure.
     */
    void test_rejcetSingleModalityNoRejection( void )
    {
        WLEMMeasurement::SPtr emd( new WLEMMeasurement() );
        WEpochRejectionSingle *reject( new WEpochRejectionSingle() );

        // define thresholds
        reject->setThresholds( THRESHOLD_EEG, THRESHOLD_EOG, THRESHOLD_MEG_GRAD, THRESHOLD_MEG_MAG );

        // add a single modality to the data set
        WLEMDEEG::SPtr modality( new WLEMDEEG() );
        emd->addModality( createModality( 10, 10, 0, modality ) );

        TS_ASSERT_EQUALS( false, reject->doRejection( emd ) );
        TS_ASSERT_EQUALS( 0, reject->getCount() );

    }

    /**
     * Test a data set with many modalities and some failures.
     */
    void test_rejcetMutipleModalitiesWithRejection( void )
    {
        WLEMMeasurement::SPtr emd( new WLEMMeasurement() );
        WEpochRejectionSingle *reject( new WEpochRejectionSingle() );

        // define thresholds
        reject->setThresholds( THRESHOLD_EEG, THRESHOLD_EOG, THRESHOLD_MEG_GRAD, THRESHOLD_MEG_MAG );

        WLEMDEEG::SPtr eegMod( new WLEMDEEG() );
        WLEMDEOG::SPtr eogMod( new WLEMDEOG() );
        WLEMDMEG::SPtr megMod( new WLEMDMEG() );
        emd->addModality( createModality( 10, 10, 5, eegMod ) );
        emd->addModality( createModality( 10, 10, 0, eogMod ) );
        emd->addModality( createModality( 10, 10, 5, megMod ) );

        TS_ASSERT_EQUALS( true, reject->doRejection( emd ) );
        TS_ASSERT_EQUALS( 2, reject->getCount() );
    }

    /**
     * Test a data set with many modalities and no failure.
     */
    void test_rejcetMutipleModalitiesNoRejection( void )
    {
        WLEMMeasurement::SPtr emd( new WLEMMeasurement() );
        WEpochRejectionSingle *reject( new WEpochRejectionSingle() );

        // define thresholds
        reject->setThresholds( THRESHOLD_EEG, THRESHOLD_EOG, THRESHOLD_MEG_GRAD, THRESHOLD_MEG_MAG );

        WLEMDEEG::SPtr eegMod( new WLEMDEEG() );
        WLEMDEOG::SPtr eogMod( new WLEMDEOG() );
        WLEMDMEG::SPtr megMod( new WLEMDMEG() );
        emd->addModality( createModality( 10, 10, 0, eegMod ) );
        emd->addModality( createModality( 10, 10, 0, eogMod ) );
        emd->addModality( createModality( 10, 10, 0, megMod ) );

        TS_ASSERT_EQUALS( false, reject->doRejection( emd ) );
        TS_ASSERT_EQUALS( 0, reject->getCount() );
    }

private:

    /**
     * Method to create an EEG modality.
     */
    WLEMData::SPtr createModality( size_t channels, size_t samples, size_t rejections, WLEMDEEG::SPtr emd )
    {
        wlog::debug( CLASS ) << emd->getModalityType();

        double succValue = ( ( 9 / 10 ) * THRESHOLD_EEG ) / 2; // value for no rejection
        double failValue = 2 * THRESHOLD_EEG; // value for rejection
        Eigen::MatrixXd matrix( channels, samples );

        for( size_t i = 0; i < channels; ++i ) // create the channels
        {
            for( size_t j = 0; j < samples; ++j ) // create the samples
            {
                if( j % 2 == 0 ) // even/odd number
                    if( rejections > 0 ) // insert an invalid value
                    {
                        matrix( i, j ) = failValue;
                        --rejections;
                    }
                    else
                        matrix( i, j ) = succValue;
                else
                    matrix( i, j ) = -succValue;

            }
        }

        // insert the data into the emd
        MatrixSPtr p( new Eigen::MatrixXd( matrix ) );
        emd->setData( p );

        return emd;
    }

    /**
     * Method to create an EOG modality.
     */
    WLEMData::SPtr createModality( size_t channels, size_t samples, size_t rejections, WLEMDEOG::SPtr emd )
    {
        double succValue = ( ( 9 / 10 ) * THRESHOLD_EOG ) / 2;
        double failValue = 2 * THRESHOLD_EOG;
        Eigen::MatrixXd matrix( channels, samples );

        for( size_t i = 0; i < channels; ++i ) // create the channels
        {
            for( size_t j = 0; j < samples; ++j ) // create the samples
            {
                if( j % 2 == 0 ) // even/odd number
                    if( rejections > 0 ) // insert an invalid value
                    {
                        matrix( i, j ) = failValue;
                        --rejections;
                    }
                    else
                        matrix( i, j ) = succValue;
                else
                    matrix( i, j ) = -succValue;

            }
        }

        // insert the data into the emd
        MatrixSPtr p( new Eigen::MatrixXd( matrix ) );
        emd->setData( p );

        return emd;
    }

    /**
     * Method to create a MEG modality.
     */
    WLEMData::SPtr createModality( size_t channels, size_t samples, size_t rejections, WLEMDMEG::SPtr emd )
    {
        wlog::debug( CLASS ) << emd->getModalityType();

        double succValueGrad = ( ( 9 / 10 ) * THRESHOLD_MEG_GRAD ) / 2; // value for no rejection
        double failValueGrad = 2 * THRESHOLD_MEG_GRAD; // value for rejection
        double succValueMag = ( ( 9 / 10 ) * THRESHOLD_MEG_MAG ) / 2; // value for no rejection
        double failValueMag = 2 * THRESHOLD_MEG_MAG; // value for rejection
        size_t channelCount = 0;
        Eigen::MatrixXd matrix( channels, samples );

        for( size_t i = 0; i < channels; ++i ) // create the channels
        {
            ++channelCount;

            double succValue;
            double failValue;

            // differ the channel by magmetometer and gradiometer
            if( channelCount % 3 == 0 ) // magnetometer
            {
                succValue = succValueMag;
                failValue = failValueMag;
            }
            else // gradiometer
            {
                succValue = succValueGrad;
                failValue = failValueGrad;
            }

            // create the samples
            for( size_t j = 0; j < samples; ++j )
            {
                if( j % 2 == 0 ) // even/odd number of iteration
                    if( rejections > 0 ) // insert an invalid value
                    {
                        matrix( i, j ) = failValue;
                        --rejections;
                    }
                    else
                        matrix( i, j ) = succValue;
                else
                    matrix( i, j ) = -succValue;

            }

        }

        // insert the data into the emd
        MatrixSPtr p( new Eigen::MatrixXd( matrix ) );
        emd->setData( p );

        return emd;
    }
};

#endif /* WEPOCHREJECTION_TEST_H_ */
