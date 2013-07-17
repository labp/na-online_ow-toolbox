#ifndef WAVERAGINGTOTALCPU_TEST_H
#define WAVERAGINGTOTALCPU_TEST_H

#include <algorithm>
#include <cstddef>
#include <vector>

#include <cxxtest/TestSuite.h>
#include <boost/shared_ptr.hpp>

#include "core/common/WLogger.h"

#include "core/data/WLEMMeasurement.h"
#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDEEG.h"

#include "../WEpochAveragingMoving.h"

#define EPS 0.0000001

class WAveragingMovingTest: public CxxTest::TestSuite
{
public:
    void setUp( void )
    {
        WLogger::startup();
    }

    void test_setGetReset( void )
    {
        WEpochAveragingMoving::SPtr averager( new WEpochAveragingMoving( 0, 42 ) );
        TS_ASSERT_EQUALS( 0, averager->getCount() );
        TS_ASSERT_EQUALS( 42, averager->getSize() );

        averager->setSize( 10 );
        TS_ASSERT_EQUALS( 10, averager->getSize() );
        TS_ASSERT_EQUALS( 0, averager->getCount() );

        averager->reset();
        TS_ASSERT_EQUALS( 0, averager->getCount() );
    }

    void test_getAverage( void )
    {
        const size_t SAMPLES = 100;
        const size_t COUNT = 20;

        WEpochAveragingMoving::SPtr averager( new WEpochAveragingMoving( 0, COUNT / 3 ) );
        WLEMMeasurement::SPtr emm;
        WLEMData::SPtr emd;
        WLEMMeasurement::SPtr emmAverage;
        WLEMData::SPtr emdAverage;

        for( size_t i = 0; i < COUNT; ++i )
        {
            emm.reset( new WLEMMeasurement() );
            emm->addModality( createEmd( 23, SAMPLES, i * SAMPLES ) );
            emm->addModality( createEmd( 42, SAMPLES, ( i + 1 ) * SAMPLES ) );

            emmAverage = averager->getAverage( emm );
            TS_ASSERT_EQUALS( emmAverage->getModalityCount(), emm->getModalityCount() );
            for( size_t mod = 0; mod < emmAverage->getModalityCount(); ++mod )
            {
                emd = emm->getModality( mod );
                emdAverage = emmAverage->getModality( mod );
                TS_ASSERT_EQUALS( emdAverage->getNrChans(), emd->getNrChans() );
                TS_ASSERT_EQUALS( emdAverage->getSamplesPerChan(), emd->getSamplesPerChan() );
                for( size_t chan = 0; chan < emdAverage->getNrChans(); ++chan )
                {
                    for( size_t smp = 0; smp < emdAverage->getSamplesPerChan(); ++smp )
                    {
                        TS_ASSERT_DELTA( emdAverage->getData()( chan, smp ),
                                        getSum( std::min( i, averager->getSize() - 1 ), ( i + mod ) * SAMPLES + smp, SAMPLES )
                                                        / std::min( i + 1, averager->getSize() ), EPS );
                    }
                }
            }
        }

        // correct result after reset the buffer?
        averager->setSize( COUNT );
        TS_ASSERT_EQUALS( 0, averager->getCount() );

        for( size_t i = 0; i < std::min( COUNT, ( size_t )2 ); ++i )
        {
            emm.reset( new WLEMMeasurement() );
            emm->addModality( createEmd( 23, SAMPLES, i * SAMPLES ) );
            emm->addModality( createEmd( 42, SAMPLES, ( i + 1 ) * SAMPLES ) );

            emmAverage = averager->getAverage( emm );
            TS_ASSERT_EQUALS( emmAverage->getModalityCount(), emm->getModalityCount() );
            for( size_t mod = 0; mod < emmAverage->getModalityCount(); ++mod )
            {
                emd = emm->getModality( mod );
                emdAverage = emmAverage->getModality( mod );
                TS_ASSERT_EQUALS( emdAverage->getNrChans(), emd->getNrChans() );
                TS_ASSERT_EQUALS( emdAverage->getSamplesPerChan(), emd->getSamplesPerChan() );
                for( size_t chan = 0; chan < emdAverage->getNrChans(); ++chan )
                {
                    for( size_t smp = 0; smp < emdAverage->getSamplesPerChan(); ++smp )
                    {
                        TS_ASSERT_DELTA( emdAverage->getData()( chan, smp ),
                                        getSum( std::min( i, averager->getSize() - 1 ), ( i + mod ) * SAMPLES + smp, SAMPLES )
                                                        / std::min( i + 1, averager->getSize() ), EPS );
                    }
                }
            }
        }
    }

protected:

private:
    WLEMData::SPtr createEmd( size_t channels, size_t samples, int startValue = 0 )
    {
        WLEMData::SPtr emd( new WLEMDEEG() );
        WLEMData::DataSPtr data( new WLEMData::DataT( channels, samples ) );

        for( size_t chan = 0; chan < channels; ++chan )
        {
            for( size_t smp = 0; smp < samples; ++smp )
            {
                ( *data )( chan, smp ) = startValue + smp;
            }
        }
        emd->setData( data );

        return emd;
    }

    WLEMData::ScalarT getSum( size_t i, size_t smp, size_t offset )
    {
        WLEMData::ScalarT result = 0;
        result += smp;
        for( size_t ii = i; 0 < ii; --ii )
        {
            smp -= offset;
            result += smp;
        }
        return result;
    }

};

#endif // WAVERAGINGTOTALCPU_TEST_H
