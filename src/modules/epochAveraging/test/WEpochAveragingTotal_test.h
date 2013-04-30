#ifndef WAVERAGINGTOTALCPU_TEST_H
#define WAVERAGINGTOTALCPU_TEST_H

#include <cstddef>
#include <vector>

#include <cxxtest/TestSuite.h>
#include <boost/shared_ptr.hpp>

#include "core/common/WLogger.h"

#include "core/data/WLDataSetEMM.h"
#include "core/data/emd/WLEMD.h"
#include "core/data/emd/WLEMDEEG.h"

#include "../WEpochAveragingTotal.h"

#define EPS 0.0000001

class WAveragingTotalTest: public CxxTest::TestSuite
{
public:
    void setUp( void )
    {
        WLogger::startup();
    }

    void test_getAverage( void )
    {
        const size_t SAMPLES = 100;
        const size_t COUNT = 5;

        boost::shared_ptr< WEpochAveragingTotal > averager( new WEpochAveragingTotal( 0 ) );
        boost::shared_ptr< LaBP::WLDataSetEMM > emm;
        boost::shared_ptr< LaBP::WLEMD > emd;
        boost::shared_ptr< LaBP::WLDataSetEMM > emmAverage;
        boost::shared_ptr< LaBP::WLEMD > emdAverage;

        for( size_t i = 0; i < COUNT; ++i )
        {
            emm.reset( new LaBP::WLDataSetEMM() );
            emm->addModality( createEmd( 21, SAMPLES, i * SAMPLES ) );
            emm->addModality( createEmd( 42, SAMPLES, ( i + 1 ) * SAMPLES ) );

            emmAverage = averager->getAverage( emm );
            TS_ASSERT_EQUALS( emmAverage->getModalityCount(), emm->getModalityCount() );

            for( size_t mod = 0; mod < emmAverage->getModalityCount(); ++mod )
            {
                emd = emm->getModality( mod );
                emdAverage = emmAverage->getModality( mod );
                TS_ASSERT_EQUALS( emdAverage->getData().size(), emd->getData().size() )
                for( size_t chan = 0; chan < emdAverage->getData().size(); ++chan )
                {
                    TS_ASSERT_EQUALS( emdAverage->getData()[chan].size(), emd->getData()[chan].size() );
                    for( size_t smp = 0; smp < emdAverage->getData()[chan].size(); ++smp )
                    {
                        TS_ASSERT_DELTA( emdAverage->getData()[chan][smp],
                                        getSum( i, ( i + mod ) * SAMPLES + smp, SAMPLES ) / ( i + 1 ), EPS );
                    }
                }
            }
        }
    }

protected:

private:
    boost::shared_ptr< LaBP::WLEMD > createEmd( size_t channels, size_t samples, int startValue = 0 )
    {
        boost::shared_ptr< LaBP::WLEMD > emd( new LaBP::WLEMDEEG() );
        boost::shared_ptr< std::vector< std::vector< double > > > data( new std::vector< std::vector< double > > );

        for( size_t chan = 0; chan < channels; ++chan )
        {
            std::vector< double > channel;
            for( size_t smp = 0; smp < samples; ++smp )
            {
                channel.push_back( startValue + smp );
            }
            data->push_back( channel );
        }
        emd->setData( data );

        return emd;
    }

    double getSum( size_t i, size_t smp, size_t offset )
    {
        double result = 0;
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
