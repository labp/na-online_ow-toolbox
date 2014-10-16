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

#ifndef WEPOCHAVERAGINGTOTAL_TEST_H
#define WEPOCHAVERAGINGTOTAL_TEST_H

#include <cstddef>
#include <vector>

#include <cxxtest/TestSuite.h>
#include <boost/shared_ptr.hpp>

#include "core/common/WLogger.h"

#include "core/data/WLEMMeasurement.h"
#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDEEG.h"

#include "../WEpochAveragingTotal.h"

#define EPS 0.0000001

class WEpochAveragingTotalTest: public CxxTest::TestSuite
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

        WEpochAveragingTotal::SPtr averager( new WEpochAveragingTotal( 0 ) );
        WLEMMeasurement::SPtr emm;
        WLEMData::SPtr emd;
        WLEMMeasurement::SPtr emmAverage;
        WLEMData::SPtr emdAverage;

        for( size_t i = 0; i < COUNT; ++i )
        {
            emm.reset( new WLEMMeasurement );
            emm->addModality( createEmd( 21, SAMPLES, i * SAMPLES ) );
            emm->addModality( createEmd( 42, SAMPLES, ( i + 1 ) * SAMPLES ) );

            emmAverage = averager->getAverage( emm );
            TS_ASSERT_EQUALS( emmAverage->getModalityCount(), emm->getModalityCount() );

            for( size_t mod = 0; mod < emmAverage->getModalityCount(); ++mod )
            {
                emd = emm->getModality( mod );
                emdAverage = emmAverage->getModality( mod );
                TS_ASSERT_EQUALS( emdAverage->getNrChans(), emd->getNrChans() );
                TS_ASSERT_EQUALS( emdAverage->getSamplesPerChan(), emd->getSamplesPerChan() );
                for( WLChanIdxT chan = 0; chan < emdAverage->getNrChans(); ++chan )
                {
                    for( WLSampleIdxT smp = 0; smp < emdAverage->getSamplesPerChan(); ++smp )
                    {
                        TS_ASSERT_DELTA( emdAverage->getData()( chan, smp ),
                                        getSum( i, ( i + mod ) * SAMPLES + smp, SAMPLES ) / ( i + 1 ), EPS );
                    }
                }
            }
        }
    }

private:
    WLEMData::SPtr createEmd( size_t channels, size_t samples, int startValue = 0 )
    {
        WLEMData::SPtr emd( new WLEMDEEG() );
        WLEMData::DataSPtr data( new WLEMData::DataT( channels, samples ) );

        for( size_t chan = 0; chan < channels; ++chan )
        {
            for( size_t smp = 0; smp < samples; ++smp )
            {
                ( *data )( chan, smp ) =  startValue + smp;
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

#endif  // WEPOCHAVERAGINGTOTAL_TEST_H
