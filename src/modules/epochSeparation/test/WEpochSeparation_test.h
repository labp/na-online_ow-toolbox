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

#ifndef WEPOCHSEPARATION_TEST_H
#define WEPOCHSEPARATION_TEST_H

#include <cstddef>
#include <list>
#include <set>

#include <cxxtest/TestSuite.h>
#include <boost/shared_ptr.hpp>

#include <core/common/WLogger.h>

#include "core/data/WLEMMeasurement.h"
#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDEEG.h"

#include "../WEpochSeparation.h"

using std::set;

class WEpochSeparationTest: public CxxTest::TestSuite
{
public:
    void setUp( void )
    {
        WLogger::startup();
    }

    void test_setGetReset( void )
    {
        TS_TRACE( "test_setGetReset ..." );

        const WLSampleNrT PRESAMPLES = 42;
        const WLSampleNrT POSTSAMPLES = 23;
        const WLSampleNrT ECHANNEL = 2;
        set< WLEMMeasurement::EventT > triggers;
        triggers.insert( 50 );

        WEpochSeparation::SPtr separation( new WEpochSeparation( ECHANNEL, triggers, PRESAMPLES, POSTSAMPLES ) );
        TS_ASSERT_EQUALS( ECHANNEL, separation->getChannel() );
        TS_ASSERT_EQUALS( 1, separation->getTriggerMask().size() );
        TS_ASSERT_EQUALS( PRESAMPLES, separation->getPreSamples() );
        TS_ASSERT_EQUALS( POSTSAMPLES, separation->getPostSamples() );

        separation->reset();
        TS_ASSERT_EQUALS( 0, separation->getChannel() );
        TS_ASSERT_EQUALS( 0, separation->getTriggerMask().size() );
        TS_ASSERT_EQUALS( 1, separation->getPreSamples() );
        TS_ASSERT_EQUALS( 1, separation->getPostSamples() );
        TS_ASSERT_EQUALS( false, separation->hasEpochs() );

        separation->setChannel( ECHANNEL );
        separation->setTriggerMask( triggers );
        separation->setPreSamples( PRESAMPLES );
        separation->setPostSamples( POSTSAMPLES );
        TS_ASSERT_EQUALS( ECHANNEL, separation->getChannel() );
        TS_ASSERT_EQUALS( 1, separation->getTriggerMask().size() );
        TS_ASSERT_EQUALS( PRESAMPLES, separation->getPreSamples() );
        TS_ASSERT_EQUALS( POSTSAMPLES, separation->getPostSamples() );
    }

    void test_extractSinglePacketNoModalityNoEventChannel( void )
    {
        TS_TRACE( "test_extractSinglePacketNoModalityNoEventChannel ..." );

        WLEMMeasurement::SPtr emm( new WLEMMeasurement() );
        const size_t SAMPLES = 10;
        emm->addModality( createEmd( 21, SAMPLES, 0 * SAMPLES ) );
        emm->addModality( createEmd( 42, SAMPLES, 1 * SAMPLES ) );

        // Single packet must be bigger than (postSamles + preSamles + 1), because WEpochSeparation returns true only for a full collected packet
        set< WLEMMeasurement::EventT > triggers;
        triggers.insert( 1 );
        WEpochSeparation::SPtr separation( new WEpochSeparation( 1, triggers, SAMPLES / 3, SAMPLES / 3 ) );
        // no event channel
        TS_ASSERT_EQUALS( separation->extract( emm ), 0 );

        separation.reset( new WEpochSeparation( 1, triggers, SAMPLES / 3, SAMPLES / 3 ) );
        emm.reset( new WLEMMeasurement() );
        WLEMMeasurement::EChannelT eChannel( SAMPLES, 1 );
        emm->addEventChannel( eChannel );
        // no modality
        TS_ASSERT_EQUALS( separation->extract( emm ), 0 );
    }

    void test_extractSinglePacketCheckMaskCheckChannel( void )
    {
        TS_TRACE( "test_extractSinglePacketCheckMaskCheckChannel ..." );

        const WLSampleNrT SAMPLES = 500;
        const WLSampleNrT PRESAMPLES = SAMPLES / 3;
        const WLSampleNrT POSTSAMPLES = SAMPLES / 3;
        const WLEMMeasurement::EventT TRIGGER_MATCH = 4;
        set< WLEMMeasurement::EventT > triggersMatch;
        triggersMatch.insert( TRIGGER_MATCH + 5 );
        triggersMatch.insert( TRIGGER_MATCH );
        const WLEMMeasurement::EventT TRIGGER_NO_MATCH = 1;
        set< WLEMMeasurement::EventT > triggersNoMatch;
        triggersNoMatch.insert( TRIGGER_NO_MATCH );
        const size_t EINDEX = SAMPLES / 2;

        WLEMMeasurement::SPtr emm( new WLEMMeasurement() );
        emm->addModality( createEmd( 21, SAMPLES, 0 * SAMPLES ) );
        emm->addModality( createEmd( 42, SAMPLES, 1 * SAMPLES ) );

        WLEMMeasurement::EChannelT eChannel1( SAMPLES, TRIGGER_NO_MATCH );
        emm->addEventChannel( eChannel1 );
        WLEMMeasurement::EChannelT eChannel2( SAMPLES, 0 );
        eChannel2[EINDEX] = TRIGGER_MATCH;
        emm->addEventChannel( eChannel2 );

        // Single packet must be bigger than (postSamles + preSamles + 1), because WEpochSeparation returns true only for a full collected packet
        WEpochSeparation::SPtr separation( new WEpochSeparation( 1, triggersNoMatch, PRESAMPLES, POSTSAMPLES ) );
        TS_ASSERT_EQUALS( separation->extract( emm ), 0 );
        // event channel 0 is ignored and event channel 1 has a different mask

        separation.reset( new WEpochSeparation( 1, triggersMatch, PRESAMPLES, POSTSAMPLES ) );
        // should have an epoch
        TS_ASSERT_LESS_THAN( 0, separation->extract( emm ) );
        TS_ASSERT( separation->hasEpochs() );
        separation->getNextEpoch();
        TS_ASSERT( !separation->hasEpochs() );

        // should not match, because trigger needs more pre samples.
        separation.reset( new WEpochSeparation( 1, triggersMatch, SAMPLES + 1, SAMPLES + 1 ) );
        TS_ASSERT_EQUALS( 0, separation->extract( emm ) );
    }

    void test_extractSinglePacketCheckData( void )
    {
        TS_TRACE( "test_extractSinglePacketCheckData ..." );

        const WLSampleNrT SAMPLES = 500;
        const WLSampleNrT PRESAMPLES = SAMPLES / 3;
        const WLSampleNrT POSTSAMPLES = SAMPLES / 3;
        const WLEMMeasurement::EventT TRIGGER_MATCH = 4;
        set< WLEMMeasurement::EventT > triggersMatch;
        triggersMatch.insert( TRIGGER_MATCH + 42 );
        triggersMatch.insert( TRIGGER_MATCH );
        const size_t EINDEX = SAMPLES / 2;

        WLEMMeasurement::SPtr emm( new WLEMMeasurement() );
        emm->addModality( createEmd( 21, SAMPLES, 0 * SAMPLES ) );
        emm->addModality( createEmd( 42, SAMPLES, 1 * SAMPLES ) );
        WLEMMeasurement::EChannelT eChannel1( SAMPLES, 1 );
        emm->addEventChannel( eChannel1 );
        WLEMMeasurement::EChannelT eChannel2( SAMPLES, 0 );
        eChannel2[EINDEX] = TRIGGER_MATCH;
        emm->addEventChannel( eChannel2 );

        WEpochSeparation::SPtr separation( new WEpochSeparation( 1, triggersMatch, PRESAMPLES, POSTSAMPLES ) );
        TS_ASSERT( separation->extract( emm ) );
        // should match
        WLEMMeasurement::SPtr emmEpoch = separation->getNextEpoch();

        // check sizes
        TS_ASSERT_EQUALS( emmEpoch->getModalityCount(), emm->getModalityCount() );
        // check modality count
        for( size_t mod = 0; mod < emm->getModalityCount(); ++mod )
        {
            TS_ASSERT_EQUALS( emmEpoch->getModality( mod )->getNrChans(), emm->getModality( mod )->getNrChans() );
            // check channel size
            TS_ASSERT_EQUALS( emmEpoch->getModality( mod )->getSamplesPerChan(),
                            separation->getPreSamples() + separation->getPostSamples() + 1 );
            // check samples size: post + pre + 1
        }

        // check data
        WLEMData::SPtr emdEpoch;
        double startValue;
        for( size_t mod = 0; mod < emmEpoch->getModalityCount(); ++mod )
        {
            emdEpoch = emmEpoch->getModality( mod );
            startValue = mod * SAMPLES + ( EINDEX - PRESAMPLES );
            for( WLChanIdxT chan = 0; chan < emdEpoch->getNrChans(); ++chan )
            {
                for( WLSampleIdxT smp = 0; smp < ( PRESAMPLES + POSTSAMPLES + 1 ); ++smp )
                {
                    TS_ASSERT_EQUALS( emdEpoch->getData()( chan, smp ), startValue + smp );
                }
            }
        }

        // check event channels
        TS_ASSERT_EQUALS( emmEpoch->getEventChannelCount(), 1 );
        WLEMMeasurement::EChannelT eChannel;
        WLEMMeasurement::EChannelT eChannelOrg;
        size_t startIndex = EINDEX - PRESAMPLES;

        eChannel = emmEpoch->getEventChannel( 0 );
        eChannelOrg = emm->getEventChannel( 1 );
        TS_ASSERT_EQUALS( eChannel.size(), ( PRESAMPLES + POSTSAMPLES + 1 ) );
        for( size_t smp = 0; smp < ( PRESAMPLES + POSTSAMPLES + 1 ); ++smp )
        {
            TS_ASSERT_EQUALS( eChannel[smp], eChannelOrg[startIndex + smp] );
        }
    }

    void test_extractMultiPacketCheckData( void )
    {
        TS_TRACE( "test_extractMultiPacketCheckData ..." );

        const WLSampleNrT BLOCKSIZE = 250;
        const WLSampleNrT SAMPLES = BLOCKSIZE * 20;
        const WLSampleNrT PRESAMPLES = 3 * BLOCKSIZE;
        const WLSampleNrT POSTSAMPLES = 2 * BLOCKSIZE;
        const WLEMMeasurement::EventT TRIGGER_MATCH = 4;
        set< WLEMMeasurement::EventT > triggersMatch;
        triggersMatch.insert( TRIGGER_MATCH + 21 );
        triggersMatch.insert( TRIGGER_MATCH );
        const size_t EINDEX = SAMPLES - ( POSTSAMPLES + 1.5 * BLOCKSIZE );

        WLEMMeasurement::SPtr emm( new WLEMMeasurement() );
        emm->addModality( createEmd( 21, SAMPLES, 0 * SAMPLES ) );
        emm->addModality( createEmd( 42, SAMPLES, 1 * SAMPLES ) );
        WLEMMeasurement::EChannelT eChannel1( SAMPLES, 1 );
        emm->addEventChannel( eChannel1 );
        WLEMMeasurement::EChannelT eChannel2( SAMPLES, 0 );
        eChannel2[EINDEX] = TRIGGER_MATCH;
        emm->addEventChannel( eChannel2 );

        WEpochSeparation::SPtr separation( new WEpochSeparation( 1, triggersMatch, PRESAMPLES, POSTSAMPLES ) );

        WLEMMeasurement::SPtr emmEpoch;
        WLEMMeasurement::SPtr emmPacket;
        for( WLSampleIdxT smp = 0; smp < SAMPLES; smp += BLOCKSIZE )
        {
            emmPacket = splitToPacket( emm, smp, BLOCKSIZE );
            if( separation->extract( emmPacket ) )
            {
                emmEpoch = separation->getNextEpoch();
            }
        }

        TS_ASSERT( emmEpoch );
        // check sizes
        TS_ASSERT_EQUALS( emmEpoch->getModalityCount(), emm->getModalityCount() );
        // check modality count
        for( size_t mod = 0; mod < emm->getModalityCount(); ++mod )
        {
            TS_ASSERT_EQUALS( emmEpoch->getModality( mod )->getNrChans(), emm->getModality( mod )->getNrChans() );
            // check channel size
            TS_ASSERT_EQUALS( emmEpoch->getModality( mod )->getSamplesPerChan(),
                            separation->getPreSamples() + separation->getPostSamples() + 1 );
            // check samples size: post + pre + 1
        }

        // check data
        WLEMData::SPtr emd;
        double startValue;
        for( size_t mod = 0; mod < emmEpoch->getModalityCount(); ++mod )
        {
            emd = emmEpoch->getModality( mod );
            startValue = mod * SAMPLES + ( EINDEX - PRESAMPLES );
            for( WLChanIdxT chan = 0; chan < emd->getNrChans(); ++chan )
            {
                for( WLSampleIdxT smp = 0; smp < ( PRESAMPLES + POSTSAMPLES + 1 ); ++smp )
                {
                    TS_ASSERT_EQUALS( emd->getData()( chan, smp ), startValue + smp );
                }
            }
        }

        // check event channels
        TS_ASSERT_EQUALS( emmEpoch->getEventChannelCount(), 1 );
        WLEMMeasurement::EChannelT eChannel;
        WLEMMeasurement::EChannelT eChannelOrg;
        size_t startIndex = EINDEX - PRESAMPLES;

        eChannel = emmEpoch->getEventChannel( 0 );
        eChannelOrg = emm->getEventChannel( 1 );
        TS_ASSERT_EQUALS( eChannel.size(), ( PRESAMPLES + POSTSAMPLES + 1 ) );
        for( size_t smp = 0; smp < ( PRESAMPLES + POSTSAMPLES + 1 ); ++smp )
        {
            TS_ASSERT_EQUALS( eChannel[smp], eChannelOrg[startIndex + smp] );
        }
    }

    void test_extractMultiResultsCheckData( void )
    {
        TS_TRACE( "test_extractMultiResultsCheckData ..." );

        const WLChanNrT CHANNELS = 42;
        const WLSampleNrT BLOCKSIZE = 200;
        const WLSampleNrT SAMPLES = BLOCKSIZE * 20;
        const WLSampleNrT PRESAMPLES = BLOCKSIZE / 2;
        const WLSampleNrT POSTSAMPLES = BLOCKSIZE;
        const WLSampleNrT EPOCHLENGTH = PRESAMPLES + POSTSAMPLES + 1;
        const WLEMMeasurement::EventT TRIGGER_MATCH1 = 1;
        const WLEMMeasurement::EventT TRIGGER_MATCH2 = 8;
        const WLEMMeasurement::EventT TRIGGER_MATCH3 = 20;
        set< WLEMMeasurement::EventT > triggersMatch;
        triggersMatch.insert( TRIGGER_MATCH1 );
        triggersMatch.insert( TRIGGER_MATCH2 );
        triggersMatch.insert( TRIGGER_MATCH3 );

        // Create test data
        std::list< size_t > eIndices;
        WLEMMeasurement::EChannelT eChannel1( SAMPLES, 0 );
        WLEMMeasurement::EventT trigger = TRIGGER_MATCH1;
        for( size_t pos = PRESAMPLES + 10; ( pos + POSTSAMPLES ) < SAMPLES; pos += PRESAMPLES )
        {
            eIndices.push_back( pos );
            eChannel1[pos] = trigger;
            eChannel1[pos + 1] = trigger;
            eChannel1[pos + 2] = trigger;

            if( trigger == TRIGGER_MATCH1 )
            {
                trigger = TRIGGER_MATCH2;
            }
            else
                if( trigger == TRIGGER_MATCH2 )
                {
                    trigger = TRIGGER_MATCH3;
                }
                else
                    if( trigger == TRIGGER_MATCH3 )
                    {
                        trigger = TRIGGER_MATCH1;
                    }
        }

        WLEMMeasurement::SPtr emm( new WLEMMeasurement() );
        emm->addEventChannel( eChannel1 );
        emm->addModality( createEmd( CHANNELS, SAMPLES, 0 * SAMPLES ) );
        emm->addModality( createEmd( CHANNELS, SAMPLES, 1 * SAMPLES ) );

        // Call trigger  routines
        WEpochSeparation::SPtr separation( new WEpochSeparation( 0, triggersMatch, PRESAMPLES, POSTSAMPLES ) );

        WLEMMeasurement::SPtr emmPacket;
        std::list< WLEMMeasurement::ConstSPtr > epochs;
        for( WLSampleIdxT smp = 0; smp < SAMPLES; smp += BLOCKSIZE )
        {
            emmPacket = splitToPacket( emm, smp, BLOCKSIZE );
            if( separation->extract( emmPacket ) )
            {
                while( separation->hasEpochs() )
                {
                    epochs.push_back( separation->getNextEpoch() );
                }
            }
        }

        // Check count of extracted epochs
        TS_ASSERT_EQUALS( eIndices.size(), epochs.size() );

        // Check data of epochs
        std::list< WLEMMeasurement::ConstSPtr >::const_iterator epIt = epochs.begin();
        std::list< size_t >::const_iterator evIt = eIndices.begin();

        WLEMData::ConstSPtr emd;
        WLEMMeasurement::ConstSPtr emmEpoch;
        WLEMData::ConstSPtr emdEpoch;

        size_t event;

        for( ; epIt != epochs.end() && evIt != eIndices.end(); ++epIt, ++evIt )
        {
            emmEpoch = *epIt;
            event = *evIt;

            // Compare modalities
            for( size_t mod = 0; mod < emmEpoch->getModalityCount(); ++mod )
            {
                emd = emm->getModality( mod );
                emdEpoch = emmEpoch->getModality( mod );
                const WLEMData::DataT& data = emd->getData();

                // Compare data
                const WLEMData::DataT& resData = emdEpoch->getData();
                for( WLChanIdxT chan = 0; chan < emdEpoch->getNrChans(); ++chan )
                {
                    TS_ASSERT_EQUALS( resData.cols(), EPOCHLENGTH );
                    for( WLEMData::DataT::Index smp = 0; smp < resData.cols(); ++smp )
                    {
                        TS_ASSERT_EQUALS( resData( chan, smp ), data( chan, event - PRESAMPLES + smp ) );
                    }
                }
            }

            // check event channels
            TS_ASSERT_EQUALS( emmEpoch->getEventChannelCount(), 1 );
            WLEMMeasurement::EChannelT& evEpoch = emmEpoch->getEventChannel( 0 );
            TS_ASSERT_EQUALS( evEpoch.size(), EPOCHLENGTH );
            for( WLSampleIdxT smp = 0; smp < static_cast< WLSampleNrT >( evEpoch.size() ); ++smp )
            {
                TS_ASSERT_EQUALS( evEpoch[smp], eChannel1[event - PRESAMPLES + smp] );
            }
        }
    }

private:
    WLEMData::SPtr createEmd( WLChanNrT channels, WLSampleIdxT samples, int startValue = 0 )
    {
        WLEMData::SPtr emd( new WLEMDEEG() );
        WLEMData::DataSPtr data( new WLEMData::DataT( channels, samples ) );

        for( WLChanIdxT chan = 0; chan < channels; ++chan )
        {
            for( WLSampleIdxT smp = 0; smp < samples; ++smp )
            {
                ( *data )( chan, smp ) = startValue + smp;
            }
        }
        emd->setData( data );

        return emd;
    }

    WLEMMeasurement::SPtr splitToPacket( WLEMMeasurement::ConstSPtr emmIn, WLSampleIdxT start, WLSampleNrT blockSize )
    {
        WLEMMeasurement::SPtr emmOut( new WLEMMeasurement( *emmIn ) );
        WLEMData::ConstSPtr emdIn;
        WLEMData::SPtr emdOut;

        // copy modality
        for( size_t mod = 0; mod < emmIn->getModalityCount(); ++mod )
        {
            emdIn = emmIn->getModality( mod );
            emdOut = emdIn->clone();
            blockSize = start + blockSize <= emdIn->getSamplesPerChan() ? blockSize : emdIn->getSamplesPerChan();
            WLEMData::DataSPtr data( new WLEMData::DataT( emdIn->getNrChans(), blockSize ) );
            const WLEMData::DataT& dataIn = emdIn->getData();
            ( *data ) = dataIn.block( 0, start, emdIn->getNrChans(), blockSize );
            emdOut->setData( data );
            emmOut->addModality( emdOut );
        }

        // copy event
        boost::shared_ptr< WLEMMeasurement::EDataT > data( new WLEMMeasurement::EDataT() );
        for( WLChanIdxT chan = 0; chan < emmIn->getEventChannelCount(); ++chan )
        {
            const WLSampleNrT n_samples = emmIn->getEventChannel( chan ).size();
            WLEMMeasurement::EChannelT channel( emmIn->getEventChannel( chan ).begin() + start,
                            start + blockSize <= n_samples ? emmIn->getEventChannel( chan ).begin() + start + blockSize :
                                            emmIn->getEventChannel( chan ).end() );
            data->push_back( channel );
        }
        emmOut->setEventChannels( data );

        return emmOut;
    }
};

#endif  // WEPOCHSEPARATION_TEST_H
