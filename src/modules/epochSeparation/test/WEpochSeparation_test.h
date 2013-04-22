#ifndef WEPOCHSEPARATION_TEST_H
#define WEPOCHSEPARATION_TEST_H

#include <cstddef>
#include <list>
#include <set>

#include <cxxtest/TestSuite.h>
#include <boost/shared_ptr.hpp>

#include "core/common/WLogger.h"

#include "core/dataHandler/WDataSetEMM.h"
#include "core/dataHandler/WDataSetEMMEMD.h"
#include "core/dataHandler/WDataSetEMMEEG.h"

#include "../WEpochSeparation.h"

using namespace std;

class WTriggerExtractorCpuTest: public CxxTest::TestSuite
{
public:
    void setUp( void )
    {
        WLogger::startup();
    }

    void test_setGetReset( void )
    {
        TS_TRACE( "test_setGetReset ..." );

        const size_t PRESAMPLES = 42;
        const size_t POSTSAMPLES = 23;
        const size_t ECHANNEL = 2;
        set< LaBP::WDataSetEMM::EventT > triggers;
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

        LaBP::WDataSetEMM::SPtr emm( new LaBP::WDataSetEMM() );
        const size_t SAMPLES = 10;
        emm->addModality( createEmd( 21, SAMPLES, 0 * SAMPLES ) );
        emm->addModality( createEmd( 42, SAMPLES, 1 * SAMPLES ) );

        // Single packet must be bigger than (postSamles + preSamles + 1), because WEpochSeparation returns true only for a full collected packet
        set< LaBP::WDataSetEMM::EventT > triggers;
        triggers.insert( 1 );
        WEpochSeparation::SPtr separation( new WEpochSeparation( 1, triggers, SAMPLES / 3, SAMPLES / 3 ) );
        // no event channel
        TS_ASSERT_EQUALS( separation->extract( emm ), 0 );

        separation.reset( new WEpochSeparation( 1, triggers, SAMPLES / 3, SAMPLES / 3 ) );
        emm.reset( new LaBP::WDataSetEMM() );
        LaBP::WDataSetEMM::EChannelT eChannel( SAMPLES, 1 );
        emm->addEventChannel( eChannel );
        // no modality
        TS_ASSERT_EQUALS( separation->extract( emm ), 0 );
    }

    void test_extractSinglePacketCheckMaskCheckChannel( void )
    {
        TS_TRACE( "test_extractSinglePacketCheckMaskCheckChannel ..." );

        const size_t SAMPLES = 500;
        const size_t PRESAMPLES = SAMPLES / 3;
        const size_t POSTSAMPLES = SAMPLES / 3;
        const LaBP::WDataSetEMM::EventT TRIGGER_MATCH = 4;
        set< LaBP::WDataSetEMM::EventT > triggersMatch;
        triggersMatch.insert( TRIGGER_MATCH + 5 );
        triggersMatch.insert( TRIGGER_MATCH );
        const LaBP::WDataSetEMM::EventT TRIGGER_NO_MATCH = 1;
        set< LaBP::WDataSetEMM::EventT > triggersNoMatch;
        triggersNoMatch.insert( TRIGGER_NO_MATCH );
        const size_t EINDEX = SAMPLES / 2;

        LaBP::WDataSetEMM::SPtr emm( new LaBP::WDataSetEMM() );
        emm->addModality( createEmd( 21, SAMPLES, 0 * SAMPLES ) );
        emm->addModality( createEmd( 42, SAMPLES, 1 * SAMPLES ) );

        LaBP::WDataSetEMM::EChannelT eChannel1( SAMPLES, TRIGGER_NO_MATCH );
        emm->addEventChannel( eChannel1 );
        LaBP::WDataSetEMM::EChannelT eChannel2( SAMPLES, 0 );
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

        // TODO (pieloth) error cause trigger is not full and tries to get data of pre samples
        separation.reset( new WEpochSeparation( 1, triggersMatch, SAMPLES + 1, SAMPLES + 1 ) );
        // should not match, because trigger is waiting for more packets
        TS_ASSERT_THROWS_ANYTHING( separation->extract( emm ) );
    }

    void test_extractSinglePacketCheckData( void )
    {
        TS_TRACE( "test_extractSinglePacketCheckData ..." );

        const size_t SAMPLES = 500;
        const size_t PRESAMPLES = SAMPLES / 3;
        const size_t POSTSAMPLES = SAMPLES / 3;
        const LaBP::WDataSetEMM::EventT TRIGGER_MATCH = 4;
        set< LaBP::WDataSetEMM::EventT > triggersMatch;
        triggersMatch.insert( TRIGGER_MATCH + 42 );
        triggersMatch.insert( TRIGGER_MATCH );
        const size_t EINDEX = SAMPLES / 2;

        LaBP::WDataSetEMM::SPtr emm( new LaBP::WDataSetEMM() );
        emm->addModality( createEmd( 21, SAMPLES, 0 * SAMPLES ) );
        emm->addModality( createEmd( 42, SAMPLES, 1 * SAMPLES ) );
        LaBP::WDataSetEMM::EChannelT eChannel1( SAMPLES, 1 );
        emm->addEventChannel( eChannel1 );
        LaBP::WDataSetEMM::EChannelT eChannel2( SAMPLES, 0 );
        eChannel2[EINDEX] = TRIGGER_MATCH;
        emm->addEventChannel( eChannel2 );

        WEpochSeparation::SPtr separation( new WEpochSeparation( 1, triggersMatch, PRESAMPLES, POSTSAMPLES ) );
        TS_ASSERT( separation->extract( emm ) );
        // should match
        LaBP::WDataSetEMM::SPtr emmEpoch = separation->getNextEpoch();

        // check sizes
        TS_ASSERT_EQUALS( emmEpoch->getModalityCount(), emm->getModalityCount() );
        // check modality count
        for( size_t mod = 0; mod < emm->getModalityCount(); ++mod )
        {
            TS_ASSERT_EQUALS( emmEpoch->getModality( mod )->getData().size(), emm->getModality( mod )->getData().size() );
            // check channel size
            TS_ASSERT_EQUALS( emmEpoch->getModality( mod )->getData().front().size(),
                            separation->getPreSamples() + separation->getPostSamples() + 1 );
            // check samples size: post + pre + 1
        }

        // check data
        LaBP::WDataSetEMMEMD::SPtr emdEpoch;
        double startValue;
        for( size_t mod = 0; mod < emmEpoch->getModalityCount(); ++mod )
        {
            emdEpoch = emmEpoch->getModality( mod );
            startValue = mod * SAMPLES + ( EINDEX - PRESAMPLES );
            for( size_t chan = 0; chan < emdEpoch->getData().size(); ++chan )
            {
                for( size_t smp = 0; smp < ( PRESAMPLES + POSTSAMPLES + 1 ); ++smp )
                {
                    TS_ASSERT_EQUALS( emdEpoch->getData()[chan][smp], startValue + smp );
                }
            }
        }

        // check event channels
        TS_ASSERT_EQUALS( emmEpoch->getEventChannelCount(), emm->getEventChannelCount() );
        LaBP::WDataSetEMM::EChannelT eChannel;
        LaBP::WDataSetEMM::EChannelT eChannelOrg;
        size_t startIndex = EINDEX - PRESAMPLES;
        for( size_t eChan = 0; eChan < emmEpoch->getEventChannelCount(); ++eChan )
        {
            eChannel = emmEpoch->getEventChannel( eChan );

            eChannelOrg = emm->getEventChannel( eChan );
            TS_ASSERT_EQUALS( eChannel.size(), ( PRESAMPLES + POSTSAMPLES + 1 ) );
            for( size_t smp = 0; smp < ( PRESAMPLES + POSTSAMPLES + 1 ); ++smp )
            {
                TS_ASSERT_EQUALS( eChannel[smp], eChannelOrg[startIndex + smp] );
            }
        }
    }

    void test_extractMultiPacketCheckData( void )
    {
        TS_TRACE( "test_extractMultiPacketCheckData ..." );

        const size_t BLOCKSIZE = 250;
        const size_t SAMPLES = BLOCKSIZE * 20;
        const size_t PRESAMPLES = 3 * BLOCKSIZE;
        const size_t POSTSAMPLES = 2 * BLOCKSIZE;
        const LaBP::WDataSetEMM::EventT TRIGGER_MATCH = 4;
        set< LaBP::WDataSetEMM::EventT > triggersMatch;
        triggersMatch.insert( TRIGGER_MATCH + 21 );
        triggersMatch.insert( TRIGGER_MATCH );
        const size_t EINDEX = SAMPLES - ( POSTSAMPLES + 1.5 * BLOCKSIZE );

        LaBP::WDataSetEMM::SPtr emm( new LaBP::WDataSetEMM() );
        emm->addModality( createEmd( 21, SAMPLES, 0 * SAMPLES ) );
        emm->addModality( createEmd( 42, SAMPLES, 1 * SAMPLES ) );
        LaBP::WDataSetEMM::EChannelT eChannel1( SAMPLES, 1 );
        emm->addEventChannel( eChannel1 );
        LaBP::WDataSetEMM::EChannelT eChannel2( SAMPLES, 0 );
        eChannel2[EINDEX] = TRIGGER_MATCH;
        emm->addEventChannel( eChannel2 );

        WEpochSeparation::SPtr separation( new WEpochSeparation( 1, triggersMatch, PRESAMPLES, POSTSAMPLES ) );

        LaBP::WDataSetEMM::SPtr emmEpoch;
        LaBP::WDataSetEMM::SPtr emmPacket;
        for( size_t smp = 0; smp < SAMPLES; smp += BLOCKSIZE )
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
            TS_ASSERT_EQUALS( emmEpoch->getModality( mod )->getData().size(), emm->getModality( mod )->getData().size() );
            // check channel size
            TS_ASSERT_EQUALS( emmEpoch->getModality( mod )->getData().front().size(),
                            separation->getPreSamples() + separation->getPostSamples() + 1 );
            // check samples size: post + pre + 1
        }

        // check data
        LaBP::WDataSetEMMEMD::SPtr emd;
        double startValue;
        for( size_t mod = 0; mod < emmEpoch->getModalityCount(); ++mod )
        {
            emd = emmEpoch->getModality( mod );
            startValue = mod * SAMPLES + ( EINDEX - PRESAMPLES );
            for( size_t chan = 0; chan < emd->getData().size(); ++chan )
            {
                for( size_t smp = 0; smp < ( PRESAMPLES + POSTSAMPLES + 1 ); ++smp )
                {
                    TS_ASSERT_EQUALS( emd->getData()[chan][smp], startValue + smp );
                }
            }
        }

        // check event channels
        TS_ASSERT_EQUALS( emmEpoch->getEventChannelCount(), emm->getEventChannelCount() );
        LaBP::WDataSetEMM::EChannelT eChannel;
        LaBP::WDataSetEMM::EChannelT eChannelOrg;
        size_t startIndex = EINDEX - PRESAMPLES;
        for( size_t eChan = 0; eChan < emmEpoch->getEventChannelCount(); ++eChan )
        {
            eChannel = emmEpoch->getEventChannel( eChan );

            eChannelOrg = emm->getEventChannel( eChan );
            TS_ASSERT_EQUALS( eChannel.size(), ( PRESAMPLES + POSTSAMPLES + 1 ) );
            for( size_t smp = 0; smp < ( PRESAMPLES + POSTSAMPLES + 1 ); ++smp )
            {
                TS_ASSERT_EQUALS( eChannel[smp], eChannelOrg[startIndex + smp] );
            }
        }
    }

    void test_extractMultiResultsCheckData( void )
    {
        TS_TRACE( "test_extractMultiResultsCheckData ..." );

        const size_t CHANNELS = 42;
        const size_t BLOCKSIZE = 200;
        const size_t SAMPLES = BLOCKSIZE * 20;
        const size_t PRESAMPLES = BLOCKSIZE / 2;
        const size_t POSTSAMPLES = BLOCKSIZE;
        const size_t EPOCHLENGTH = PRESAMPLES + POSTSAMPLES + 1;
        const LaBP::WDataSetEMM::EventT TRIGGER_MATCH1 = 1;
        const LaBP::WDataSetEMM::EventT TRIGGER_MATCH2 = 8;
        const LaBP::WDataSetEMM::EventT TRIGGER_MATCH3 = 20;
        set< LaBP::WDataSetEMM::EventT > triggersMatch;
        triggersMatch.insert( TRIGGER_MATCH1 );
        triggersMatch.insert( TRIGGER_MATCH2 );
        triggersMatch.insert( TRIGGER_MATCH3 );

        // Create test data
        std::list< size_t > eIndices;
        LaBP::WDataSetEMM::EChannelT eChannel1( SAMPLES, 0 );
        LaBP::WDataSetEMM::EventT trigger = TRIGGER_MATCH1;
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

        LaBP::WDataSetEMM::SPtr emm( new LaBP::WDataSetEMM() );
        emm->addEventChannel( eChannel1 );
        emm->addModality( createEmd( CHANNELS, SAMPLES, 0 * SAMPLES ) );
        emm->addModality( createEmd( CHANNELS, SAMPLES, 1 * SAMPLES ) );

        // Call trigger  routines
        WEpochSeparation::SPtr separation( new WEpochSeparation( 0, triggersMatch, PRESAMPLES, POSTSAMPLES ) );

        LaBP::WDataSetEMM::SPtr emmPacket;
        std::list< LaBP::WDataSetEMM::ConstSPtr > epochs;
        for( size_t smp = 0; smp < SAMPLES; smp += BLOCKSIZE )
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
        std::list< LaBP::WDataSetEMM::ConstSPtr >::const_iterator epIt = epochs.begin();
        std::list< size_t >::const_iterator evIt = eIndices.begin();

        LaBP::WDataSetEMMEMD::ConstSPtr emd;
        LaBP::WDataSetEMM::ConstSPtr emmEpoch;
        LaBP::WDataSetEMMEMD::ConstSPtr emdEpoch;

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
                LaBP::WDataSetEMMEMD::DataT& data = emd->getData();

                // Compare data
                LaBP::WDataSetEMMEMD::DataT& resData = emdEpoch->getData();
                for( size_t chan = 0; chan < emdEpoch->getNrChans(); ++chan )
                {
                    TS_ASSERT_EQUALS( resData[chan].size(), EPOCHLENGTH );
                    for( size_t smp = 0; smp < resData[chan].size(); ++smp )
                    {
                        TS_ASSERT_EQUALS( resData[chan][smp], data[chan][event - PRESAMPLES + smp] );
                    }
                }
            }

            // check event channels
            TS_ASSERT_EQUALS( emmEpoch->getEventChannelCount(), emm->getEventChannelCount() );
            for( size_t chan = 0; chan < emmEpoch->getEventChannelCount(); ++chan )
            {
                LaBP::WDataSetEMM::EChannelT& evEpoch = emmEpoch->getEventChannel( chan );
                TS_ASSERT_EQUALS( evEpoch.size(), EPOCHLENGTH );
                for( size_t smp = 0; smp < evEpoch.size(); ++smp )
                {
                    TS_ASSERT_EQUALS( evEpoch[smp], eChannel1[event - PRESAMPLES + smp] );
                }
            }
        }
    }

protected:

private:
    LaBP::WDataSetEMMEMD::SPtr createEmd( size_t channels, size_t samples, int startValue = 0 )
    {
        LaBP::WDataSetEMMEMD::SPtr emd( new LaBP::WDataSetEMMEEG() );
        boost::shared_ptr< LaBP::WDataSetEMMEMD::DataT > data( new LaBP::WDataSetEMMEMD::DataT() );

        for( size_t chan = 0; chan < channels; ++chan )
        {
            LaBP::WDataSetEMMEMD::ChannelT channel;
            for( size_t smp = 0; smp < samples; ++smp )
            {
                channel.push_back( startValue + smp );
            }
            data->push_back( channel );
        }
        emd->setData( data );

        return emd;
    }

    LaBP::WDataSetEMM::SPtr splitToPacket( LaBP::WDataSetEMM::ConstSPtr emmIn, size_t start, size_t blockSize )
    {
        LaBP::WDataSetEMM::SPtr emmOut( new LaBP::WDataSetEMM( *emmIn ) );
        LaBP::WDataSetEMMEMD::ConstSPtr emdIn;
        LaBP::WDataSetEMMEMD::SPtr emdOut;

        // copy modality
        for( size_t mod = 0; mod < emmIn->getModalityCount(); ++mod )
        {
            emdIn = emmIn->getModality( mod );
            emdOut = emdIn->clone();
            boost::shared_ptr< LaBP::WDataSetEMMEMD::DataT > data( new LaBP::WDataSetEMMEMD::DataT() );
            for( size_t chan = 0; chan < emdIn->getData().size(); ++chan )
            {
                LaBP::WDataSetEMMEMD::ChannelT channel( emdIn->getData()[chan].begin() + start,
                                start + blockSize <= emdIn->getData()[chan].size() ? emdIn->getData()[chan].begin() + start
                                                + blockSize :
                                                emdIn->getData()[chan].end() );
                data->push_back( channel );
            }
            emdOut->setData( data );
            emmOut->addModality( emdOut );
        }

        // copy event
        boost::shared_ptr< LaBP::WDataSetEMM::EDataT > data( new LaBP::WDataSetEMM::EDataT() );
        for( size_t chan = 0; chan < emmIn->getEventChannelCount(); ++chan )
        {
            LaBP::WDataSetEMM::EChannelT channel( emmIn->getEventChannel( chan ).begin() + start,
                            start + blockSize <= emmIn->getEventChannel( chan ).size() ? emmIn->getEventChannel( chan ).begin()
                                            + start + blockSize :
                                            emmIn->getEventChannel( chan ).end() );
            data->push_back( channel );
        }
        emmOut->setEventChannels( data );

        return emmOut;
    }

};

#endif // WEPOCHSEPARATION_TEST_H
