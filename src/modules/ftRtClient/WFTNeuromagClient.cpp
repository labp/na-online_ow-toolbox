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

#include <cmath>

#include <core/common/WLogger.h>

#include "core/data/emd/WLEMData.h"
#include "core/data/emd/WLEMDEEG.h"

#include "WFTNeuromagClient.h"

const std::string WFTNeuromagClient::CLASS = "WFTClientStreaming";

WFTNeuromagClient::WFTNeuromagClient()
{
    m_streaming = false;
}

bool WFTNeuromagClient::isStreaming() const
{
    return m_streaming;
}

bool WFTNeuromagClient::start()
{
    wlog::debug( CLASS ) << "start() called!";

    resetClient(); // reset the base client class.

    if( isStreaming() )
    {
        wlog::warn( CLASS ) << "Could start streaming. Client is already streaming!";
        return true;
    }

    if( !isConnected() )
    {
        wlog::warn( CLASS ) << "Client is not connected. Client is trying to connect.";

        if( !this->connect() )
        {
            wlog::error( CLASS ) << "Error while connecting to the FieldTrip Buffer Server. Client is disconnect.";
            return false;
        }
    }

    wlog::info( CLASS ) << "Prepare streaming.";
    if( prepareStreaming() )
    {
        wlog::info( CLASS ) << "Preparation for streaming finished. Header information are ready to retrieval.";
    }
    else
    {
        wlog::error( CLASS ) << "Error while Preparation.";
        return false;
    }

    return m_streaming = true;
}

void WFTNeuromagClient::stop()
{
    m_streaming = false;
}

bool WFTNeuromagClient::createEMM( WLEMMeasurement& emm )
{
    if( m_ftData->getDataDef().bufsize == 0 )
    {
        return false;
    }

    int chans = m_ftData->getDataDef().nchans;
    int samps = m_ftData->getDataDef().nsamples;
    float *dataSrc, *fdata;
    SimpleStorage floatStore;

    // convert data to the single precision float format
    /*
     if( m_ftData->getDataDef().data_type != DATATYPE_FLOAT32 )
     {
     floatStore.resize( sizeof(float) * samps * chans );
     fdata = ( float * )floatStore.data();

     convertData( fdata, m_ftData->getData(), samps, chans, m_ftData->getDataDef().data_type );
     dataSrc = fdata;
     }
     else // data arrived in float format
     {
     dataSrc = ( float * )m_ftData->getData();
     }
     */

    dataSrc = ( float * )m_ftData->getData();

    WLEMData::SPtr modality( new WLEMDEEG );

    WLEMData::DataSPtr dataPtr( new WLEMData::DataT( chans, samps ) ); // create data matrix
    WLEMData::DataT& data = *dataPtr;

    // insert value into the matrix
    for( int i = 0; i < samps; ++i ) // iterate all samples
    {
        // point to the next samples start position
        const float *sampleSrc = dataSrc + i * chans;
        WLEMData::SampleT sample( chans, 1 ); // create new sample for the data matrix

        for( int j = 0; j < chans; ++j )
        {
            float x = sampleSrc[j];
            if( isnanf( x ) || isinff( x ) )
            {
                return false;
            }
            sample( j ) = x; // copy the samples values into the vector
        }

        data.col( i ) = sample; // add sample-vector to the matrix
    }

    modality->setData( dataPtr );
    emm.addModality( modality );

    return true;
}

bool WFTNeuromagClient::prepareStreaming()
{
    return doHeaderRequest();
}

void WFTNeuromagClient::convertData( float *dest, const void *src, unsigned int nsamp, unsigned int nchans, UINT32_T dataType )
{
    switch( dataType )
    {
        case DATATYPE_UINT8 :
            convertToFloat< uint8_t >( dest, src, nsamp, nchans );
            break;
        case DATATYPE_INT8 :
            convertToFloat< int8_t >( dest, src, nsamp, nchans );
            break;
        case DATATYPE_UINT16 :
            convertToFloat< uint16_t >( dest, src, nsamp, nchans );
            break;
        case DATATYPE_INT16 :
            convertToFloat< int16_t >( dest, src, nsamp, nchans );
            break;
        case DATATYPE_UINT32 :
            convertToFloat< uint32_t >( dest, src, nsamp, nchans );
            break;
        case DATATYPE_INT32 :
            convertToFloat< int32_t >( dest, src, nsamp, nchans );
            break;
        case DATATYPE_UINT64 :
            convertToFloat< uint64_t >( dest, src, nsamp, nchans );
            break;
        case DATATYPE_INT64 :
            convertToFloat< int64_t >( dest, src, nsamp, nchans );
            break;
        case DATATYPE_FLOAT64 :
            convertToFloat< double >( dest, src, nsamp, nchans );
            break;
    }
}

template< typename T >
void WFTNeuromagClient::convertToFloat( float *dest, const void *src, unsigned int nsamp, unsigned int nchans )
{
    const T *srcT = static_cast< const T * >( src );
    for( unsigned int j = 0; j < nsamp; j++ )
    {
        for( unsigned int i = 0; i < nchans; i++ )
        {
            dest[i] = ( float )srcT[i];
        }
        dest += nchans;
        srcT += nchans;
    }
}
