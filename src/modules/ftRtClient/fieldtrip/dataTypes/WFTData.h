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

#ifndef WFTDATA_H_
#define WFTDATA_H_

#include <boost/shared_ptr.hpp>

#include <SimpleStorage.h>
#include <message.h>

#include "modules/ftRtClient/fieldtrip/dataTypes/WFTObject.h"
#include "modules/ftRtClient/fieldtrip/io/request/WFTRequest.h"
#include "modules/ftRtClient/fieldtrip/io/response/WFTResponse.h"
#include "WFTRequestableObject.h"

class WFTData: public WFTRequestableObject
{
public:
    /**
     * A shared pointer on a WFTData.
     */
    typedef boost::shared_ptr< WFTData > SPtr;

    /**
     * Creates an empty WFTData object.
     */
    WFTData();

    /**
     * Constructs a WFTData object with the given meat information.
     *
     * \param numChannels The number of channels.
     * \param numSamples The number of samples.
     * \param dataType The used data type.
     */
    WFTData( UINT32_T numChannels, UINT32_T numSamples, UINT32_T dataType );

    /**
     * Destroys the WFTData.
     */
    virtual ~WFTData();

    /**
     * Inherit from WFTRequestableObject.
     *
     * \return Returns the object as Put-request.
     */
    WFTRequest::SPtr asRequest();

    /**
     * Inherit from WFTRequestableObject.
     *
     * \param The response object.
     * \return Returns whether the parsing was successful.
     */
    bool parseResponse( WFTResponse::SPtr );

    /**
     * Inherit from WFTObject.
     *
     * \return Returns the whole object size including the meta information.
     */
    UINT32_T getSize() const;

    /**
     * Gets a reference on the fixed meta information part.
     *
     * \return Returns a reference on a WFTDataDefT object.
     */
    WFTDataDefT& getDataDef();

    /**
     * Gets a pointer to the data storage. The meta information tells the properties about the stored data.
     */
    void *getData();

    /**
     * Gets whether the stored data has to convert manually into the wished data type.
     *
     * \return Returns true if there is the data type T already, else false.
     */
    template< typename T >
    bool needDataToConvert();

    /**
     * This method is used to convert the data of @src into the data type T pointing @dest to.
     *
     * \param dest The destination storage.
     * \param src The source storage.
     * \param nsamp The number of samples in @src.
     * \param nchans The number of channels in @src.
     * \param dataType The FieldTrip data type @src uses.
     */
    template< typename T >
    void convertData( T *dest, const void *src, unsigned int nsamp, unsigned int nchans, UINT32_T dataType );

protected:
    /**
     * The fixed meta information.
     */
    WFTDataDefT m_def;

    /**
     * A structure to govern the data storage.
     */
    SimpleStorage m_buf;

private:
    /**
     * Method for simple converting data from @src to @dest using a basic type cast to DestT.
     * The function will be called from convertData().
     *
     * \param dest The destination storage.
     * \param src The source storage.
     * \param nsamp The number of samples in @src.
     * \param nchans The number of channels in @src.
     */
    template< typename DestT, typename SrcT >
    void convertToTargetType( DestT *dest, const void *src, unsigned int nsamp, unsigned int nchans );
};

template< typename T >
inline bool WFTData::needDataToConvert()
{
    if( typeid(T) == typeid(float) )
    {
        return getDataDef().data_type != DATATYPE_FLOAT32;
    }
    else
        if( typeid(T) == typeid(double) )
        {
            return getDataDef().data_type != DATATYPE_FLOAT64;
        }

    return true;
}

template< typename T >
inline void WFTData::convertData( T* dest, const void* src, unsigned int nsamp, unsigned int nchans, UINT32_T dataType )
{
    switch( dataType )
    {
        case DATATYPE_UINT8 :
            convertToTargetType< T, uint8_t >( dest, src, nsamp, nchans );
            break;
        case DATATYPE_INT8 :
            convertToTargetType< T, int8_t >( dest, src, nsamp, nchans );
            break;
        case DATATYPE_UINT16 :
            convertToTargetType< T, uint16_t >( dest, src, nsamp, nchans );
            break;
        case DATATYPE_INT16 :
            convertToTargetType< T, int16_t >( dest, src, nsamp, nchans );
            break;
        case DATATYPE_UINT32 :
            convertToTargetType< T, uint32_t >( dest, src, nsamp, nchans );
            break;
        case DATATYPE_INT32 :
            convertToTargetType< T, int32_t >( dest, src, nsamp, nchans );
            break;
        case DATATYPE_UINT64 :
            convertToTargetType< T, uint64_t >( dest, src, nsamp, nchans );
            break;
        case DATATYPE_INT64 :
            convertToTargetType< T, int64_t >( dest, src, nsamp, nchans );
            break;
        case DATATYPE_FLOAT32 :
            convertToTargetType< T, float >( dest, src, nsamp, nchans );
            break;
        case DATATYPE_FLOAT64 :
            convertToTargetType< T, double >( dest, src, nsamp, nchans );
            break;
    }
}

template< typename DestT, typename SrcT >
inline void WFTData::convertToTargetType( DestT* dest, const void* src, unsigned int nsamp, unsigned int nchans )
{
    const SrcT *srcT = static_cast< const SrcT * >( src );
    for( unsigned int j = 0; j < nsamp; j++ )
    {
        for( unsigned int i = 0; i < nchans; i++ )
        {
            dest[i] = ( DestT )srcT[i];
        }
        dest += nchans;
        srcT += nchans;
    }
}

#endif  // WFTDATA_H_
