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

#ifndef WFTDATA_H_
#define WFTDATA_H_

#include <boost/shared_ptr.hpp>

#include <SimpleStorage.h>

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
     * @param numChannels The number of channels.
     * @param numSamples The number of samples.
     * @param dataType The used data type.
     */
    WFTData( UINT32_T numChannels, UINT32_T numSamples, UINT32_T dataType );

    /**
     * Inherit from WFTRequestableObject.
     *
     * @return Returns the object as Put-request.
     */
    WFTRequest::SPtr asRequest();

    /**
     * Inherit from WFTRequestableObject.
     *
     * @param The response object.
     * @return Returns whether the parsing was successful.
     */
    bool parseResponse( WFTResponse::SPtr );

    /**
     * Inherit from WFTObject.
     *
     * @return Returns the whole object size including the meta information.
     */
    UINT32_T getSize() const;

    /**
     * Gets a reference on the fixed meta information part.
     *
     * @return Returns a reference on a WFTDataDefT object.
     */
    WFTDataDefT& getDataDef();

    /**
     * Gets a pointer to the data storage. The meta information tells the properties about the stored data.
     */
    void *getData();

    /**
     * Gets whether the stored data has to convert manually into the wished data type.
     *
     * @return Returns true if there is the data type T already, else false.
     */
    template< typename T >
    bool needDataToConvert()
    {
        if( typeid(T) == typeid(float) )
        {
            return getDataDef().data_type != DATATYPE_FLOAT32 ;
        }
        else
            if( typeid(T) == typeid(double) )
            {
                return getDataDef().data_type != DATATYPE_FLOAT64 ;
            }

        return true;
    }

protected:

    /**
     * The fixed meta information.
     */
    WFTDataDefT m_def;

    /**
     * A structure to govern the data storage.
     */
    SimpleStorage m_buf;

};

#endif /* WFTDATA_H_ */
