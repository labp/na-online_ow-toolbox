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

#ifndef WFTEVENT_H_
#define WFTEVENT_H_

#include <ostream>
#include <string>

#include <boost/shared_ptr.hpp>

#include <message.h>

#include "modules/ftRtClient/ftbClient/dataTypes/WFTObject.h"

/**
 * The WFTEvent represents a single FieldTrip event. Is is defined by a header contains the fixed structure and the data.
 */
class WFTEvent: public WFTObject
{
public:
    /**
     * A shared pointer on an WFTEvent.
     */
    typedef boost::shared_ptr< WFTEvent > SPtr;

    /**
     * The class name.
     */
    static const std::string CLASS;

    /**
     * Constructs a new WFTEvent.
     *
     * \param def The header information, decribing the event.
     * \param type The data type of event.
     * \param value The value of the event.
     */
    WFTEvent( WFTEventDefT def, const std::string type, const std::string value );

    /**
     * Constructs a new WFTEvent.
     *
     * \param sample The index of sample this event relates to.
     * \param offset The offset of event w.r.t. sample (time).
     * \param duration The duration of the event.
     * \param type The data type of event.
     * \param value The value of the event.
     */
    WFTEvent( INT32_T sample, INT32_T offset, INT32_T duration, const std::string type, const std::string value );

    /**
     * Destroys the WFTEvent.
     */
    virtual ~WFTEvent();

    /**
     * Inherited method from WFTObject.
     *
     * \return Returns the size of the whole object including the event header.
     */
    UINT32_T getSize() const;

    /**
     * Gets a reference on the event header.
     *
     * \return A reference on the event header.
     */
    WFTEventDefT& getDef();

    /**
     * Gets the event header.
     *
     * \return The event header.
     */
    WFTEventDefT getDef() const;

    /**
     * Gets the type.
     *
     * \return The type.
     */
    std::string const getType() const;

    /**
     * Gets the value.
     *
     * \return The value.
     */
    std::string const getValue() const;

private:
    /**
     * The event header.
     */
    WFTEventDefT m_def;

    /**
     * The type.
     */
    const std::string m_type;

    /**
     * The value.
     */
    const std::string m_value;
};

/**
 * Overloads the << operator to print the events data.
 *
 * \param strm The ostream.
 * \param event The event object.
 *
 * \return Returns an ostream, which contains the events information.
 */
inline std::ostream& operator<<( std::ostream &strm, const WFTEvent& event )
{
    strm << WFTEvent::CLASS << ":";
    strm << " Sample = " << event.getDef().sample;
    strm << ", Duration = " << event.getDef().duration;
    strm << ", Offset = " << event.getDef().offset;
    strm << ", Type = " << event.getType();
    strm << ", Value = " << event.getValue();

    return strm;
}

#endif  // WFTEVENT_H_
