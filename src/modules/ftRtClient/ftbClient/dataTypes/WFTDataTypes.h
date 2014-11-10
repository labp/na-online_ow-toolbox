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

#ifndef WFTDATATYPES_H_
#define WFTDATATYPES_H_

#include <boost/shared_ptr.hpp>

#include "modules/ftRtClient/ftb/WFtBuffer.h"
#include "modules/ftRtClient/ftb/WFtbChunk.h"

/**
 * A shared pointer on a FieldTrip Buffer message.
 */
typedef boost::shared_ptr< wftb::MessageT > WFTMessageT_SPtr;

/**
 * A shared pointer on a constant FieldTrip Buffer message.
 */
typedef boost::shared_ptr< const wftb::MessageT > WFTMessageT_ConstSPtr;

/**
 * A shared pointer on a FieldTrip header chunk.
 */
typedef boost::shared_ptr< wftb::ChunkT > WFTChunkT_SPtr;

/**
 * A shared pointer on a constant FieldTrip header chunk.
 */
typedef boost::shared_ptr< const wftb::ChunkT > WFTChunkT_ConstSPtr;

/**
 * A shared pointer on a FieldTrip chunk header.
 */
typedef boost::shared_ptr< wftb::ChunkDefT > WFTChunkDefT_SPtr;

/**
 * A shared pointer on a constant FieldTrip chunk header.
 */
typedef boost::shared_ptr< const wftb::ChunkDefT > WFTChunkDefT_ConstSPtr;

#endif  // WFTDATATYPES_H_
