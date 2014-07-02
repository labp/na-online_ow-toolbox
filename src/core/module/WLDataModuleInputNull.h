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

#ifndef WLDATAMODULEINPUTNULL_H_
#define WLDATAMODULEINPUTNULL_H_

#include <core/kernel/WDataModuleInput.h>

/**
 * A Null Object based input for the \ref WDataModule, which represents a not set input.
 *
 * \author pieloth
 */
class WLDataModuleInputNull: public WDataModuleInput
{
public:
    /**
     * Convenience typedef for a boost::shared_ptr< WDataModuleInput >.
     */
    typedef boost::shared_ptr< WLDataModuleInputNull > SPtr;

    /**
     * Convenience typedef for a boost::shared_ptr< const WDataModuleInput >.
     */
    typedef boost::shared_ptr< const WLDataModuleInputNull > ConstSPtr;

    /**
     * Default constructor.
     */

    WLDataModuleInputNull();
    virtual ~WLDataModuleInputNull();

    virtual std::string getName() const;

    virtual std::string asString() const;

    virtual std::ostream& serialize( std::ostream& out ) const;
};

#endif  // WLDATAMODULEINPUTNULL_H_
