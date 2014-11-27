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

#ifndef WLCONSTANTSMODULE_H_
#define WLCONSTANTSMODULE_H_

#include <string>

/**
 * \brief Constants for a module.
 * Constants for a module, e.g. for a consistent naming.
 *
 * \author pieloth
 * \ingroup module
 */
namespace WLConstantsModule
{
    // NOTE(pieloth): "NA-Online: " causes an error on loading a saved WDataModule
    const std::string NAME_PREFIX = "[NA-Online]";

    std::string generateModuleName( const std::string& name );

    const size_t BUFFER_SIZE = 8;

    const std::string CONNECTOR_NAME_IN = "in";
    const std::string CONNECTOR_DESCR_IN = "in";
    const std::string CONNECTOR_NAME_OUT = "out";
    const std::string CONNECTOR_DESCR_OUT = "out";
}

inline std::string WLConstantsModule::generateModuleName( const std::string& name )
{
    return WLConstantsModule::NAME_PREFIX + " " + name;
}

#endif  // WLCONSTANTSMODULE_H_
