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

#ifndef WLEMMCOMMAND_H_
#define WLEMMCOMMAND_H_

#include <string>

#include <boost/shared_ptr.hpp>

#include <core/dataHandler/WDataSet.h>
#include <core/common/WPrototyped.h>

#include "core/data/WLDataSetEMM.h"

/**
 * A container class to transfer data and commands between modules.
 */
class WLEMMCommand: public WDataSet
{
public:
    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WLEMMCommand > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WLEMMCommand > ConstSPtr;

    typedef std::string MiscParamT;

    struct Command
    {
        /**
         * Indicates which action should be done.
         */
        enum Enum
        {
            COMPUTE, //!< Compute EMM data.
            INIT, //!< Initialize the module and algorithm.
            MISC, //!< Free for custom commands which can be used with MiscParam.
            RESET //!< Resets the module and algorithm.
        };
    };

    explicit WLEMMCommand( Command::Enum command = Command::MISC );

    explicit WLEMMCommand( const WLEMMCommand& o );

    virtual ~WLEMMCommand();

    // ---------------------
    // Methods from WDataSet
    // ---------------------

    /**
     * Gets the name of this prototype.
     *
     * \return the name.
     */
    virtual const std::string getName() const;

    /**
     * Gets the description for this prototype.
     *
     * \return the description
     */
    virtual const std::string getDescription() const;

    /**
     * Returns a prototype instantiated with the true type of the deriving class.
     *
     * \return the prototype.
     */
    static boost::shared_ptr< WPrototyped > getPrototype();

    // -------------------------
    // Methods for WLDataSetLaBP
    // -------------------------

    /**
     * Gets the command.
     */
    Command::Enum getCommand() const;

    /**
     * Sets the command.
     *
     * \param Command to set.
     */
    void setCommand( Command::Enum command );

    /**
     * Gets the parameter for MISC command.
     */
    MiscParamT getMiscParam() const;

    /**
     * Sets a parameter for MISC command.
     */
    void setMiscParam( MiscParamT param );

    /**
     * Gets the EMM data object.
     */
    LaBP::WLDataSetEMM::ConstSPtr getEmm() const;

    /**
     * Gets the EMM data object.
     */
    LaBP::WLDataSetEMM::SPtr getEmm();

    /**
     * Sets the EMM data object.
     */
    void setEmm( LaBP::WLDataSetEMM::SPtr emm );

    /**
     * Checks if a EMM object is set.
     *
     * \return true if a EMM object is set.
     */
    bool hasEmm() const;

private:
    Command::Enum m_command;

    MiscParamT m_miscParam;

    LaBP::WLDataSetEMM::SPtr m_emm;
};

#endif  // WLEMMCOMMAND_H_
