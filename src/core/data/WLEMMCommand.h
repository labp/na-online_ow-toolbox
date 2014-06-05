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

#include <ostream>
#include <string>

#include <boost/any.hpp>
#include <boost/shared_ptr.hpp>

#include <core/common/WTransferable.h>

#include "core/data/WLEMMeasurement.h"

/**
 * A container class to transfer data and commands between modules.
 */
class WLEMMCommand: public WTransferable
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

    static const std::string CLASS;

    /**
     * Definition for the Command.MISC identifier.
     */
    typedef std::string MiscCommandT;

    /**
     * Definition for an optional Parameter.
     */
    typedef boost::any ParamT;

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
            TIME_UPDATE, //!< Indicates a time point update in the view.
            RESET //!< Resets the module and algorithm.
        };
    };

    explicit WLEMMCommand( Command::Enum command = Command::MISC );

    explicit WLEMMCommand( const WLEMMCommand& o );

    virtual ~WLEMMCommand();

    static WLEMMCommand::SPtr instance( Command::Enum command = Command::MISC );

    // -----------------------
    // Methods from WPrototype
    // -----------------------

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
     *
     * \note Method is needed by WModuleOutputData
     */
    static boost::shared_ptr< WPrototyped > getPrototype();

    // -------------------------
    // Methods for WLEMMCommand
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
     * Gets the identifier for MISC command.
     */
    MiscCommandT getMiscCommand() const;

    /**
     * Sets a identifier for MISC command.
     */
    void setMiscCommand( MiscCommandT param );

    /**
     * Gets the EMM data object.
     */
    WLEMMeasurement::ConstSPtr getEmm() const;

    /**
     * Gets the EMM data object.
     */
    WLEMMeasurement::SPtr getEmm();

    /**
     * Sets the EMM data object.
     */
    void setEmm( WLEMMeasurement::SPtr emm );

    /**
     * Checks if a EMM object is set.
     *
     * \return true if a EMM object is set.
     */
    bool hasEmm() const;

    /**
     * Gets the optional parameter.
     */
    const ParamT& getParameter() const;

    /**
     * Sets a optional generic parameter, like a union.
     *
     * \param param Parameter to set.
     */
    void setParameter( ParamT param );

    /**
     * Tries to cast a parameter to type T or throws an exception.
     */
    template< typename T >
    static T castParameter( const ParamT& param );

    /**
     * Tries to cast the parameter to type T or throws an exception.
     */
    template< typename T >
    T getParameterAs() const
    {
        return boost::any_cast< T >( m_param );
    }

private:
    /**
     * The prototype as singleton.
     */
    static WLEMMCommand::SPtr m_prototype;

    Command::Enum m_command;
    MiscCommandT m_miscCommand;

    ParamT m_param;

    WLEMMeasurement::SPtr m_emm;
};

template< typename T >
T WLEMMCommand::castParameter( const ParamT& param )
{
    return boost::any_cast< T >( param );
}

inline std::ostream& operator<<( std::ostream &strm, const WLEMMCommand& obj )
{
    strm << obj.CLASS << ": cmd=" << obj.getCommand() << "; emm=" << obj.hasEmm();
    return strm;
}

#endif  // WLEMMCOMMAND_H_
