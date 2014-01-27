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

#ifndef WLPROFILER_H_
#define WLPROFILER_H_

#include <ostream>
#include <string>

#include <boost/shared_ptr.hpp>

/**
 * Base class for profiler measurements.
 *
 * \author pieloth
 */
class WLProfiler
{
public:
    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WLProfiler > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WLProfiler > ConstSPtr;

    /**
     * Constructor. If autoLog is true, the implementation must start the measurement in constructor and must stop and log the result in destructor.
     *
     * \param source Class or source which is measured.
     * \param action A action or method which is measured.
     */
    WLProfiler( std::string source, std::string action, bool autoLog = true );

    /**
     * The destructor should do an log to WLProfilerLogger, if isAutoLog is true!
     */
    virtual ~WLProfiler();

    /**
     * Method to log or store the result.
     *
     * \param strm Output stream write the result to.
     * \return the strm argument.
     */
    virtual std::ostream& write( std::ostream& strm ) const = 0;

    /**
     * Gets the name of the profiler.
     * \return the name
     */
    virtual std::string getName() const = 0;

    /**
     * Gets the source for this profiler.
     *
     * \return the source.
     */
    std::string getSource() const;

    /**
     * Gets the action for this profiler.
     *
     * \return the action.
     */
    std::string getAction() const;

    /**
     * Gets  the autoLog flag.
     *
     * \return true if autoLog is on.
     */
    bool isAutoLog() const;

protected:
    /**
     * Class or source which is measured.
     */
    std::string m_source;

    /**
     * A action or method which is measured.
     */
    std::string m_action;

    /**
     * Indicates if the profiler should log the result in destructor.
     */
    bool m_autoLog;
};

/**
 * Overload for streamed output.
 *
 * \param strm Output stream
 * \param profiler Profiler to write
 */
inline std::ostream& operator<<( std::ostream &strm, const WLProfiler& profiler )
{
    return profiler.write( strm );
}

inline std::string WLProfiler::getSource() const
{
    return m_source;
}

inline std::string WLProfiler::getAction() const
{
    return m_action;
}

inline bool WLProfiler::isAutoLog() const
{
    return m_autoLog;
}

#endif  // WLPROFILER_H_
