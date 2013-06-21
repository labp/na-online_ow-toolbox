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

#ifndef WLLIFETIMEPROFILER_H_
#define WLLIFETIMEPROFILER_H_

#include <string>

#include <boost/shared_ptr.hpp>

#include "WLTimeProfiler.h"

#include "WLProfiler.h"

/**
 * A profiler for EMM objects. It counts the clones in the copy constructor and age in ms from the first creation.
 */
class WLLifetimeProfiler: public WLProfiler
{
public:
    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WLLifetimeProfiler > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WLLifetimeProfiler > ConstSPtr;

    static const std::string CLASS;

    typedef WLTimeProfiler::TimeT AgeT;

    WLLifetimeProfiler( std::string source, std::string action );

    /**
     * Creates a copy and increments the clone counter by 1
     *
     * \param Instance to copy.
     */
    WLLifetimeProfiler( const WLLifetimeProfiler& o );

    virtual ~WLLifetimeProfiler();

    WLLifetimeProfiler::SPtr clone() const;

    static WLLifetimeProfiler::SPtr instance( std::string source, std::string action );

    /**
     * Method to log or store the result.
     *
     * \param strm Output stream write the result to.
     * \return the strm argument.
     */
    virtual std::ostream& write( std::ostream& strm ) const;

    /**
     * Gets the name of the profiler.
     * \return the name
     */
    virtual std::string getName() const;

    /**
     * Gets the counter of the clones.
     *
     * \return clones
     */
    size_t getCloneCounter() const;

    /**
     * Gets the age in milliseconds.
     *
     * \return age
     */
    AgeT getAge() const;

    /**
     * Pause the lifetime.
     */
    void pause();

private:
    WLTimeProfiler::SPtr m_timeProfiler;

    /**
     * Counts the ancestors of this instances. Value is incremented in the copy constructor.
     */
    size_t m_clones;
};

#endif  // WLLIFETIMEPROFILER_H_
