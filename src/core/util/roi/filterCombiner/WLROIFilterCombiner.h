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

#ifndef WLROIFILTERCOMBINER_H_
#define WLROIFILTERCOMBINER_H_

#include <boost/shared_ptr.hpp>

/**
 * The WLROIFilterCombiner provides an interface to combine the ROI-FilterType structures.
 * Derived classes have to implement the combining method for a concrete type.
 * \see \cite Maschke2014
 *
 * \author maschke
 * \ingroup util
 */
template< typename Filter >
class WLROIFilterCombiner
{
public:
    /**
     * Destroys the WLROIFilterCombiner.
     */
    virtual ~WLROIFilterCombiner();

    /**
     * Interface method to combine two filter structures.
     *
     * \return Returns true if combining was successfull, otherwise false.
     */
    virtual bool combine() = 0;

    /**
     * Sets the filters to combine.
     *
     * \param filter1 The first filter.
     * \param filter2 The second filter.
     */
    void setFilter( boost::shared_ptr< Filter > first, boost::shared_ptr< Filter > second );

    /**
     * Gets the combined filter structure.
     *
     * \return Returns a @FilterType object.
     */
    boost::shared_ptr< Filter > getCombined();

protected:
    boost::shared_ptr< Filter > m_first;

    boost::shared_ptr< Filter > m_second;
};

template< typename Filter >
inline WLROIFilterCombiner< Filter >::~WLROIFilterCombiner()
{
}

template< typename Filter >
inline void WLROIFilterCombiner< Filter >::setFilter( boost::shared_ptr< Filter > first, boost::shared_ptr< Filter > second )
{
    m_first = first;
    m_second = second;
}

template< typename Filter >
inline boost::shared_ptr< Filter > WLROIFilterCombiner< Filter >::getCombined()
{
    return m_first;
}

#endif  // WLROIFILTERCOMBINER_H_
