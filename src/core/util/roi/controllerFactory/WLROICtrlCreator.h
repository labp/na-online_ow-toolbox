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

#ifndef WLROICTRLCREATOR_H_
#define WLROICTRLCREATOR_H_

#include <boost/shared_ptr.hpp>

#include <core/graphicsEngine/WROI.h>

/**
 * The WLROICtrlCreator provides an interface to create an instance of @Base under the condition of @DataType and @FilterType.
 * \see \cite Maschke2014
 *
 * \author maschke
 * \ingroup util
 */
template< typename Base, typename DataType = void >
class WLROICtrlCreator
{
public:
    /**
     * A shared pointer on a WLROICtrlCreator.
     */
    typedef boost::shared_ptr< WLROICtrlCreator< Base, DataType > > type;

    /**
     * Destroys the WLROICtrlCreator.
     */
    virtual ~WLROICtrlCreator();

    /**
     * Interface method for creating a new @Base object by the derived class with the defined
     * parameter @roi and @data.
     *
     * \param roi The ROI.
     * \param data The data container.
     * \return
     */
    virtual Base *create( osg::ref_ptr< WROI > roi, boost::shared_ptr< DataType > data ) = 0;
};

template< typename Base, typename DataType >
inline WLROICtrlCreator< Base, DataType >::~WLROICtrlCreator()
{
}

#endif  // WLROICTRLCREATOR_H_
