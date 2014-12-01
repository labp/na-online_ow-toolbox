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

#ifndef WLEMDDRAWABLE2DMULTIDYNAMICSOURCE_H_
#define WLEMDDRAWABLE2DMULTIDYNAMICSOURCE_H_

#include <string>

#include <boost/shared_ptr.hpp>

#include <core/ui/WUIViewWidget.h>

#include "core/data/WLEMMeasurement.h"

#include "WLEMDDrawable2DMultiDynamic.h"

/**
 * \author pieloth
 * \ingroup gui
 */
class WLEMDDrawable2DMultiDynamicSource: public WLEMDDrawable2DMultiDynamic
{
public:
    /**
     * Abbreviation for a shared pointer on a instance of this class.
     */
    typedef boost::shared_ptr< WLEMDDrawable2DMultiDynamicSource > SPtr;

    /**
     * Abbreviation for a const shared pointer on a instance of this class.
     */
    typedef boost::shared_ptr< const WLEMDDrawable2DMultiDynamicSource > ConstSPtr;

    static const std::string CLASS;

    explicit WLEMDDrawable2DMultiDynamicSource( WUIViewWidget::SPtr widget );
    virtual ~WLEMDDrawable2DMultiDynamicSource();

    virtual void draw( WLEMMeasurement::SPtr emm );
};

#endif  // WLEMDDRAWABLE2DMULTIDYNAMICSOURCE_H_
