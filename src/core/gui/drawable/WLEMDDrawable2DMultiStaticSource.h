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

#ifndef WLEMDDRAWABLE2DMULTISTATICSOURCE_H_
#define WLEMDDRAWABLE2DMULTISTATICSOURCE_H_

#include <string>

#include <boost/shared_ptr.hpp>

#include <core/ui/WUIViewWidget.h>

#include "core/data/WLEMMeasurement.h"

#include "WLEMDDrawable2DMultiStatic.h"

/**
 * \author pieloth
 * \ingroup gui
 */
class WLEMDDrawable2DMultiStaticSource: public WLEMDDrawable2DMultiStatic
{
public:
    /**
     * Abbreviation for a shared pointer on a instance of this class.
     */
    typedef boost::shared_ptr< WLEMDDrawable2DMultiStatic > SPtr;

    /**
     * Abbreviation for a const shared pointer on a instance of this class.
     */
    typedef boost::shared_ptr< const WLEMDDrawable2DMultiStatic > ConstSPtr;

    static const std::string CLASS;

    explicit WLEMDDrawable2DMultiStaticSource( WUIViewWidget::SPtr widget );
    virtual ~WLEMDDrawable2DMultiStaticSource();

    virtual void draw( WLEMMeasurement::SPtr emm );
};

#endif  // WLEMDDRAWABLE2DMULTISTATICSOURCE_H_
