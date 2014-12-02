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

#ifndef WLROISELECTORSOURCE_H_
#define WLROISELECTORSOURCE_H_

#include <list>

#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include "core/data/WLEMMSurface.h"
#include "core/gui/drawable/WLEMDDrawable3D.h"
#include "WLROISelector.h"

/**
 * The WLROISelectorSource is a derivation of the abstract WLROISelector.
 * It provides the adapter between the ROI configuration, the ROI Manager of OpenWalnut and the source reconstruction algorithm.
 *
 * \author maschke
 * \ingroup util
 */
class WLROISelectorSource: public WLROISelector< WLEMMSurface, std::list< size_t > >
{

public:
    /**
     * A shared pointer on a WLROISelectorSource.
     */
    typedef boost::shared_ptr< WLROISelectorSource > SPtr;

    /**
     * A shared pointer on a constant WLROISelectorSource.
     */
    typedef boost::shared_ptr< const WLROISelectorSource > ConstSPtr;

    static const std::string CLASS; //!< Class name for logging purpose.

    /**
     * Constructs a new WLROISelectorSource.
     *
     * \param data The data container.
     */
    WLROISelectorSource( WLEMMSurface::SPtr data, WLEMDDrawable3D::SPtr drawable3D );

protected:
    /**
     * Event method when creating a new ROI.
     *
     * \param A reference pointer on the new ROI.
     */
    virtual void slotAddRoi( osg::ref_ptr< WROI > );

    /**
     * Event method when deleting a ROI.
     *
     * \param A reference pointer on the ROI to delete.
     */
    virtual void slotRemoveRoi( osg::ref_ptr< WROI > );

private:
    WLEMDDrawable3D::SPtr m_drawable3D; //!< The 3D drawable.

};

#endif  // WLROISELECTORSOURCE_H_
