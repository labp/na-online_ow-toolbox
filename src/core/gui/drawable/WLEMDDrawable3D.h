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

#ifndef WLEMDDRAWABLE3D_H_
#define WLEMDDRAWABLE3D_H_

#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>
#include <osg/Projection>
#include <osg/Geometry>
#include <osg/ref_ptr>
#include <osg/StateSet>

#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>
#include <core/ui/WUIViewWidget.h>

#include "core/data/enum/WLEModality.h"
#include "core/gui/colorMap/WLColorMap.h"

#include "WLEMDDrawable.h"

class WLEMDDrawable3D: public WLEMDDrawable
{
public:
    /**
     * Abbreviation for a shared pointer on a instance of this class.
     */
    typedef boost::shared_ptr< WLEMDDrawable3D > SPtr;

    /**
     * Abbreviation for a const shared pointer on a instance of this class.
     */
    typedef boost::shared_ptr< const WLEMDDrawable3D > ConstSPtr;

    /**
     * Class name for logs.
     */
    static const std::string CLASS;

    /**
     * Creates an instance for the requested parameters.
     *
     * \param widget widget to fill.
     * \param modality modality to draw.
     *
     * \return Instance of a WLEMDDrawable3D implementation.
     */
    static WLEMDDrawable3D::SPtr getInstance( WUIViewWidget::SPtr widget, WLEModality::Enum modality );

    /**
     * Constructor.
     *
     * \param widget
     */
    explicit WLEMDDrawable3D( WUIViewWidget::SPtr m_widget );

    /**
     * Destructor.
     */
    virtual ~WLEMDDrawable3D();

    virtual void draw( WLEMMeasurement::SPtr emm );

    virtual bool hasData() const;

    /**
     * Gets selected sample/index.
     *
     * \return sample index
     */
    virtual ptrdiff_t getSelectedSample() const;

    /**
     * Set selected sample/index to mark, draw or else.
     */
    virtual bool setSelectedSample( ptrdiff_t pixe );

    virtual float getSelectedTime() const;

    virtual bool setSelectedTime( float relative );

    /**
     * Returns the color map for the surface.
     *
     * \return Color map which is used.
     */
    virtual WLColorMap::SPtr getColorMap() const;

    /**
     * Sets the color map to use fpr the surface.
     * \param colorMap
     */
    virtual void setColorMap( WLColorMap::SPtr colorMap );

protected:
    virtual bool mustDraw() const;

    virtual void osgNodeCallback( osg::NodeVisitor* nv );

    WLEMMeasurement::ConstSPtr m_emm;

    void osgAddSurface( const std::vector< WPosition >& positions, const std::vector< WVector3i >& faces );

    void osgAddColorMap();

    float m_zoomFactor;

    bool m_colorMapChanged;

    /**
     * Index of the selected sample.
     */
    ptrdiff_t m_selectedSample;

    /**
     * A flag to indicate a change of m_selectedSample.
     */
    bool m_selectedSampleChanged;

    WLColorMap::SPtr m_colorMap;

    bool m_surfaceChanged;

    osg::ref_ptr< osg::Projection > m_colorMapNode;

    osg::ref_ptr< osg::Geode > m_surfaceGeode;

    osg::ref_ptr< osg::Geometry > m_surfaceGeometry;

    osg::ref_ptr< osg::StateSet > m_state;
};

#endif  // WLEMDDRAWABLE3D_H_
