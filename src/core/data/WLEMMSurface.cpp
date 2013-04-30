// TODO license

#include <vector>

#include <boost/shared_ptr.hpp>

#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>

#include "core/dataHandler/WDataSetEMMEnumTypes.h"
#include "WLEMMSurface.h"

LaBP::WLEMMSurface::WLEMMSurface()
{
    setVertexUnit( WEUnit::UNKNOWN_UNIT );
    setVertexExponent( WEExponent::BASE );

    m_vertex.reset( new std::vector< WPosition >( 0 ) );
    m_faces.reset( new std::vector< WVector3i >( 0 ) );
}

LaBP::WLEMMSurface::WLEMMSurface( boost::shared_ptr< std::vector< WPosition > > vertex, WEUnit::Enum vertexUnit,
                WEExponent::Enum vertexExponent, boost::shared_ptr< std::vector< WVector3i > > faces,
                Hemisphere::Enum hemisphere ) :
                m_vertex( vertex ), m_vertexUnit( vertexUnit ), m_vertexExponent( vertexExponent ), m_faces( faces ), m_hemisphere(
                                hemisphere )
{
}

LaBP::WLEMMSurface::WLEMMSurface( const WLEMMSurface& surface )
{
    m_hemisphere = surface.m_hemisphere;
    m_vertexUnit = surface.m_vertexUnit;
    m_vertexExponent = surface.m_vertexExponent;

    m_vertex.reset( new std::vector< WPosition >( 0 ) );
    m_faces.reset( new std::vector< WVector3i >( 0 ) );
}

LaBP::WLEMMSurface::~WLEMMSurface()
{
}

boost::shared_ptr< std::vector< WPosition > >  LaBP::WLEMMSurface::getVertex() const
{
    return m_vertex;
}

void LaBP::WLEMMSurface::setVertex( boost::shared_ptr< std::vector< WPosition > > vertex )
{
    m_vertex = vertex;
}

LaBP::WEUnit::Enum LaBP::WLEMMSurface::getVertexUnit() const
{
    return m_vertexUnit;
}

void LaBP::WLEMMSurface::setVertexUnit( LaBP::WEUnit::Enum unit )
{
    m_vertexUnit = unit;
}

LaBP::WEExponent::Enum LaBP::WLEMMSurface::getVertexExponent() const
{
    return m_vertexExponent;
}

void LaBP::WLEMMSurface::setVertexExponent( LaBP::WEExponent::Enum exponent )
{
    m_vertexExponent = exponent;
}

std::vector< WVector3i >& LaBP::WLEMMSurface::getFaces() const
{
    return *m_faces;
}

void LaBP::WLEMMSurface::setFaces( boost::shared_ptr< std::vector< WVector3i > > faces )
{
    m_faces = faces;
}

LaBP::WLEMMSurface::Hemisphere::Enum LaBP::WLEMMSurface::getHemisphere() const
{
    return m_hemisphere;
}
void LaBP::WLEMMSurface::setHemisphere( Hemisphere::Enum val )
{
    m_hemisphere = val;
}
