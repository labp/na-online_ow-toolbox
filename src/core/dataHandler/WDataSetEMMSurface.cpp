// TODO license

#include <vector>

#include <boost/shared_ptr.hpp>

#include "core/common/math/linearAlgebra/WPosition.h"
#include "core/common/math/linearAlgebra/WVectorFixed.h"

#include "WDataSetEMMEnumTypes.h"
#include "WDataSetEMMSurface.h"

LaBP::WDataSetEMMSurface::WDataSetEMMSurface()
{
    setVertexUnit( WEUnit::UNKNOWN_UNIT );
    setVertexExponent( WEExponent::BASE );

    m_vertex.reset( new std::vector< WPosition >( 0 ) );
    m_faces.reset( new std::vector< WVector3i >( 0 ) );
}

LaBP::WDataSetEMMSurface::WDataSetEMMSurface( boost::shared_ptr< std::vector< WPosition > > vertex, WEUnit::Enum vertexUnit,
                WEExponent::Enum vertexExponent, boost::shared_ptr< std::vector< WVector3i > > faces,
                Hemisphere::Enum hemisphere ) :
                m_vertex( vertex ), m_vertexUnit( vertexUnit ), m_vertexExponent( vertexExponent ), m_faces( faces ), m_hemisphere(
                                hemisphere )
{
}

LaBP::WDataSetEMMSurface::WDataSetEMMSurface( const WDataSetEMMSurface& surface )
{
    m_hemisphere = surface.m_hemisphere;
    m_vertexUnit = surface.m_vertexUnit;
    m_vertexExponent = surface.m_vertexExponent;

    m_vertex.reset( new std::vector< WPosition >( 0 ) );
    m_faces.reset( new std::vector< WVector3i >( 0 ) );
}

LaBP::WDataSetEMMSurface::~WDataSetEMMSurface()
{
}

boost::shared_ptr< std::vector< WPosition > >  LaBP::WDataSetEMMSurface::getVertex() const
{
    return m_vertex;
}

void LaBP::WDataSetEMMSurface::setVertex( boost::shared_ptr< std::vector< WPosition > > vertex )
{
    m_vertex = vertex;
}

LaBP::WEUnit::Enum LaBP::WDataSetEMMSurface::getVertexUnit() const
{
    return m_vertexUnit;
}

void LaBP::WDataSetEMMSurface::setVertexUnit( LaBP::WEUnit::Enum unit )
{
    m_vertexUnit = unit;
}

LaBP::WEExponent::Enum LaBP::WDataSetEMMSurface::getVertexExponent() const
{
    return m_vertexExponent;
}

void LaBP::WDataSetEMMSurface::setVertexExponent( LaBP::WEExponent::Enum exponent )
{
    m_vertexExponent = exponent;
}

std::vector< WVector3i >& LaBP::WDataSetEMMSurface::getFaces() const
{
    return *m_faces;
}

void LaBP::WDataSetEMMSurface::setFaces( boost::shared_ptr< std::vector< WVector3i > > faces )
{
    m_faces = faces;
}

LaBP::WDataSetEMMSurface::Hemisphere::Enum LaBP::WDataSetEMMSurface::getHemisphere() const
{
    return m_hemisphere;
}
void LaBP::WDataSetEMMSurface::setHemisphere( Hemisphere::Enum val )
{
    m_hemisphere = val;
}
