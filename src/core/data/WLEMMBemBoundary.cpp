// TODO license

#include <vector>

#include <boost/shared_ptr.hpp>

#include <core/common/math/linearAlgebra/WPosition.h>
#include <core/common/math/linearAlgebra/WVectorFixed.h>

#include "core/dataHandler/WDataSetEMMEnumTypes.h"

#include "WLEMMBemBoundary.h"

LaBP::WLEMMBemBoundary::WLEMMBemBoundary()
{
    setVertexUnit( WEUnit::UNKNOWN_UNIT );
    setVertexExponent( WEExponent::BASE );

    setConductivityUnit( WEUnit::UNKNOWN_UNIT );

    m_vertex.reset( new std::vector< WPosition >( 0 ) );
    m_faces.reset( new std::vector< WVector3i >( 0 ) );
}

LaBP::WLEMMBemBoundary::~WLEMMBemBoundary()
{
}

std::vector< WPosition >& LaBP::WLEMMBemBoundary::getVertex() const
{
    return *m_vertex;
}

void LaBP::WLEMMBemBoundary::setVertex( boost::shared_ptr< std::vector< WPosition > > vertex )
{
    m_vertex = vertex;
}

LaBP::WEUnit::Enum LaBP::WLEMMBemBoundary::getVertexUnit() const
{
    return m_vertexUnit;
}

void LaBP::WLEMMBemBoundary::setVertexUnit( LaBP::WEUnit::Enum unit )
{
    m_vertexUnit = unit;
}

LaBP::WEExponent::Enum LaBP::WLEMMBemBoundary::getVertexExponent() const
{
    return m_vertexExponent;
}

void LaBP::WLEMMBemBoundary::setVertexExponent( LaBP::WEExponent::Enum exponent )
{
    m_vertexExponent = exponent;
}

LaBP::WEBemType::Enum LaBP::WLEMMBemBoundary::getBemType() const
{
    return m_bemType;
}

void LaBP::WLEMMBemBoundary::setBemType( LaBP::WEBemType::Enum type )
{
    m_bemType = type;
}

std::vector< WVector3i >& LaBP::WLEMMBemBoundary::getFaces() const
{
    return *m_faces;
}

void LaBP::WLEMMBemBoundary::setFaces( boost::shared_ptr< std::vector< WVector3i > > faces )
{
    m_faces = faces;
}

float LaBP::WLEMMBemBoundary::getConductivity() const
{
    return m_conductivity;
}

void LaBP::WLEMMBemBoundary::setConductivity( float conductivity )
{
    m_conductivity = conductivity;
}

LaBP::WEUnit::Enum LaBP::WLEMMBemBoundary::getConductivityUnit() const
{
    return m_conductivityUnit;
}

void LaBP::WLEMMBemBoundary::setConductivityUnit( LaBP::WEUnit::Enum unit )
{
    m_conductivityUnit = unit;
}
