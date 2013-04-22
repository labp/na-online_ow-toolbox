// TODO license

#include <vector>

#include <boost/shared_ptr.hpp>

#include "core/common/math/linearAlgebra/WPosition.h"
#include "core/common/math/linearAlgebra/WVectorFixed.h"

#include "WDataSetEMMEnumTypes.h"

#include "WDataSetEMMBemBoundary.h"

LaBP::WDataSetEMMBemBoundary::WDataSetEMMBemBoundary()
{
    setVertexUnit( WEUnit::UNKNOWN_UNIT );
    setVertexExponent( WEExponent::BASE );

    setConductivityUnit( WEUnit::UNKNOWN_UNIT );

    m_vertex.reset( new std::vector< WPosition >( 0 ) );
    m_faces.reset( new std::vector< WVector3i >( 0 ) );
}

LaBP::WDataSetEMMBemBoundary::~WDataSetEMMBemBoundary()
{
}

std::vector< WPosition >& LaBP::WDataSetEMMBemBoundary::getVertex() const
{
    return *m_vertex;
}

void LaBP::WDataSetEMMBemBoundary::setVertex( boost::shared_ptr< std::vector< WPosition > > vertex )
{
    m_vertex = vertex;
}

LaBP::WEUnit::Enum LaBP::WDataSetEMMBemBoundary::getVertexUnit() const
{
    return m_vertexUnit;
}

void LaBP::WDataSetEMMBemBoundary::setVertexUnit( LaBP::WEUnit::Enum unit )
{
    m_vertexUnit = unit;
}

LaBP::WEExponent::Enum LaBP::WDataSetEMMBemBoundary::getVertexExponent() const
{
    return m_vertexExponent;
}

void LaBP::WDataSetEMMBemBoundary::setVertexExponent( LaBP::WEExponent::Enum exponent )
{
    m_vertexExponent = exponent;
}

LaBP::WEBemType::Enum LaBP::WDataSetEMMBemBoundary::getBemType() const
{
    return m_bemType;
}

void LaBP::WDataSetEMMBemBoundary::setBemType( LaBP::WEBemType::Enum type )
{
    m_bemType = type;
}

std::vector< WVector3i >& LaBP::WDataSetEMMBemBoundary::getFaces() const
{
    return *m_faces;
}

void LaBP::WDataSetEMMBemBoundary::setFaces( boost::shared_ptr< std::vector< WVector3i > > faces )
{
    m_faces = faces;
}

float LaBP::WDataSetEMMBemBoundary::getConductivity() const
{
    return m_conductivity;
}

void LaBP::WDataSetEMMBemBoundary::setConductivity( float conductivity )
{
    m_conductivity = conductivity;
}

LaBP::WEUnit::Enum LaBP::WDataSetEMMBemBoundary::getConductivityUnit() const
{
    return m_conductivityUnit;
}

void LaBP::WDataSetEMMBemBoundary::setConductivityUnit( LaBP::WEUnit::Enum unit )
{
    m_conductivityUnit = unit;
}
