#include <float.h>
#include "LFBemSurface.h"

LFBemSurface::LFBemSurface() :
    m_SurfaceId( bs_unknown2 ), m_BemSigma( -FLT_MAX ), m_NumberOfNodes( -1 ), m_NumberOfTriangles(-1)
{

}

void LFBemSurface::Init()
{
    m_SurfaceId = bs_unknown2;
    m_BemSigma = -FLT_MAX;
    m_NumberOfNodes = -1;
    m_NumberOfTriangles = -1;
    m_BemSurfaceNodes.clear();
    m_BemSurfaceNormals.clear();
    m_BemSurfaceTriangles.clear();
}

LFBemSurface::bem_surf_id_t LFBemSurface::GetSurfaceId()
{
    return m_SurfaceId;
}

float LFBemSurface::GetBemSigma()
{
    return m_BemSigma;
}

int32_t LFBemSurface::GetNumberOfNodes()
{
    return m_NumberOfNodes;
}

int32_t LFBemSurface::GetNumberOfTriangles()
{
    return m_NumberOfTriangles;
}

LFArrayFloat3d& LFBemSurface::GetBemSurfaceNodes()
{
    return m_BemSurfaceNodes;
}

LFArrayFloat3d& LFBemSurface::GetBemSurfaceNormals()
{
    return m_BemSurfaceNormals;
}

vector< int32_t >& LFBemSurface::GetBemSurfaceTriangles()
{
    return m_BemSurfaceTriangles;
}

void LFBemSurface::SetSurfaceId( const bem_surf_id_t src )
{
    m_SurfaceId = src;
}

void LFBemSurface::SetBemSigma( const float src )
{
    m_BemSigma = src;
}

void LFBemSurface::SetNumberOfNodes( const int32_t src )

{
    m_NumberOfNodes = src;
}

void LFBemSurface::SetNumberOfTriangles( const int32_t src )

{
    m_NumberOfTriangles = src;
}
