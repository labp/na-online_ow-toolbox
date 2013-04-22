
#include "LFBem.h"

LFBem::LFBem() :
m_BemCoordinateFrame( c_unknown )
{

}

void LFBem::Init()
{
    m_BemCoordinateFrame = c_unknown;
    m_LFCoordTrans.Init();
    m_LFBemSurface.Init();
    m_BemSolutionMatrix.clear();
}

LFBem::coord_t LFBem::GetBemCoordinateFrame()
{
    return m_BemCoordinateFrame;
}

LFCoordTrans& LFBem::GetLFCoordTrans()
{
    return m_LFCoordTrans;
}

LFBemSurface& LFBem::GetLFBemSurface()
{
    return m_LFBemSurface;
}

LFArrayFloat2d& LFBem::GetBemSolutionMatrix()
{
    return m_BemSolutionMatrix;
}

void LFBem::SetBemCoordinateFrame( const LFBem::coord_t src )
{
    m_BemCoordinateFrame = src;
}
