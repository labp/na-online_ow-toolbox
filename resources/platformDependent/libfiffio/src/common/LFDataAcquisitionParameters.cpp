#include "LFDataAcquisitionParameters.h"

void LFDataAcquisitionParameters::Init()
{
    m_DAcqPars.clear();
    m_DAcqStim.clear();
}

string& LFDataAcquisitionParameters::GetDAcqPars()
{
    return m_DAcqPars;
}

string& LFDataAcquisitionParameters::GetDAcqStim()
{
    return m_DAcqStim;
}
