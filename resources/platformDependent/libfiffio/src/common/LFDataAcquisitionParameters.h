#ifndef LFDATAACQUISITIONPARAMETERS_H
#define LFDATAACQUISITIONPARAMETERS_H

#include <string>
using std::string;

/**
 * The acquisition setup parameters (117)
 */
class LFDataAcquisitionParameters
{
protected:
    string m_DAcqPars;/**< Megacq parameters (150) */
    string m_DAcqStim;/**< Megacq stimulus parameters (151) */
public:
    /**
     * Sets all member variables to defaults
     */
    void Init();
    /**
     * Returns megacq parameters
     */
    string& GetDAcqPars();
    /**
     * Returns megacq stimulus parameters
     */
    string& GetDAcqStim();
};

#endif
