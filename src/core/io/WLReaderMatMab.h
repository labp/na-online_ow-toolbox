// TODO doc & license
#ifndef WLREADERMATMAB_H_
#define WLREADERMATMAB_H_

#include <string>

#include <boost/shared_ptr.hpp>
#include <Eigen/Core>

#include "core/data/WLDataTypes.h"
#include "core/io/WLReader.h"

class WLReaderMatMab: public WLReader
{
public:
    /**
     * Abbreviation for a shared pointer.
     */
    typedef boost::shared_ptr< WLReaderMatMab > SPtr;

    /**
     * Abbreviation for const shared pointer.
     */
    typedef boost::shared_ptr< const WLReaderMatMab > ConstSPtr;

    /**
     * Constructs a reader object.
     *
     * \param fname path to file which should be loaded
     */
    explicit WLReaderMatMab( std::string fname );
    virtual ~WLReaderMatMab();

    ReturnCode::Enum read( WLMatrix::SPtr& matrix );

private:
    ReturnCode::Enum readMab( WLMatrix::SPtr matrix, std::string fName, size_t rows, size_t cols );
};

#endif /* WLREADERMATMAB_H_ */
