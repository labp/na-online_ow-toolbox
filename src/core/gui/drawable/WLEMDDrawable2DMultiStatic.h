/*
 * WLEMDDrawable2DMultiStatic.h
 *
 *  Created on: 16.04.2013
 *      Author: pieloth
 */

#ifndef WLEMDDRAWABLE2DMULTISTATIC_H_
#define WLEMDDRAWABLE2DMULTISTATIC_H_

#include "WLEMDDrawable2DMultiChannel.h"

namespace LaBP
{

    class WLEMDDrawable2DMultiStatic: public LaBP::WLEMDDrawable2DMultiChannel
    {
    public:
        /**
         * Abbreviation for a shared pointer on a instance of this class.
         */
        typedef boost::shared_ptr< WLEMDDrawable2DMultiStatic > SPtr;

        /**
         * Abbreviation for a const shared pointer on a instance of this class.
         */
        typedef boost::shared_ptr< const WLEMDDrawable2DMultiStatic > ConstSPtr;

        const static std::string CLASS;

        WLEMDDrawable2DMultiStatic( WCustomWidget::SPtr widget );
        virtual ~WLEMDDrawable2DMultiStatic();

        virtual void draw( LaBP::WLDataSetEMM::SPtr emm );

        virtual bool hasData() const;

        virtual std::pair< LaBP::WLDataSetEMM::SPtr, size_t > getSelectedData( ValueT pixel ) const;

    protected:
        virtual void osgNodeCallback( osg::NodeVisitor* nv );

        virtual void osgAddChannels( const LaBP::WLEMD* emd );

        LaBP::WLDataSetEMM::SPtr m_emm;
    };

} /* namespace LaBP */
#endif /* WLEMDDRAWABLE2DMULTISTATIC_H_ */
