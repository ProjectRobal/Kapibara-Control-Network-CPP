#pragma once


#include <memory>

#include "simd_vector_lite.hpp"

#include "layer.hpp"
#include "layer_kac.hpp"

#include "activation/silu.hpp"


class KapiBara_SubLayer : public snn::Layer
{

    std::shared_ptr<snn::LayerKAC<4,64,40,snn::SiLu>> first;
    std::shared_ptr<snn::LayerKAC<64,32,40,snn::SiLu>> second;
    std::shared_ptr<snn::LayerKAC<32,2,40,snn::SiLu>> third;

    public:

    KapiBara_SubLayer()
    {

        this->first = std::make_shared<snn::LayerKAC<4,64,40,snn::SiLu>>();
        this->second = std::make_shared<snn::LayerKAC<64,32,40,snn::SiLu>>();
        this->third = std::make_shared<snn::LayerKAC<32,2,40,snn::SiLu>>();

    }

    void setup()
    {
        this->first->setup();
        this->second->setup();
        this->third->setup();
    }

    void applyReward(long double reward)
    {
        this->first->applyReward(reward);
        this->second->applyReward(reward);
        this->third->applyReward(reward);
    }

    void shuttle()
    {
        this->first->shuttle();
        this->second->shuttle();
        this->third->shuttle();
    }

    snn::SIMDVectorLite<2> fire(const snn::SIMDVectorLite<4>& input)
    {
        snn::SIMDVectorLite output1 = this->first->fire(input);
        snn::SIMDVectorLite output2 = this->second->fire(output1);

        snn::SIMDVectorLite output3 = this->third->fire(output2);

        return output3;
    }

    int8_t load()
    {
        if( this->first->load() < 0 || this->second->load() < 0 || this->third->load() < 0 )
        {
            return -1;
        }

        return 0;
    }

    int8_t save() const
    {
        if( this->first->save() < 0 || this->second->save() < 0 || this->third->save() < 0 )
        {
            return -1;
        }

        return 0;
    }

    int8_t load(std::istream& in)
    {
        if( this->first->load(in) < 0 || this->second->load(in) < 0 || this->third->load(in) < 0 )
        {
            return -1;
        }

        return 0;
    }

    int8_t save(std::ostream& out) const
    {
        if( this->first->save(out) < 0 || this->second->save(out) < 0 || this->third->save(out) < 0 )
        {
            return -1;
        }

        return 0;
    }

};