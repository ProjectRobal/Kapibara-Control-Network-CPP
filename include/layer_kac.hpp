#pragma once

#include <array>
#include <functional>
#include <fstream>
#include <algorithm>
#include <string>

#include "block_kac.hpp"
#include "initializer.hpp"

#include "simd_vector.hpp"
#include "simd_vector_lite.hpp"

#include "config.hpp"

#include "activation.hpp"
#include "activation/linear.hpp"

#include "layer.hpp"

#include "layer_utils.hpp"

#include <filesystem>

#include "nlohmann/json.hpp"

#include "layer_counter.hpp"

#include "initializers/hu.hpp"
/*


 A layer that use evolutionary algorithm for learing called CoSyne.

*/
namespace snn
{    
    template<size_t inputSize,size_t N,size_t Populus,class Activation = Linear,class weight_initializer = HuInit<inputSize>>
    class LayerKAC : public Layer
    { 
        const uint32_t LAYER_KAC_ID = 2148;

        BlockKAC<inputSize,Populus,weight_initializer>* blocks;

        std::uniform_real_distribution<double> uniform;

        size_t id;

        struct metadata
        {
            uint32_t id;
            size_t input_size;
            size_t node_size;
            size_t population_size;
        };
        
        public:

        LayerKAC()
        {
            this->blocks = new BlockKAC<inputSize,Populus>[N];

            this->id = LayerCounter::LayerIDCounter++ ;

            this->uniform=std::uniform_real_distribution<double>(0.f,1.f);
        }

        void setup()
        {

            for(size_t i=0;i<N;++i)
            {
                // this->blocks[i]= BlockKAC<inputSize,Populus>();
                this->blocks[i].setup();
                // this->blocks.back().chooseWorkers();
            }
        }

        void applyReward(long double reward)
        {

            // reward/=this->blocks.size();

            for(size_t i=0;i<N;++i)
            {
                this->blocks[i].giveReward(reward);
            }
        }

        void shuttle()
        {
            
            for(size_t i=0;i<N;++i)
            {
                this->blocks[i].chooseWorkers();
            }   
        }

        SIMDVectorLite<N> fire(const SIMDVectorLite<inputSize>& input)
        {
            SIMDVectorLite<N> output(0);


            for(size_t i=0;i<N;++i)
            {
                number v = this->blocks[i].fire(input);
                output[i] = v;

            }

            Activation::activate(output);

            return output;
        }

        int8_t load()
        {
            std::string filename = "layer_"+std::to_string(this->id)+".layer";

            std::ifstream file;

            file.open(filename,std::ios::in);

            if(!file.good())
            {
                return -2;
            }

            int8_t ret = this->load(file);

            file.close();

            return ret;

        }

        int8_t save() const
        {
            std::string filename = "layer_"+std::to_string(this->id)+".layer";

            std::ofstream file;

            file.open(filename,std::ios::out);

            if(!file.good())
            {
                return -2;
            }

            int8_t ret = this->save(file);

            file.close();

            return ret;

        }

        int8_t load(std::istream& in)
        {

            LayerKAC::metadata meta={0};

            in.read((char*)&meta,sizeof(LayerKAC::metadata));

            if( meta.id != LayerKAC::LAYER_KAC_ID || meta.input_size != inputSize || meta.node_size != N || meta.population_size != Populus )
            {
                return -3;
            }

            for(size_t i=0;i<N;++i)
            {
                if(in.good())
                {
                    this->blocks[i].load(in);
                }
                else
                {
                    return -1;
                }
            }

            return 0;
        }

        int8_t save(std::ostream& out) const
        {

            LayerKAC::metadata meta = {
                .id = LayerKAC::LAYER_KAC_ID,
                .input_size = inputSize,
                .node_size = N,
                .population_size = Populus
            };

            out.write((char*)&meta,sizeof(LayerKAC::metadata));
            
            for(size_t i=0;i<N;++i)
            {
                if(out.good())
                {
                    this->blocks[i].dump(out);
                }
                else
                {
                    return -1;
                }
            }

            return 0;
        }        

        ~LayerKAC()
        {
            delete [] this->blocks;
        }

    };  
    
} // namespace snn
