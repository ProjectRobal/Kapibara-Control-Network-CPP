#pragma once

#include <algorithm>
#include <vector>
#include <array>
#include <memory>
#include <random>
#include <fstream>


#include "simd_vector.hpp"
#include "neuron.hpp"
#include "initializer.hpp"
#include "crossover.hpp"
#include "mutation.hpp"

#include "config.hpp"

#include "misc.hpp"

namespace snn
{

    template<size_t Populus,size_t AmountofMembersThatPass=5>
    class SubBlock
    {   

        std::shared_ptr<Mutation> mutate;

        std::mt19937 gen; 

        std::normal_distribution<number> distribution;
        std::normal_distribution<number> global;

        std::uniform_real_distribution<float> uniform;

        number weight;

        long double reward;

        SIMDVector pop_weights;
        SIMDVector pop_rewards;

        number max_std;
        number min_std;

        public:

        static size_t SubBLockId;

        SubBlock()
        {
            std::random_device rd;

            this->gen = std::mt19937(rd());

            this->weight = 0;
            this->reward = 0;
            
            this->uniform = std::uniform_real_distribution<float>(0.f,1.f);
            
        }

        SubBlock(std::shared_ptr<Mutation> _mutate)
        :SubBlock()
        {
            this->mutate = _mutate;
        }

        void setup(size_t inputSize,std::shared_ptr<Initializer> init)
        {
            number std = std::sqrt(2.f/inputSize);
            this->distribution = std::normal_distribution<number>(0.f,std);   
            this->global = std::normal_distribution<number>(0.f,std);  

            this->max_std = std;
            this->min_std = MIN_STD;

            this->weight = this->distribution(this->gen);

        }

        number get_max()
        {
            number max = 0;
            for(size_t i=1;i<this->pop_weights.size();++i)
            {
                if( pop_rewards[i] > pop_rewards[max] )
                {
                    max = i;
                }
            }

            return this->pop_weights.pop(max);
        }

        void estimate_specie(const SIMDVector& weights)
        {
            number mean = weights.reduce()/weights.size();

            SIMDVector w = weights - mean;

            w = w*w;

            number std = std::sqrt(w.reduce()/w.size());

            this->distribution = std::normal_distribution<number>(mean,std);
        }

        void chooseWorkers()
        {

            // if(this->reward>=0.f)
            // {
            //     this->distribution = std::normal_distribution<number>(this->weight,MIN_STD);
            // }
            // else
            // {
                this->pop_rewards.append(this->reward);
                this->pop_weights.append(this->weight);
            // }

            if( this->pop_rewards.size() >= Populus )
            {
                snn::quicksort_mask(this->pop_weights,0,this->pop_weights.size()-1,this->pop_rewards);

                snn::SIMDVector best;

                for(size_t i=0;i<AmountofMembersThatPass;++i)
                {
                    best.append(this->pop_weights[this->pop_weights.size()-1 - i]);
                }

                // split it into species

                // this->split_into_species(best);

                this->estimate_specie(best);

                this->pop_rewards.clear();
                this->pop_weights.clear();
            }


            // number mutation_probability_level = std::max(std::exp(this->reward),static_cast<number>(1.f));

            number mutation_probability_level = (this->reward<0)*MUTATION_PROBABILITY + (this->reward>=0);

            number mutation_chooser = this->uniform(this->gen);

            if( mutation_chooser < mutation_probability_level )
            {
                this->weight = this->global(this->gen);
            }
            else
            {
                this->weight = this->distribution(this->gen);
            }

            
            this->reward = 0.f;
        }

        // let's tread reward not as reward rather than error
        void giveReward(long double reward)
        {   
            this->reward += reward;
        }

        number get()
        {
            return this->weight;
        }

        number operator()()
        {
            return this->get();
        } 


        void dump(std::ofstream& out) const
        {
            // for(auto neuron : this->population)
            // {
            //     neuron->save(out);
            // }   

            char buffer[SERIALIZED_NUMBER_SIZE];

            serialize_number(this->distribution.mean(),buffer);

            out.write(buffer,SERIALIZED_NUMBER_SIZE);

            serialize_number(this->distribution.stddev(),buffer);

            out.write(buffer,SERIALIZED_NUMBER_SIZE);

        }

        void load(std::ifstream& in)
        {
            char buffer[SERIALIZED_NUMBER_SIZE];

            in.read(buffer,SERIALIZED_NUMBER_SIZE);

            number mean = deserialize_number<number>(buffer);

            in.read(buffer,SERIALIZED_NUMBER_SIZE);

            number std = deserialize_number<number>(buffer);

            this->distribution = std::normal_distribution<number>(mean,std);

        }

        
    };

}