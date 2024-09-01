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

    template<size_t Populus,size_t MaxSpeciesCount=4>
    class SubBlock
    {   

        std::shared_ptr<Mutation> mutate;

        size_t mating_counter;

        std::mt19937 gen; 
        number std;

        std::normal_distribution<number> distribution[MaxSpeciesCount];
        std::normal_distribution<number> global;

        std::uniform_real_distribution<float> uniform;

        number weight;
        number mean;

        long double reward;
        long double last_reward;

        size_t population_counter;

        SIMDVector pop_weights;
        SIMDVector pop_rewards;

        size_t population_count;

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
            this->mean = 0;

            this->population_counter = 0;

            this->uniform = std::uniform_real_distribution<float>(0.f,1.f);

            this->population_count = 0;
            this->last_reward = -99999;
            
        }

        SubBlock(std::shared_ptr<Mutation> _mutate)
        :SubBlock()
        {
            this->mutate = _mutate;
        }

        void setup(size_t inputSize,std::shared_ptr<Initializer> init)
        {
            this->std = std::sqrt(2.f/inputSize);
            this->distribution[0] = std::normal_distribution<number>(0.f,this->std);   
            this->global = std::normal_distribution<number>(0.f,this->std);  

            this->max_std = this->std;
            this->min_std = MIN_STD;

            this->weight = this->distribution[0](this->gen);

            this->population_count = 1;

            // this->pop_rewards = SIMDVector(0.f,Populus+5);
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

        void estimate_specie(const SIMDVector& weights,size_t specie_id)
        {
            number mean = weights.reduce()/weights.size();

            SIMDVector w = weights - mean;

            w = w*w;

            number std = std::sqrt(w.reduce()/w.size());

            this->distribution[specie_id] = std::normal_distribution<number>(mean,std);
        }

        void split_into_species(const SIMDVector& weights)
        {
            size_t species_total=0; 
            SIMDVector specie;

            specie.append(weights[0]);

            for(size_t i=1;i<weights.size();++i)
            {
                number diff = abs(weights[i]/weights[i-1]);

                // if difference is small enought put it into the same specie
                if((( diff <  1.25f )&&(diff>0.1f))||( species_total == MaxSpeciesCount-1 ))
                { 
                    specie.append(weights[i]);
                }
                else
                {
                    // move to next specie

                    this->estimate_specie(specie,species_total++);
                    
                    specie.clear();
                    specie.append(weights[i]);
                }

            }

            if( specie.size()>0 )
            {
                this->estimate_specie(specie,species_total++);
            }

            this->population_count = species_total;
        }

        void chooseWorkers()
        {

            this->pop_rewards.append(this->reward);
            this->pop_weights.append(this->weight);

            if( this->pop_rewards.size() >= Populus )
            {
                snn::quicksort_mask(this->pop_weights,0,this->pop_weights.size()-1,this->pop_rewards);

                snn::SIMDVector best;

                for(size_t i=0;i<Populus/2;++i)
                {
                    best.append(this->pop_weights[this->pop_weights.size()-1 - i]);
                }

                // split it into species

                this->split_into_species(best);

                this->pop_rewards.clear();
                this->pop_weights.clear();
                this->population_counter = 0;
            }


            number mutation_probability_level = std::max(std::exp(this->reward),static_cast<number>(1.f));


            number mutation_chooser = this->uniform(this->gen);

            if( mutation_chooser > mutation_probability_level )
            {
                this->weight = this->global(this->gen);
            }
            else
            {
                this->weight = this->distribution[this->population_counter](this->gen);

                this->population_counter++;

                if( this->population_counter == this->population_count )
                {
                    this->population_counter = 0;
                }
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

            // uint64_t maiting = this->mating_counter;

            // char maiting_data[sizeof(uint64_t)]={0};

            // out.write(maiting_data,sizeof(uint64_t));
        }

        void load(std::ifstream& in)
        {
            // for(auto neuron : this->population)
            // {
            //     neuron->load(in);
            // }

            // uint64_t maiting = 0;

            // char maiting_data[sizeof(uint64_t)]={0};

            // in.read(maiting_data,sizeof(uint64_t));

            // memmove(reinterpret_cast<char*>(&maiting),maiting_data,sizeof(uint64_t));

            // this->mating_counter=maiting;
        }

        
    };

}