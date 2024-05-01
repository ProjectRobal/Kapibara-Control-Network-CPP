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

namespace snn
{

    template<class NeuronT,size_t Populus>
    class Block
    {     

        struct BlockHeader {

            char block_type;
            size_t populus_size;
            size_t working_size;

        };   

        std::shared_ptr<Crossover> crossing;
        std::shared_ptr<Mutation> mutate;

        size_t mating_counter;

        std::uniform_int_distribution<size_t> uniform;

        std::array<std::shared_ptr<Neuron>,Populus> population;
        std::shared_ptr<Neuron> worker;

        public:

        Block(std::shared_ptr<Crossover> _crossing,std::shared_ptr<Mutation> _mutate)
        : crossing(_crossing),
        mutate(_mutate),
        uniform(0,Populus-1),
        population({NULL}),
        worker(NULL),
        mating_counter(0)
        {

        }

        void setup(std::shared_ptr<Initializer> init)
        {
            for(auto& p : this->population)
            {
                p=std::make_shared<NeuronT>();
                p->setup(init);
            }
        }

        void chooseWorkers()
        {
            std::random_device rd; 

            // Mersenne twister PRNG, initialized with seed from previous random device instance
            std::mt19937 gen(rd()); 

            this->worker=this->population[this->uniform(gen)];
        }


        void giveReward(long double reward)
        {          
            this->worker->giveReward(reward);
            
            if(this->worker->used()<USESES_TO_MAITING)
            {
                this->worker->use();
                if(this->worker->used()==USESES_TO_MAITING)
                {
                    ++this->mating_counter;
                }
            }
            
        }

        bool readyToMate()
        {
            return this->mating_counter >= MAITING_THRESHOLD*Populus;
        }

        void maiting(std::shared_ptr<Initializer> init)
        {
            this->mating_counter=0;

            std::sort(this->population.begin(),this->population.end(),
            [](const std::shared_ptr<Neuron>& a,const std::shared_ptr<Neuron>& b)->bool
            {
                return a->reward()>b->reward();
            });

            size_t keep_of=Populus*AMOUNT_THAT_PASS;

            auto pivot=this->population.begin()+keep_of-1;
            auto iter2=pivot;
            auto iter=this->population.begin();

            for(auto it=this->population.begin();it!=pivot;++it)
            {
                (*it)->reset();
            }

            while(iter<pivot)
            {
                auto ite=iter2;
                (*(ite))=(*iter)->crossover(this->crossing,**(iter+1));
                (*(ite))->mutate(this->mutate);
                ite++;
                iter+=2;
            }

            while(pivot<this->population.end())
            {
                std::shared_ptr<NeuronT> n_neuron=std::make_shared<NeuronT>();
                n_neuron->setup(init);
                n_neuron->mutate(this->mutate);

                (*pivot)=n_neuron;

                pivot++;
            }

        }

        number fire(SIMDVector input)
        {

            return this->worker->fire1(input);

        }       

        size_t inputSize()
        {
            return this->population[0]->input_size();
        }

        size_t outputSize()
        {
            return 1;
        }

        

    };
}