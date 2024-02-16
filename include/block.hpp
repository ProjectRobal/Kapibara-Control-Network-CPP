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

    template<class NeuronT,size_t Working,size_t Populus>
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
        std::array<std::shared_ptr<Neuron>,Working> workers;

        std::array<std::shared_ptr<Neuron>,Working> best_workers;

        public:

        Block(std::shared_ptr<Crossover> _crossing,std::shared_ptr<Mutation> _mutate)
        : crossing(_crossing),
        mutate(_mutate),
        uniform(0,Populus-1),
        population({NULL}),
        workers({NULL})
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

            for(auto& w : this->workers)
            {
                w=this->population[this->uniform(gen)];
            }
        }

        // store a best partition of workes for later use
        void keepWorkers()
        {
            this->best_workers=this->workers;
        }

        void giveRewardToSavedWorkers(long double reward)
        {
            reward/=this->best_workers.size();

            for(auto& w : this->best_workers)
            {
                w->giveReward(reward);
                
                if(w->used()<USESES_TO_MAITING)
                {
                    w->use();
                    if(w->used()==USESES_TO_MAITING)
                    {
                        ++this->mating_counter;
                    }
                }
            }
        }

        void giveReward(long double reward)
        {
            reward/=this->workers.size();

            for(auto& w : this->workers)
            {
                w->giveReward(reward);
                
                if(w->used()<USESES_TO_MAITING)
                {
                    w->use();
                    if(w->used()==USESES_TO_MAITING)
                    {
                        ++this->mating_counter;
                    }
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

        SIMDVector fire(SIMDVector input)
        {
            SIMDVector output=this->workers.front()->fire(input);

            for(auto iter=this->workers.begin()+1;iter!=this->workers.end();iter++)
            {
                output+=(*iter)->fire(input);
            }

            return output/this->workers.size();
        }       

        size_t inputSize()
        {
            return this->population[0]->input_size();
        }

        size_t outputSize()
        {
            return this->population[0]->output_size();
        }

        void save(std::ofstream& file) const
        {
            // save block header with information about size etc

            BlockHeader header={
                .block_type='@',
                .populus_size=Populus,
                .working_size=Working
            };

            // save header

            file.write((char*)&header,sizeof(BlockHeader));

            // save populus

            for(size_t i=0;i<this->population.size();++i)
            {
                this->population[i]->save(file);
            }

        }

        const std::array<std::shared_ptr<Neuron>,Working>& getWorkers()
        {
            return this->workers;
        }

        bool load(std::ifstream& file)
        {

            // load header

            BlockHeader header={0};

            file.read((char*)&header,sizeof(BlockHeader));

            if(header.block_type != '@')
            {
                std::cerr<<"Wrong block header!!"<<std::endl;
                return false;
            }

            if(header.populus_size != Populus)
            {
                std::cerr<<"Wrong population size!"<<std::endl;
                return false;
            }

            if(header.working_size != Working)
            {
                std::cerr<<"Wrong working population size!"<<std::endl;
                return false;
            }

            for(size_t i=0;i<header.populus_size;i++)
            {
                std::shared_ptr<NeuronT> neuron=std::make_shared<NeuronT>();

                if(! neuron->load(file) )
                {
                    std::cerr<<"Neuron: "<<i<<" corrupted!"<<std::endl;
                    return false;
                }

            }


            return true;
        }

    };
}