#pragma once

#include <iostream>
#include <memory>
#include <fstream>

#include "simd_vector.hpp"
#include "initializer.hpp"
#include "crossover.hpp"
#include "mutation.hpp"

#include "config.hpp"

namespace snn
{

    class Neuron
    {
        protected:

        struct NeuronHeader
        {
            char header='$';
            size_t input_size;
            size_t output_size;
            long double score;
            size_t use_count;
        };

        long double score;
        size_t use_count;

        bool validateHeader(const NeuronHeader& header) const
        {
            return header.header == '$';
        }

        NeuronHeader getHeader(const size_t& input_size,const size_t& output_size) const
        {
            return {
                .input_size=input_size,
                .output_size=output_size,
                .score=this->score,
                .use_count=this->use_count
                };
        }

        public:

        const size_t id=0;

        Neuron()
        : score(0.f),
        use_count(0)
        {}

        virtual std::shared_ptr<Neuron> crossover(std::shared_ptr<Crossover> cross,const Neuron& neuron)=0;

        virtual void mutate(std::shared_ptr<Mutation> mutate)=0;

        virtual void setup(std::shared_ptr<Initializer> init)=0;

        virtual SIMDVector fire(const SIMDVector& input)
        {
            return SIMDVector(0,1);
        }

        virtual number fire1(const SIMDVector& input)
        {
            return 0;
        }

        virtual size_t input_size()=0;

        virtual size_t output_size()=0;

        virtual void giveReward(const long double& score)
        {
            this->score+=score;
        }

        const long double& reward() const
        {
            return this->score;
        }

        virtual void use()
        {
            ++this->use_count;
        }

        const size_t& used() const
        {
            return this->use_count;
        }

        Neuron& operator++()
        {
            this->use();

            return *this;
        }

        virtual void reset()
        {
            this->score=0;
            this->use_count=0;
        }

        virtual bool operator < (const Neuron& neuron) const
        {
            return this->score < neuron.score;
        }

        virtual bool operator > (const Neuron& neuron) const
        {
            return this->score > neuron.score;
        }

        virtual bool operator <= (const Neuron& neuron) const
        {
            return this->score <= neuron.score;
        }

        virtual bool operator >= (const Neuron& neuron) const
        {
            return this->score >= neuron.score;
        }

        virtual bool operator == (const Neuron& neuron) const
        {
            return this->score == neuron.score;
        }

        virtual bool operator != (const Neuron& neuron) const
        {
            return this->score != neuron.score;
        }

        virtual void save(std::ofstream& file) const =0;

        virtual bool load(std::ifstream& file)=0;

        virtual ~Neuron(){}
    };

}