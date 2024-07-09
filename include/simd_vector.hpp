#pragma once

#include <iostream>
#include <vector>
#include <array>
#include <initializer_list>
#include <functional>

#include "parallel.hpp"

#include "config.hpp"

namespace snn
{

    class SIMDVector
    {

        protected:

        size_t ptr;
        std::vector<SIMD> vec;

        SIMD get_partially_filled_simd(size_t N,number value,number else_number=0.f) const;

        public:

        SIMDVector();

        SIMDVector(std::function<number(size_t)> init_func,size_t N);

        SIMDVector(number v,size_t N);

        SIMDVector(const std::initializer_list<number>& arr);

        SIMDVector(const SIMDVector& vec);

        SIMDVector(SIMDVector&& vec);

        template<typename T>
        static snn::SIMDVector from_array(T *v,size_t N)
        {
            snn::SIMDVector output;
            for(size_t i=0;i<N;++i)
            {
                output.append(v[i]);
            }

            return output;
        }

        template<typename T>
        void to_array(T *v) const
        {
            for(size_t i=0;i<this->size();++i)
            {
                v[i] = this->get(i);
            }
        }

        template<typename T>
        static snn::SIMDVector from_vector(const std::vector<T>& v)
        {
            snn::SIMDVector output;
            for(const T samp : v)
            {
                output.append(samp);
            }

            return output;
        }

        

        void reserve(size_t N);

        void operator=(const SIMDVector& vec);

        void operator=(SIMDVector&& vec);

        void extend(const SIMDVector& vec);

        void set(const number& n, const size_t& i);

        number get(const size_t& i) const;

        number pop();

        number append(number n);

        void append(const SIMD_MASK& mask);

        void append(const SIMD& simd);

        const SIMD& get_block(const size_t& i) const;

        SIMDVector operator+(const SIMDVector& v) const;

        SIMDVector operator-(const SIMDVector& v) const;

        SIMDVector operator*(const SIMDVector& v) const;

        SIMDVector operator/(const SIMDVector& v) const;

        SIMDVector operator*(number v) const;

        SIMDVector operator/(number v) const;

        SIMDVector operator-(number v) const;

        SIMDVector operator+(number v) const;

        SIMDVector operator==(const SIMDVector& v) const;

        SIMDVector operator!=(const SIMDVector& v) const;

        SIMDVector operator>=(const SIMDVector& v) const;

        SIMDVector operator<=(const SIMDVector& v) const;

        SIMDVector operator>(const SIMDVector& v) const;

        SIMDVector operator<(const SIMDVector& v) const;

        SIMDVector operator==(number v) const;

        SIMDVector operator!=(number v) const;

        SIMDVector operator>=(number v) const;

        SIMDVector operator<=(number v) const;

        SIMDVector operator>(number v) const;

        SIMDVector operator<(number v) const;

        void operator+=(const SIMDVector& v);

        void operator-=(const SIMDVector& v);

        void operator*=(const SIMDVector& v);

        void operator/=(const SIMDVector& v);

        void operator*=(number v);

        void operator/=(number v);

        void operator-=(number v);

        void operator+=(number v);

        SIMDVector operator-() const
        {
            return (*this)*-1;
        }

        size_t size() const
        {
            return (this->vec.size()-1)*MAX_SIMD_VECTOR_SIZE + ( this->ptr );
        }

        size_t block_count() const
        {
            return this->vec.size();
        }

        number reduce() const;

        number length() const;

        number operator[](const size_t& i) const;

        void clear()
        {
            this->ptr=0;
            this->vec.clear();
        }

        void copy_metadata(const SIMDVector& vec)
        {
            this->ptr=vec.ptr;
        }

        void print(std::ostream& out) const
        {
            for(size_t i=0;i<this->size();++i)
            {
                out<<(*this)[i]<<" ";
            }
        }

        ~SIMDVector();

    };


}

std::ostream& operator<<(std::ostream& out,const snn::SIMDVector& vec);

