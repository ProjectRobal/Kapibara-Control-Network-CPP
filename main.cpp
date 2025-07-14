#include <experimental/simd>
#include <iostream>
#include <string_view>
#include <cmath>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <numeric>
#include <fstream>
#include <sys/stat.h>
#include <fcntl.h>

#include <opencv2/opencv.hpp>


#include "config.hpp"

#include "simd_vector.hpp"

#include "layer_kac.hpp"
 
#include "initializers/gauss.hpp"
#include "initializers/constant.hpp"
#include "initializers/uniform.hpp"
#include "initializers/hu.hpp"

// #include "layer_sssm.hpp"
// #include "layer_sssm_evo.hpp"

#include "activation/sigmoid.hpp"
#include "activation/relu.hpp"
#include "activation/softmax.hpp"
#include "activation/silu.hpp"


#include "simd_vector_lite.hpp"

#include "layer_counter.hpp"

#include "arbiter.hpp"

#include "kapibara_sublayer.hpp"

#include "RResNet.hpp"

#include "attention.hpp"

#include "static_kan_block.hpp"


size_t get_action_id(const snn::SIMDVector& actions)
{
    std::random_device rd; 

    // Mersenne twister PRNG, initialized with seed from previous random device instance
    std::mt19937 gen(rd()); 

    std::uniform_real_distribution<number> uniform_chooser(0.f,1.f);

    number shift = 0;

    number choose = uniform_chooser(gen);

    size_t action_id = 0;

    for(size_t i=0;i<actions.size();++i)
    {
        if( choose <= actions[i] + shift )
        {
            action_id = i;
            break;
        }

        shift += actions[i];
    }

    return action_id;
}

void send_fifo(const snn::SIMDVector& to_send)
{
    std::fstream fifo;

    fifo.open("fifo",std::ios::out);

    if(!fifo.good())
    {
        std::cerr<<"Cannot open fifo for writing"<<std::endl;
        return;
    }

    fifo<<to_send[0]<<";"<<to_send[1]<<std::endl;

    fifo.close();
}

template<size_t Size>
void send_fifo(const snn::SIMDVectorLite<Size>& to_send)
{
    std::fstream fifo;

    fifo.open("fifo",std::ios::out);

    if(!fifo.good())
    {
        std::cerr<<"Cannot open fifo for writing"<<std::endl;
        return;
    }

    fifo<<to_send[0]<<";"<<to_send[1]<<std::endl;

    fifo.close();
}

snn::SIMDVector read_fifo()
{
    std::fstream fifo;

    fifo.open("fifo_in",std::ios::in);

    if(!fifo.good())
    {
        std::cerr<<"Cannot open fifo for reading"<<std::endl;
        return snn::SIMDVector();
    }

    snn::SIMDVector output;

    std::string line;

    std::getline(fifo,line);

    std::stringstream line_read(line);

    std::string num;

    while(std::getline(line_read,num,';'))
    {
        output.append(std::stof(num,NULL));
    }

    return output;

}

snn::SIMDVectorLite<6> read_fifo_static()
{
    std::fstream fifo;

    fifo.open("fifo_in",std::ios::in);

    if(!fifo.good())
    {
        std::cerr<<"Cannot open fifo for reading"<<std::endl;
        return snn::SIMDVectorLite<6>();
    }

    snn::SIMDVectorLite<6> output;

    std::string line;

    std::getline(fifo,line);

    std::stringstream line_read(line);

    std::string num;

    size_t i=0;

    while(std::getline(line_read,num,';'))
    {
        output[i] = static_cast<number>(std::stof(num,NULL));

        i+=1;
    }

    return output;

}

#include "block_kac.hpp"

size_t snn::BlockCounter::BlockID = 0;

size_t snn::LayerCounter::LayerIDCounter = 0;

/*

    KapiBara input variables:

    quanterion - 4 values
    speed from encoders - 2 values
    spectogram 16x16 - 256 values
    2d points array from camera, compressed to 16x16 - 256 values
    face embeddings - 64 values when more than two faces are spotted average thier embeddings

    Total 518 values


*/

template<size_t N>
size_t max_id(const snn::SIMDVectorLite<N>& p)
{
    size_t max_i = 0;

    for( size_t i=1 ; i <N ; ++i )
    {
        if( p[i] > p[max_i] )
        {
            max_i = i;
        }
    }

    return max_i;
}


template<size_t N>
long double cross_entropy_loss(const snn::SIMDVectorLite<N>& p1,const snn::SIMDVectorLite<N>& p2)
{
    number loss = 0.f;

    for(size_t i=0;i<N;++i)
    {
        number v = -p1[i]*std::log(p2[i]+0.000000001f);

        loss += v;
    }

    return loss;
}

/*

    Algorithm is great but it fails when:

    - Inputs are too much correlated with each other ( are very similar to each other )
    - Outputs are small ( well I had to decrease the error threshold for points nuding )


*/

template<size_t Size>
number variance(const snn::SIMDVectorLite<Size>& input)
{
    number mean = input.reduce() / Size;

    snn::SIMDVectorLite m_mean = input - mean;

    m_mean = m_mean*m_mean;

    return m_mean.reduce() / Size;
}

/*

    Idea is simple we take image and then using convolution and KAN layer generate map of rewards in 2D space.
    
    To fit data we generate noise mask and makes convolution fit to it then that 
    noise map we give to output layer.

    I don't have better idea for now, I should probably think about some defragmentation algorithm for it.


    What can we do to speed it up?

    We can have fixed number of nodes in activations, so we can avoid using std::vectors.

    We can use uint16_t instead of floats for some speed ups, sounds like a good idea generally, but it doesn't give significant speed ups.

    I get rid of smart pointers and it gave major speed ups.

*/

struct SplineNode
{
    number x;
    number y;

    SplineNode(number x,number y)
    {
        this->x = x;
        this->y = y;
    }
};


class Spline
{
    protected:

    std::vector<SplineNode*> nodes;

    public:

    Spline()
    {
        this->nodes.reserve(1024);
    }

    void fit(number x,number y)
    {

        SplineNode* node = new SplineNode(x,y);

        this->nodes.push_back(node);

        std::sort(this->nodes.begin(),this->nodes.end(),[](SplineNode* a, SplineNode* b)
                {
                    return a->x < b->x;
                });
    }

    std::pair<SplineNode*,SplineNode*> search(number x)
    {

        if( this->nodes.size() == 0 )
        {
            return std::pair<SplineNode*,SplineNode*>(nullptr,nullptr);
        }

        if( this->nodes.size() == 1 )
        {
            return std::pair<SplineNode*,SplineNode*>(this->nodes[0],this->nodes[0]);
        }

        if( x < this->nodes[0]->x )
        {
            return std::pair<SplineNode*,SplineNode*>(nullptr,this->nodes[0]);
        }
        
        if( x > this->nodes[this->nodes.size()-1]->x )
        {
            return std::pair<SplineNode*,SplineNode*>(this->nodes[0],nullptr);
        }


        size_t p = 0;
        size_t q = this->nodes.size()-1;

        size_t center = (p+q)/2;

        while( (q-p) > 1 )
        {
            SplineNode* center_node = this->nodes[center];

            if( x > center_node->x )
            {
                p = center;
            }
            else if( x < center_node->x )
            {
                q = center;
            }
            else
            {
                return std::pair<SplineNode*,SplineNode*>(center_node,this->nodes[center+1]);
            }

            center = (p+q)/2;
        }

        SplineNode* left = this->nodes[center];

        SplineNode* right = this->nodes[center+1];

        return std::pair<SplineNode*,SplineNode*>(left,right);

    }

    number fire(number x)
    {
        if( nodes.size() == 0 )
        {
            return 0.f;
        }

        std::pair<SplineNode*,SplineNode*> nodes = this->search(x);

        if( !nodes.first && !nodes.second )
        {
            return 0.f;
        }

        if( nodes.first && !nodes.second )
        {
            return nodes.first->x == x ? nodes.first->y : 0;
        }

        if( !nodes.first && nodes.second )
        {
            return nodes.second->x == x ? nodes.second->y : 0;
        }

        if( nodes.first->x == nodes.second->x )
        {
            return nodes.first->x == x ? nodes.first->y : 0; 
        }   

        // let's use linear approximation

        number _x = x - nodes.first->x;

        SplineNode* left = nodes.first;
        SplineNode* right = nodes.second;

        number a = ( right->y - left->y )/( right->x - left->x );

        return a*_x + left->y;

    }

    ~Spline()
    {
        for( SplineNode* node : this->nodes )
        {
            delete node;
        }

        this->nodes.clear();
    }

};


template<size_t inputSize>
class EvoKan
{
    protected:

    Spline* splines;

    snn::SIMDVectorLite<inputSize> w;

    public:

    EvoKan()
    {
        this->splines = new Spline[inputSize];
    }

    void fit(const snn::SIMDVectorLite<inputSize>& input,number output,number target)
    {

        number tar = target/static_cast<number>(inputSize);

        for(size_t i=0;i<inputSize;++i)
        {
            this->splines[i].fit(input[i],tar);
        }

    }

    number fire(const snn::SIMDVectorLite<inputSize>& input)
    {

        snn::SIMDVectorLite<inputSize> x_left;
        snn::SIMDVectorLite<inputSize> y_left;

        snn::SIMDVectorLite<inputSize> x_right;
        snn::SIMDVectorLite<inputSize> y_right;

        snn::SIMDVectorLite<inputSize> a;

        for(size_t i=0;i<inputSize;++i)
        {
            number _x = input[i];

            std::pair<SplineNode*,SplineNode*> nodes = this->splines[i].search(_x);

            // when both nodes are valid

            SplineNode* left = nodes.first;
            SplineNode* right = nodes.second;

            if( !left && !right)
            {
                y_left[i] = 0.f;
                y_right[i] = 0.f;

                x_left[i] = 0.f;
                x_right[i] = 1.f;

                continue;
            }

            if( left && !right )
            {
                y_left[i] = 0.f;
                y_right[i] = left->x == input[i] ? left->y : 0;

                x_left[i] = 0.f;
                x_right[i] = 1.f;

                continue;
            }

            if( !left && right )
            {
                y_left[i] = 0.f;
                y_right[i] = right->x == input[i] ? right->y : 0;

                x_left[i] = 0.f;
                x_right[i] = 1.f;

                continue;
            }

            if( left == right )
            {

                y_left[i] = 0.f;
                y_right[i] = left->y;

                x_left[i] = 0.f;
                x_right[i] = 1.f;

                continue;
            }

            x_left[i] = left->x;
            y_left[i] = left->y;

            x_right[i] = right->x;
            y_right[i] = right->y;

        }

        snn::SIMDVectorLite<inputSize> x = input - x_left;

        a = ( y_right - y_left )/(x_right - x_left);

        this->w = a*x + y_left;

        return w.reduce();
    }

    ~EvoKan()
    {
        delete [] this->splines;
    }

};

template<size_t Size>
void test_simd()
{
    snn::SIMDVectorLite<Size-1> a(1);

    assert( a.reduce() == Size-1 );

    snn::SIMDVectorLite<Size-2> b(1);

    assert( b.reduce() == Size-2 );

    snn::SIMDVectorLite<Size> x1;

    for(size_t i=0;i<Size;++i)
    {
        x1[i] = i;
        assert(x1[i] == i);
    }

    snn::SIMDVectorLite<Size> x2;

    for(size_t i=0;i<Size;++i)
    {
        x2[i] = i;
        assert(x2[i] == i);
    }

    snn::SIMDVectorLite<Size> x = x1 + x2;

    for(size_t i=0;i<Size;++i)
    {
        assert(x[i] == i+i);
    }

    x = x1-x2;

    for(size_t i=0;i<Size;++i)
    {
        assert(x[i] == i-i);
    }

    x = x1*x2;

    for(size_t i=0;i<Size;++i)
    {
        assert(x[i] == i*i);
    }

    x2[0] = 1.f;

    x = x1/x2;

    for(size_t i=1;i<Size;++i)
    {
        assert(x[i] > 0.99f && x[i] < 1.01f);
    }

}

int main(int argc,char** argv)
{
    std::cout<<"Starting..."<<std::endl;

    std::cout<<"SIMD 19 length test"<<std::endl;
    test_simd<19>();
    std::cout<<"Passed"<<std::endl;

    std::cout<<"SIMD 32 length test"<<std::endl;
    test_simd<32>();
    std::cout<<"Passed"<<std::endl;

    std::cout<<"SIMD 33 length test"<<std::endl;
    test_simd<33>();
    std::cout<<"Passed"<<std::endl;

    std::cout<<"SIMD 96 length test"<<std::endl;
    test_simd<96>();
    std::cout<<"Passed"<<std::endl;

    std::cout<<"SIMD 100 length test"<<std::endl;
    test_simd<100>();
    std::cout<<"Passed"<<std::endl;


    std::chrono::time_point<std::chrono::system_clock> start, end;

    const size_t samples_count = 32;


    snn::UniformInit<(number)-0.5f,(number)0.5f> noise;

    snn::UniformInit<(number)0.f,(number)1.f> chooser;

    const size_t dataset_size = 16;

    snn::SIMDVectorLite<64> dataset[dataset_size];

    number outputs[dataset_size];

    for(auto& input : dataset)
    {
        for(size_t i=0;i<64;++i)
        {
            input[i] = noise.init()*10.f;
        }

    }

    for(size_t i=0;i<dataset_size;++i)
    {
        outputs[i] = noise.init()*10.f;
    }
    
    start = std::chrono::system_clock::now();

    number output_last = 0;

    end = std::chrono::system_clock::now();

    EvoKan<64> kan;

    std::cout<<"Dataset:"<<std::endl;

    snn::SIMDVectorLite<64> mask;

    for(size_t i=0;i<64;++i)
    {
        if( noise.init()+ 0.5f < 0.5f)
        {
            mask[i] = noise.init() + 0.5f;
        }
        else
        {
            mask[i] = 0.f;
        }
    }

    mask /= mask.reduce();

    for(size_t i=0;i<dataset_size;++i)
    {
        kan.fit(dataset[i],0.f,outputs[i]);
       
        // std::cout<<"x: "<<x<<" y: "<<y<<std::endl;

    }

    std::cout<<"Test fit:"<<std::endl;

    number error = 0.f;

    for(size_t i=0;i<dataset_size;++i)
    {
        number output = 0.f;

        number y = outputs[i];

        output = kan.fire(dataset[i]);

        error += abs( y - output );
    }

    std::cout<<"Error: "<<error/dataset_size<<std::endl;

    char c;

    std::cin>>c;

    // Save plot

    number x_min = -10.f;
    number x_max = 10.f;

    const number step = 0.01f;

    std::fstream file;
    
    file.open("test_plot.csv",std::ios::out);

    snn::SIMDVectorLite<64> input;

    while( x_min <= x_max )
    {
        input[0] = x_min;

        number _y = kan.fire(input);

        file<<x_min<<";"<<_y<<std::endl;

        x_min += step;
    }

    file.close();
    

    return 0;
}

