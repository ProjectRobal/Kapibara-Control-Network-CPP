

    return 0;

    long double best_reward=-100;

    snn::ReLu activate;

    std::shared_ptr<snn::ReLu> relu=std::make_shared<snn::ReLu>();

    const size_t input_size=4;
    const size_t output_size=2;
    
    snn::ActorCriticNetwork<input_size,output_size,512,64> network;

    network.setup(10,norm_gauss,cross,mutation);

    std::shared_ptr<snn::Layer<snn::ForwardNeuron<4>,1,256>> layer=std::make_shared<snn::Layer<snn::ForwardNeuron<4>,1,256>>(16,norm_gauss,cross,mutation);
    std::shared_ptr<snn::Layer<snn::ForwardNeuron<16>,1,256>> layer1=std::make_shared<snn::Layer<snn::ForwardNeuron<16>,1,256>>(8,norm_gauss,cross,mutation);
    std::shared_ptr<snn::Layer<snn::ForwardNeuron<8>,1,256>> layer2=std::make_shared<snn::Layer<snn::ForwardNeuron<8>,1,256>>(2,norm_gauss,cross,mutation);

    layer->setActivationFunction(relu);
    layer1->setActivationFunction(relu);
    layer2->setActivationFunction(relu);

    network.addLayer(layer);
    network.addLayer(layer1);
    network.addLayer(layer2);
    
    // we will try to find poles in this polynomials in form of a[0]*x^3 + a[1]*x^2 + a[2] * x + a[3] = 0; 

    snn::SIMDVector inputs({0.25,0.5,0.6,0.4});  

    size_t step=1;  

    size_t maxSteps=50000;

    CartPole interface;

    interface.set_wait(1);
    interface.add_inputs(0);
    interface.add_inputs(0);
    interface.add_inputs(0);
    interface.add_inputs(0);
    interface.add_outputs(0);
    interface.add_outputs(0);
    interface.set_reward(0);


    if(access("fifo",F_OK) != 0)
    {
        if(mkfifo("fifo",0666) == -1)
        {
            std::cerr<<"Cannot create fifo"<<std::endl;
        }
    }

    if(access("fifo_in",F_OK) != 0)
    {
        if(mkfifo("fifo_in",0666) == -1)
        {
            std::cerr<<"Cannot create input fifo"<<std::endl;
        }
    }

    std::string buffer;


    std::cout<<"Starting"<<std::endl;

    int fifo=0;

    /*
    
    Network:    CartPole
            <-  Sends inputs
    Sends outputs ->
            <-  Sends reward
    Loop

    */

   //std::cout<<network.step(inputs)<<std::endl;

    
    while(maxSteps--)
    {

        // recive inputs

        interface=getInterface();

        if(interface.inputs_size()<4)
        {
            continue;
        }

        inputs.set(interface.inputs(0),0);
        inputs.set(interface.inputs(1),1);
        inputs.set(interface.inputs(2),2);
        inputs.set(interface.inputs(3),3);

        std::chrono::time_point start=std::chrono::steady_clock::now();

        snn::SIMDVector outputs=network.step(inputs);

        interface.set_outputs(0,outputs[0]);
        interface.set_outputs(1,outputs[1]);

        // send outputs

        sendInterface(interface);

        // get reward

        interface=getInterface();

        long double reward=interface.reward()-1;


        if(reward>best_reward)
        {
            std::cout<<"Best reward: "<<reward<<" at step: "<<step<<std::endl;
            best_reward=reward;
        }

        std::cout<<"Reward: "<<reward<<std::endl;

        network.applyReward(reward);

        std::chrono::time_point end=std::chrono::steady_clock::now();

        const std::chrono::duration<double> elapsed_seconds(end-start);

        std::cout<<"Time "<<elapsed_seconds<<" s"<<std::endl;

        //std::cout<<"Reward: "<<reward<<std::endl;

        if(!interface.SerializeToString(&buffer))
        {
            return -12;
        }        

        ++step;

    }