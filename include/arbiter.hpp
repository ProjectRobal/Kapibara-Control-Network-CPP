#pragma once

/*
    A class responsible for saving/loading layers 

*/

#include <memory>
#include <mutex>
#include <iostream>
#include <list>
#include <vector>
#include <cstdint>
#include <filesystem>

#include <thread>

#include <openssl/sha.h>

#include "layer.hpp"

namespace snn
{

    class Arbiter
    {
        std::vector<std::shared_ptr<Layer>> layers;

        std::string get_sha256_file(const std::string& filename) const
        {
            return filename+".sha256";
        }

        std::string get_backup_file(const std::string& filename) const
        {
            return filename+".bak";
        }

        bool calculate_sha256_for_file(const std::string& filename,char* hash) const
        {
            std::fstream file;

            file.open(filename,std::ios::in|std::ios::binary);

            if(!file.good())
            {
                return false;
            }

            char buff[32768];

            // SHA256 hash 
            SHA256_CTX sha256;

            if(!SHA256_Init(&sha256))
            {
                return false;
            }

            while(file.good())
            {
                file.read(buff,32768);

                size_t readed = file.gcount();

                SHA256_Update(&sha256,buff,readed);
            }

            SHA256_Final((unsigned char*)hash,&sha256);

            file.close();

            return true;
        }

        bool create_sha256_file(const std::string& filename) const
        {
            char buff[32768];

            // SHA256 hash 
            char hash[SHA256_DIGEST_LENGTH];

            if(!this->calculate_sha256_for_file(filename,hash))
            {
                return false;
            }

            std::string hash_filename = this->get_sha256_file(filename);

            std::fstream file;

            file.open(hash_filename,std::ios::out|std::ios::binary);

            file.write(hash,SHA256_DIGEST_LENGTH);

            if(!file.good())
            {
                file.close();

                return false;
            }

            file.close();

            return true;
        }


        public:

        void addLayer(std::shared_ptr<Layer> layer)
        {
            layers.push_back(layer);
        }

        void setup()
        {
            for(std::shared_ptr<Layer> layer : this->layers)
            {
                layer->setup();
            }
        }

        static void shuttle_thread(const std::vector<std::shared_ptr<Layer>>& layers,size_t start,size_t end)
        {
            while(start<end)
            {
                layers[start]->shuttle();
                start++;
            }
        }

        static void concurrent_shuttle(const std::vector<std::shared_ptr<Layer>>& layers,size_t ptr,size_t& free_slot,std::mutex& mux)
        {
            while(ptr<layers.size())
            {
                layers[ptr]->shuttle();

                std::lock_guard _mux(mux);

                ptr = free_slot;

                free_slot++;

                if(free_slot>=layers.size())
                {
                    return;
                }

            }
        }

        void shuttle()
        {

            // for(std::shared_ptr<Layer> layer : this->layers)
            // {
            //     layer->shuttle();
            // }

            // return;

            std::thread workers[USED_THREADS];

            size_t free_slot;

            std::mutex _mux;

            _mux.lock();

            for(size_t i=0;i<USED_THREADS;++i)
            {
                workers[i] = std::thread(concurrent_shuttle,std::ref(this->layers),i,std::ref(free_slot),std::ref(_mux));
            }

            free_slot = USED_THREADS;

            _mux.unlock();

            for(auto& worker : workers)
            {
                worker.join();   
            }

        }

        void applyReward(long double reward)
        {
            for(std::shared_ptr<Layer> layer : this->layers)
            {
                layer->applyReward(reward);
            }
        }

        int8_t save() const
        {
            for(std::shared_ptr<Layer> layer : this->layers)
            {
                int8_t ret = layer->save();

                if( ret != 0 )
                {
                    return ret;
                }
            }

            return 0;
        }

        

        int8_t save(const std::string& filename) const
        {
            std::string sha256_file = this->get_sha256_file(filename);

            // set current file as backup
            if( std::filesystem::exists(filename) && std::filesystem::exists(sha256_file) )
            {
                std::string backup_file = this->get_backup_file(filename);

                std::filesystem::rename(filename,backup_file);

                // backup sha256 module
                std::string sha256_backup_file = this->get_sha256_file(backup_file);

                std::filesystem::rename(sha256_file,sha256_backup_file);

            }

            // save file

            std::fstream file;

            file.open(filename,std::ios::out|std::ios::binary);

            int8_t ret = 0;

            if(!file.good())
            {
                return -15;
            }

            ret = this->save(file);

            file.close();

            // create file with ssh key

            if(!create_sha256_file(filename))
            {
                return -16;
            }

            return ret;
        }

        int8_t save(std::ostream& out) const
        {
            for(std::shared_ptr<Layer> layer : this->layers)
            {
                int8_t ret = layer->save(out);

                if( ret != 0 )
                {
                    return ret;
                }
            }

            return 0;
        }

        int8_t load() const
        {
            for(std::shared_ptr<Layer> layer : this->layers)
            {
                int8_t ret = layer->load();

                if( ret != 0 )
                {
                    return ret;
                }
            }

            return 0;
        }

        int8_t load_backup(const std::string& filename) const
        {
            std::string filename_backup = this->get_backup_file(filename);

            int8_t ret = this->load(filename_backup);

            if( ret == 0 )
            {
                // copy backup file
                std::filesystem::copy_file(filename_backup,filename,std::filesystem::copy_options::update_existing);

                std::string backup_hash_file = this->get_sha256_file(filename_backup);
                std::string hash_file = this->get_sha256_file(filename);

                // copy hash file
                std::filesystem::copy_file(backup_hash_file,hash_file,std::filesystem::copy_options::update_existing);
            }

            return ret;
        }

        int8_t load(const std::string& filename) const
        {

            int8_t ret = 0;

            // try open normal file
            std::string hash_file = this->get_sha256_file(filename);

            if(!std::filesystem::exists(hash_file))
            {
                return -14;
            }
            // read hash 

            std::fstream file;

            char file_hash[SHA256_DIGEST_LENGTH];

            file.open(hash_file,std::ios::in|std::ios::binary);

            if(!file.good())
            {
                // read backup
                return -16;
            }

            file.read(file_hash,SHA256_DIGEST_LENGTH);

            if(file.gcount()!=SHA256_DIGEST_LENGTH)
            {
                // read backup
                return -17;
            }

            char hash[SHA256_DIGEST_LENGTH];

            if(!this->calculate_sha256_for_file(filename,hash))
            {
                // read backup
                return -18;
            }

            if(strncmp(file_hash,hash,SHA256_DIGEST_LENGTH)!=0)
            {
                // read backup
                return -19;
            }

            file.close();


            file.open(filename,std::ios::in|std::ios::binary);

            if(!file.good())
            {
                // read backup
                return -15;
            }

            ret = this->load(file);

            file.close();

            return ret;

        }

        int8_t load(std::istream& in) const
        {
            for(std::shared_ptr<Layer> layer : this->layers)
            {
                int8_t ret = layer->load(in);

                if( ret != 0 )
                {
                    return ret;
                }
            }

            return 0;
        }
    };

}