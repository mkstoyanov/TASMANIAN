/*
 * Copyright (c) 2017, Miroslav Stoyanov
 *
 * This file is part of
 * Toolkit for Adaptive Stochastic Modeling And Non-Intrusive ApproximatioN: TASMANIAN
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
 *    and the following disclaimer in the documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse
 *    or promote products derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * UT-BATTELLE, LLC AND THE UNITED STATES GOVERNMENT MAKE NO REPRESENTATIONS AND DISCLAIM ALL WARRANTIES, BOTH EXPRESSED AND IMPLIED.
 * THERE ARE NO EXPRESS OR IMPLIED WARRANTIES OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, OR THAT THE USE OF THE SOFTWARE WILL NOT INFRINGE ANY PATENT,
 * COPYRIGHT, TRADEMARK, OR OTHER PROPRIETARY RIGHTS, OR THAT THE SOFTWARE WILL ACCOMPLISH THE INTENDED RESULTS OR THAT THE SOFTWARE OR ITS USE WILL NOT RESULT IN INJURY OR DAMAGE.
 * THE USER ASSUMES RESPONSIBILITY FOR ALL LIABILITIES, PENALTIES, FINES, CLAIMS, CAUSES OF ACTION, AND COSTS AND EXPENSES, CAUSED BY, RESULTING FROM OR ARISING OUT OF,
 * IN WHOLE OR IN PART THE USE, STORAGE OR DISPOSAL OF THE SOFTWARE.
 */

#ifndef __TASMANIAN_THREAD_ENGINE_HPP
#define __TASMANIAN_THREAD_ENGINE_HPP

#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <deque>
#include <vector>
#include <functional>
#include <iostream>

namespace TasGrid{

class ThreadEngine{
private:
    using Job = std::function<void(void)>;
    enum status{ status_idle, status_working };
public:
    ThreadEngine(size_t cnum_threads) : num_threads(cnum_threads), shutdown(0), thread_status(cnum_threads, status_idle){

        workers.reserve(num_threads);
        for(size_t j=0; j<num_threads; j++)
            workers.emplace_back([&, j](void)->
                void{

                    while(shutdown == 0){ // until we get the shutdown signal

                        std::unique_ptr<Job> checked_out;
                        {
                            std::unique_lock<std::mutex> lock(jobs);
                            new_job.wait(lock, [&]()->bool{ return ((!job_queue.empty()) || (shutdown != 0)); });
                            // the lock is on at this point
                            if (shutdown == 0 && (!job_queue.empty())){
                                thread_status[j] = status_working;
                                checked_out = std::move(job_queue.front());
                                job_queue.pop_front();
                            }
                        }

                        while(checked_out){
                            //std::cout << "Computing: " << j << std::endl;
                            (*checked_out.get())(); // do actual work
                            //std::cout << "Computed: " << j << std::endl;

                            {
                                std::lock_guard<std::mutex> lock(jobs); // checkout the next item
                                if (job_queue.empty()){
                                    checked_out.reset(); // nothing more to do
                                }else{
                                    checked_out = std::move(job_queue.front());
                                    job_queue.pop_front();
                                }
                            }
                        }
                        //std::cout << "Done: " << j << std::endl;

                        if (shutdown == 0){ // if still working, report that we are now idle
                            {
                                std::lock_guard<std::mutex> lock(jobs);
                                thread_status[j] = status_idle;
                            } // unlock the worker state so the main thread can read it
                            new_status.notify_one(); // tell the main thread that we are done
                        }

                    }

                });

    }
    ~ThreadEngine(){
        {
            std::lock_guard<std::mutex> lock(jobs);
            shutdown = 1;
        }
        new_job.notify_all(); // tell everyone to check the new job, which is "ignore the queue and shutdown"

        for(auto &w : workers) w.join();
    }

    size_t getNumThreads() const{ return num_threads; }

    void compute(int num_runs, std::function<void(int i)> work){
        if (num_runs == 0) return;
        //std::cout << "threads = " << num_threads << " run = " << num_runs << std::endl;

        int num_blocks = (int) (16 * num_threads);
        int whole_jumps = num_runs / num_blocks;
        int rem_jumps = num_runs % num_blocks;
        //std::cout << "blocks = " << num_blocks <<  "  wj = " << whole_jumps << "  rj = " << rem_jumps << std::endl;
        auto get_offset = [&](int j)->int{ return static_cast<int>(j * whole_jumps + std::min(j, rem_jumps)); };

        { // enqueue the jobs
            std::lock_guard<std::mutex> lock(jobs);

            for(int b=0; b<num_blocks; b++){
                job_queue.emplace_back(std::make_unique<Job>(
                    [&, b](void)->void{
                        //std::cout << "Working on: " << b << "  " << get_offset(b) << "  " << get_offset(b+1) << std::endl;
                        for(int i=get_offset(b); i<get_offset(b+1); i++) work(i);
                    }
                ));
            }
            //std::cout << "enqueued: " << job_queue.size() << std::endl;
        }
        new_job.notify_all();

        { // wait till all threads go back to idle state, i.e., finish the queue
            std::unique_lock<std::mutex> lock(jobs);
            new_status.wait(lock, [&]()->bool{
                return job_queue.empty() && std::none_of(thread_status.begin(), thread_status.end(), [&](status &w)->bool{ return (w == status_working); });
            });
        }
    }

private:

    size_t num_threads;
    std::atomic<int> shutdown;

    std::vector<std::thread> workers;

    std::mutex jobs;

    std::vector<status> thread_status;
    std::deque<std::unique_ptr<Job>> job_queue;

    std::condition_variable new_job;
    std::condition_variable new_status;
};


}

#endif
