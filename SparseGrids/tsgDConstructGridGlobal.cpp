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

#ifndef __TASMANIAN_SPARSE_GRID_DYNAMIC_CONST_GLOBAL_CPP
#define __TASMANIAN_SPARSE_GRID_DYNAMIC_CONST_GLOBAL_CPP

#include "tsgDConstructGridGlobal.hpp"

namespace TasGrid{

DynamicConstructorDataGlobal::DynamicConstructorDataGlobal(size_t cnum_dimensions, size_t cnum_outputs)
    : num_dimensions(cnum_dimensions), num_outputs(cnum_outputs){}
DynamicConstructorDataGlobal::~DynamicConstructorDataGlobal(){}

template<bool use_ascii> void DynamicConstructorDataGlobal::write(std::ostream &os) const{
    if (use_ascii == mode_ascii){ os << std::scientific; os.precision(17); }
    auto tensor_refs =  makeReverseReferenceVector(tensors);

    IO::writeNumbers<use_ascii, IO::pad_line, int>(os, (int) tensor_refs.size());
    for(auto d : tensor_refs){
        IO::writeNumbers<use_ascii, IO::pad_rspace, double>(os, d->weight);
        IO::writeVector<use_ascii, IO::pad_line>(d->tensor, os);
    }
    writeNodeDataList<use_ascii>(data, os);
}

template<bool use_ascii> void DynamicConstructorDataGlobal::read(std::istream &is){
    int num_entries = IO::readNumber<use_ascii, int>(is);

    for(int i=0; i<num_entries; i++){
        tensors.emplace_front(TensorData{
                              std::vector<int>(num_dimensions), // tensor
                              MultiIndexSet(), // points, will be set later
                              std::vector<bool>(), // loaded, will be set later
                              IO::readNumber<use_ascii, double>(is) // weight
                              });
        IO::readVector<use_ascii>(is, tensors.front().tensor);
    }

    data = readNodeDataList<use_ascii>(num_dimensions, num_outputs, is);
}

template void DynamicConstructorDataGlobal::write<mode_ascii>(std::ostream &) const; // instantiate for faster build
template void DynamicConstructorDataGlobal::write<mode_binary>(std::ostream &) const;
template void DynamicConstructorDataGlobal::read<mode_ascii>(std::istream &);
template void DynamicConstructorDataGlobal::read<mode_binary>(std::istream &);

int DynamicConstructorDataGlobal::getMaxTensor() const{
    int max_tensor = 0;
    for(auto const &t : tensors)
        max_tensor = std::max(max_tensor, *std::max_element(t.tensor.begin(), t.tensor.end()));
    return max_tensor;
}

void DynamicConstructorDataGlobal::reloadPoints(std::function<int(int)> getNumPoints){
    for(auto &t : tensors){
        MultiIndexSet dummy_set(num_dimensions, std::vector<int>(t.tensor));
        t.points = MultiIndexManipulations::generateNestedPoints(1, dummy_set, getNumPoints);
        t.loaded = std::vector<bool>((size_t) t.points.getNumIndexes(), false);
    }

    for(auto const &p : data){ // rebuild the loaded vectors
        for(auto &t : tensors){
            int i = t.points.getSlot(p.point);
            if (i != -1) t.loaded[i] = true;
        }
    }
}

void DynamicConstructorDataGlobal::clearTesnors(){
    for(auto t = tensors.begin(), p = tensors.before_begin(); t != tensors.end(); t++){
        if (t->weight >= 0.0){
            tensors.erase_after(p);
            t = p;
        }else{
            p++;
        } // at each step, p starts before t, regardless if t is erased, p ends up equal to t, then t++
    }
}

MultiIndexSet DynamicConstructorDataGlobal::getInitialTensors() const{
    Data2D<int> tens(num_dimensions, 0);
    for(auto const &t : tensors){
        if (t.weight < 0.0)
            tens.appendStrip(t.tensor);
    }
    return MultiIndexSet(tens);
}

void DynamicConstructorDataGlobal::addTensor(const int *tensor, std::function<int(int)> getNumPoints, double weight){
    tensors.emplace_front(TensorData{
                          std::vector<int>(tensor, tensor + num_dimensions),
                          MultiIndexManipulations::generateNestedPoints(1, MultiIndexSet(num_dimensions, std::vector<int>(tensor, tensor + num_dimensions)), getNumPoints),
                          std::vector<bool>(),
                          weight
                          });

    tensors.front().loaded = std::vector<bool>((size_t) tensors.front().points.getNumIndexes(), false);
    for(auto const &p : data){
        int slot = tensors.front().points.getSlot(p.point);
        if (slot != -1) tensors.front().loaded[slot] = true;
    }
}

void DynamicConstructorDataGlobal::getNodesIndexes(std::vector<int> &inodes){
    inodes = std::vector<int>();
    auto get_weight = [](const TensorData &tensor)->double{
        if (tensor.weight <= 0.0) return tensor.weight;
        if (tensor.loaded.empty()) return 0.0; // should not be happening, should have ejected
        int num_loaded = (int) std::count_if(tensor.loaded.begin(), tensor.loaded.end(), [](bool p)->bool{ return p; });
        return tensor.weight * ((double(tensor.loaded.size()) - double(num_loaded)) / double(tensor.loaded.size()));
    };
    tensors.sort([&](const TensorData &a, const TensorData &b)->bool{ return (get_weight(a) < get_weight(b)); });
    for(auto const &t : tensors){
        if (!t.loaded.empty()){
            for(int i=0; i<t.points.getNumIndexes(); i++){
                if (!t.loaded[i])
                    inodes.insert(inodes.end(), t.points.getIndex(i), t.points.getIndex(i) + num_dimensions);
            }
        }
    }
}

bool DynamicConstructorDataGlobal::addNewNode(const std::vector<int> &point, const std::vector<double> &value){
    data.emplace_front(NodeData{point, value});
    for(auto &t : tensors){
        int slot = t.points.getSlot(point);
        if (slot != -1){
            t.loaded[slot] = true;
            if (std::all_of(t.loaded.begin(), t.loaded.end(), [](bool a)-> bool{ return a; })){ // if all entries are true
                t.loaded = std::vector<bool>();
                return true; // all points associated with this tensor have been loaded, signal to call ejectCompleteTensor()
            }else{
                return false; // point added to the tensor, but the tensor is not complete yet
            }
        }
    }
    return false; // could not find a tensor that contains this node???
}

void DynamicConstructorDataGlobal::ejectCompleteTensor(MultiIndexSet const &current_tensors, MultiIndexSet &new_tensors, MultiIndexSet &new_points, StorageSet &vals){
    new_points = MultiIndexSet(); // reset tensors
    vals = StorageSet();

    Data2D<int> candidate_tensors(num_dimensions, 0); // get a list of completed tensors
    for(auto &t : tensors)
        if (t.loaded.empty())
            candidate_tensors.appendStrip(t.tensor);

    // get the completed tensors that can be added to current_tensors while still preserving lower-completion
    new_tensors = MultiIndexManipulations::getLargestCompletion(current_tensors, MultiIndexSet(candidate_tensors));
    if (new_tensors.empty()) return;

    vals.resize((int) num_outputs, 0);
    auto p = tensors.before_begin(), t = tensors.begin();
    while(t != tensors.end()){
        if (new_tensors.missing(t->tensor)){ // if not using this tensor, advance to the next
            t++;
            p++;
        }else{ // if using the tensor
            int num_searching = t->points.getNumIndexes();
            Data2D<double> wvals(num_outputs, num_searching); // find all the values of the points
            int found = 0;

            auto d = data.before_begin();
            auto v = data.begin();
            while(found < num_searching){ // until all are found
                int slot = t->points.getSlot(v->point);
                if (slot == -1){ // this is not a point that we are looking for
                    d++;
                    v++;
                }else{ // point found, copy the value and extract it from the list
                    std::copy_n(v->value.begin(), num_outputs, wvals.getStrip(slot));
                    data.erase_after(d);
                    v = d;
                    v++;
                    found++;
                }
            }

            vals.addValues(new_points, t->points, wvals.getStrip(0));
            new_points.addMultiIndexSet(t->points);
            tensors.erase_after(p);
            t = p;
            t++;
        }
    }
}

}

#endif
