/*
 * Copyright (c) 2021, Miroslav Stoyanov & William Kong
 *
 * This file is part of
 * Toolkit for Adaptive Stochastic Modeling And Non-Intrusive ApproximatioN: TASMANIAN
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following
 * conditions are met:
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
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * UT-BATTELLE, LLC AND THE UNITED STATES GOVERNMENT MAKE NO REPRESENTATIONS AND DISCLAIM ALL WARRANTIES, BOTH EXPRESSED AND
 * IMPLIED. THERE ARE NO EXPRESS OR IMPLIED WARRANTIES OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, OR THAT THE USE
 * OF THE SOFTWARE WILL NOT INFRINGE ANY PATENT, COPYRIGHT, TRADEMARK, OR OTHER PROPRIETARY RIGHTS, OR THAT THE SOFTWARE WILL
 * ACCOMPLISH THE INTENDED RESULTS OR THAT THE SOFTWARE OR ITS USE WILL NOT RESULT IN INJURY OR DAMAGE. THE USER ASSUMES
 * RESPONSIBILITY FOR ALL LIABILITIES, PENALTIES, FINES, CLAIMS, CAUSES OF ACTION, AND COSTS AND EXPENSES, CAUSED BY, RESULTING
 * FROM OR ARISING OUT OF, IN WHOLE OR IN PART THE USE, STORAGE OR DISPOSAL OF THE SOFTWARE.
 */

#ifndef __TASMANIAN_ADDONS_CEXOTICQUADRATURE_CPP
#define __TASMANIAN_ADDONS_CEXOTICQUADRATURE_CPP

#include "TasmanianAddons.hpp"

using tsg_weight_fn = double(*)(double);
extern "C"{
    void tsgCreateExoticQuadratureFromGrid(void* custom_tabulated, const int level, const double shift, void* grid,
                                           const char* description, const bool is_symmetric) {
        *((TasGrid::CustomTabulated*) custom_tabulated) =
                TasGrid::getExoticQuadrature(level, shift, *reinterpret_cast<TasGrid::TasmanianSparseGrid*>(grid), description, is_symmetric);
    }
    void tsgCreateExoticQuadratureFromFunction(void* custom_tabulated, const int level, const double shift, tsg_weight_fn weight_fn,
                                               const int nref, const char* description, const bool is_symmetric) {
        *((TasGrid::CustomTabulated*) custom_tabulated) =
                TasGrid::getExoticQuadrature(level, shift, weight_fn, nref, description, is_symmetric);
    }
    void tsgCreateNestedExoticQuadratureFromFunction(
            void* custom_tabulated, const int level, const char *sRule,
            const double shift, tsg_weight_fn weight_fn, const int nref, const char* description) {
        TasGrid::TypeOneDRule rule = TasGrid::IO::getRuleString(sRule);
        *((TasGrid::CustomTabulated*) custom_tabulated) =
                TasGrid::getNestedExoticQuadrature(level, rule, shift, weight_fn, nref, description);
    }
} // extern "C"
#endif
