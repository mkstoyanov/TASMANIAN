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

#ifndef __TASMANIAN_DREAM_HPP
#define __TASMANIAN_DREAM_HPP

#include "TasmanianSparseGrid.hpp"

#include "tsgDreamEnumerates.hpp"
#include "tsgDreamState.hpp"
#include "tsgDreamCoreRandom.hpp"
#include "tsgDreamSample.hpp"
#include "tsgDreamSampleGrid.hpp"
#include "tsgDreamSamplePosterior.hpp"
#include "tsgDreamSamplePosteriorGrid.hpp"

//! \file TasmanianDREAM.hpp
//! \brief DiffeRential Evolution Adaptive Metropolis methods.
//! \author Miroslav Stoyanov
//! \ingroup TasmanianDREAM
//!
//! The main header required to gain access to the DREAM capabilities of Tasmanian.
//! The header will include all files needed by the DREAM module including
//! the TasmanianSparseGrid.hpp header.

//! \defgroup TasmanianDREAM DREAM: DiffeRential Evolution Adaptive Metropolis.
//!
//! \par DREAM
//! DiffeRential Evolution Adaptive Metropolis ...

//! \brief Encapsulates the Tasmanian DREAM module.
//! \ingroup TasmanianDREAM
namespace TasDREAM{}

// cleanup maros used by the headers above, no need to contaminate other codes
#undef __TASDREAM_CHECK_GRID_STATE_DIMS
#undef __TASDREAM_PDF_GRID_PRIOR
#undef __TASDREAM_PDF_POSTERIOR
#undef __TASDREAM_GRID_EXTRACT_RULE
#undef __TASDREAM_GRID_DOMAIN_GLLAMBDA
#undef __TASDREAM_GRID_DOMAIN_GHLAMBDA
#undef __TASDREAM_GRID_DOMAIN_DEFAULTS
#undef __TASDREAM_LIKELIHOOD_GRID_LIKE
#undef __TASDREAM_HYPERCUBE_DOMAIN
#undef __TASDREAM_CHECK_LOWERUPPER

#endif
