/////////////////////////////////////////////////////////////////////////////
// Authors: SangGi Do(sanggido@unist.ac.kr), Mingyu Woo(mwoo@eng.ucsd.edu)
//          (respective Ph.D. advisors: Seokhyeong Kang, Andrew B. Kahng)
//
//          Original parsing structure was made by Myung-Chul Kim (IBM).
//
// BSD 3-Clause License
//
// Copyright (c) 2018, SangGi Do and Mingyu Woo
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
///////////////////////////////////////////////////////////////////////////////

#include "circuit.h"
#include <time.h>
#include "mymeasure.h"

using opendp::cell;
using opendp::circuit;
using opendp::pixel;
using opendp::rect;
using opendp::row;

using std::cerr;
using std::cout;
using std::endl;

double lower_left_x, lower_left_y, upper_right_x, upper_right_y;
int cap_mode = 0;  // lxm:0表示默认电容，1表示仿真电容，don't use
double util;
std::vector< opendp::density_bin > bins;  // lxm:functionality bins

int main(int argc, char* argv[]) {
  cout << "===================================================================="
          "======="
       << endl;
  cout << "   CT-DP: Efficient Clock-Tree Guided Detailed Placement for Power "
          "Optimization < "
          "CT-DP_v2.0 >    "
       << endl;
  cout << "   Developers : Xinmiao Li, Hao Tang                               "
          "       "
       << endl;
  cout << "===================================================================="
          "======="
       << endl;

  CMeasure measure;
  circuit ckt;
  //  READ input files - parser.cpp
  ckt.wire_flag = 0;  // lxm:默认不开启纯线长优化模式
  measure.start_clock();
  ckt.read_files(argc, argv);
  measure.stop_clock("Parser");
  // ckt.init_lg_flag = ckt.init_check();
  // cout << "init_lg_flag: " << ckt.init_lg_flag << endl;

  // lxm:abacus
  // ckt.thread_num = 1;
  ckt.parallelAbacus();
  measure.stop_clock("Abacus");

  // // lxm:如果想对比纯线长优化，就将simple_placement换成wirelength_placement
  // if(ckt.wire_flag) {
  //   cout << " - - - - - < WIRELENGTH OPTIMIZATION > - - - - - " << endl;
  // ckt.wirelength_placement(measure);
  // }
  // else {
  //   cout << " - - - - - < CT-DP OPTIMIZATION > - - - - - " << endl;
  //   ckt.simple_placement(measure);
  // }
  measure.stop_clock("All");
  ckt.write_def(ckt.out_def_name);

  measure.print_clock();

  // EVALUATION - utility.cpp
  ckt.evaluation();

  // CHECK LEGAL - check_legal.cpp
  ckt.check_legality();
  delete[] ckt.grid;
  if(ckt.init_lg_flag) {
    delete[] ckt.grid_init;
  }
  cout << " - - - - - < Program END > - - - - - " << endl;
  return 0;
}
