///////////////////////////////////////////////////////////////////////////////
// Authors: Ilgweon Kang and Lutong Wang
//          (respective Ph.D. advisors: Chung-Kuan Cheng, Andrew B. Kahng),
//          based on Dr. Jingwei Lu with ePlace and ePlace-MS
//
//          Many subsequent improvements were made by Mingyu Woo
//          leading up to the initial release.
//
// BSD 3-Clause License
//
// Copyright (c) 2018, The Regents of the University of California
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

#ifndef __NS__
#define __NS__

#include "global.h"
#include "Router.h"
// #include "../circuit.h"

class MyNesterov {
 private:
  int temp_iter;
  int iter = 1;

 public:
  MyNesterov() {};
  Router ctsRouter;
  std::pair< double, double > root_coordi;
  std::map< int, int >
      ffToTreeNode_map;  // ff_cell的id与treenode的id的映射<ff_cell->id,treenode->id>
  double cluster_ll_x = std::numeric_limits< double >::max();
  double cluster_ll_y = std::numeric_limits< double >::max();
  double cluster_ur_x = std::numeric_limits< double >::lowest();
  double cluster_ur_y = std::numeric_limits< double >::lowest();
  void doCTS(std::vector< opendp::cell * > ff_list);
  void doCTS_cluster(std::vector< opendp::cell * > ff_list);
  std::pair< double, double > doCTS_root(std::vector< opendp::cell * > ff_list);
  double getTheCellToParWDist(opendp::cell *theCell);
  double getTheCellToParWDist(opendp::cell *theCell, double x, double y);
  pair< double, double > getGrandParentCoord(
      opendp::cell *theCell);  // lxm:用于获取爷节点坐标
  double _totalClkWL = 0;      // 用于迭代中止条件判定
  double sum_hpwl = 0;         // lxm:用于限制分区的最大损失线长
};

#endif
