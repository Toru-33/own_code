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

#include <error.h>
#include <sys/time.h>
#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <unistd.h>
#include <map>
#include <math.h>
// #include "bookShelfIO.h"
// #include "bin.h"
// #include "charge.h"
#include "global.h"
#include "../circuit.h"
// #include "mkl.h"
#include "ns.h"
#include "Router.h"
#include <string>
#include <cstdlib>

// static int backtrack_cnt = 0;
FPOS zeroFPoint;  // th
// extern map<string, bool> clkMap;
extern double tot_clkWL;  ///
extern int cur_iter_num;
extern double cur_ovfl;
// extern Router ctsRouter;
// Router ctsRouter;
map< string, int > sink_mp;  // 。
int total_iter_num = 0;
extern double lower_left_x, lower_left_y, upper_right_x,
    upper_right_y;  // 总的布局边界
// th
void time_start(double* time_cost) {
  struct timeval time_val;
  time_t time_secs;
  suseconds_t time_micro;
  gettimeofday(&time_val, NULL);
  time_micro = time_val.tv_usec;
  time_secs = time_val.tv_sec;
  *time_cost = (double)time_micro / 1000000 + time_secs;
  return;
}

void time_end(double* time_cost) {
  struct timeval time_val;
  time_t time_secs;
  suseconds_t time_micro;
  gettimeofday(&time_val, NULL);
  time_micro = time_val.tv_usec;
  time_secs = time_val.tv_sec;
  *time_cost = (double)time_micro / 1000000 + time_secs - *time_cost;
}
void MyNesterov::doCTS(std::vector< opendp::cell* > ff_list) {
  ctsRouter.sinks.clear();
  ctsRouter.sinks.reserve(ff_list.size() + 1);
  ffToTreeNode_map.clear();

  ctsRouter.vertexMS.clear();
  ctsRouter.vertexTRR.clear();
  ctsRouter.vertexDistE.clear();
  ctsRouter.vertexDistE_2.clear();
  ctsRouter.treeNodes.clear();
  ctsRouter.internal_num = 0;
  ctsRouter.gr_node_size = -1;
  ctsRouter.totalClkWL = 0;
  _totalClkWL = 0;

  // cout<<"before doCTS totalClkwl:"<<ctsRouter.totalClkWL<<endl;

  int sz = 2 * ff_list.size() - 1;
  if(ctsRouter.vertexMS.capacity() == 0) ctsRouter.vertexMS.resize(sz + 1);
  if(ctsRouter.vertexTRR.capacity() == 0) ctsRouter.vertexTRR.resize(sz + 1);
  if(ctsRouter.vertexDistE.capacity() == 0)
    ctsRouter.vertexDistE.resize(sz + 1);
  if(ctsRouter.vertexDistE_2.capacity() == 0)
    ctsRouter.vertexDistE_2.resize(sz + 1);
  if(ctsRouter.treeNodes.capacity() == 0) ctsRouter.treeNodes.resize(sz + 1);

  assert(ctsRouter.vertexMS.capacity() == sz + 1);
  assert(ctsRouter.vertexTRR.capacity() == sz + 1);
  assert(ctsRouter.vertexDistE.capacity() == sz + 1);
  assert(ctsRouter.vertexDistE_2.capacity() == sz + 1);
  assert(ctsRouter.treeNodes.capacity() == sz + 1);

  int num_id = 1;
  for(auto ff : ff_list) {
    double _x, _y;
    if(ff->isPlaced) {
      _x = ff->x_coord;
      _y = ff->y_coord;
    }
    else {
      _x = ff->init_x_coord;
      _y = ff->init_y_coord;
    }
    // cout <<"docts:"<<num_id<<":("<< _x <<","<< _y<<")" <<endl;
    ffToTreeNode_map[ff->id] = num_id;  // ff_cell的id与treenode的id的映射
    ctsRouter.sinks[num_id].id = num_id;
    ctsRouter.sinks[num_id].x = round(_x, 5);
    ctsRouter.sinks[num_id].y = round(_y, 5);
    ctsRouter.sinks[num_id].cap = 35;  // 35fF
    num_id++;
  }
  // cout<<ctsRouter.sinks.size()<<endl;
  ctsRouter.num_sinks = ff_list.size();

  ctsRouter.layout.clear();
  ctsRouter.layout.resize(4);

  // 这个layout应该是全局cell的左下右上坐标，但是gmin，gmax此处不可用，需要使用opendp上对应的变量

  ctsRouter.layout[0] = GridPoint(lower_left_x, lower_left_y);
  ctsRouter.layout[1] = GridPoint(lower_left_x, upper_right_y);
  ctsRouter.layout[2] = GridPoint(upper_right_x, upper_right_y);
  ctsRouter.layout[3] = GridPoint(upper_right_x, lower_left_y);

  // double time_cts = 0;

  // time_start(&time_cts);
  // cout << "dz!!!" << endl;
  ctsRouter.route();
  _totalClkWL = ctsRouter.totalClkWL;
  // cout<<"doCTS iter:"<<iter++<<endl;
  // time_end(&time_cts);
  // std::cout << "time_cts:" << time_cts << std::endl;
}

void MyNesterov::doCTS_cluster(std::vector< opendp::cell* > ff_list) {
  ctsRouter.sinks.clear();
  ctsRouter.sinks.reserve(ff_list.size() + 1);
  ffToTreeNode_map.clear();

  ctsRouter.vertexMS.clear();
  ctsRouter.vertexTRR.clear();
  ctsRouter.vertexDistE.clear();
  ctsRouter.vertexDistE_2.clear();
  ctsRouter.treeNodes.clear();
  ctsRouter.internal_num = 0;
  ctsRouter.gr_node_size = -1;
  ctsRouter.totalClkWL = 0;
  _totalClkWL = 0;
  sum_hpwl = 0;  // lxm:为了计算分区内的cell总线长，限制最大线长损失

  int sz = 2 * ff_list.size() - 1;
  if(ctsRouter.vertexMS.capacity() == 0) ctsRouter.vertexMS.resize(sz + 1);
  if(ctsRouter.vertexTRR.capacity() == 0) ctsRouter.vertexTRR.resize(sz + 1);
  if(ctsRouter.vertexDistE.capacity() == 0)
    ctsRouter.vertexDistE.resize(sz + 1);
  if(ctsRouter.vertexDistE_2.capacity() == 0)
    ctsRouter.vertexDistE_2.resize(sz + 1);
  if(ctsRouter.treeNodes.capacity() == 0) ctsRouter.treeNodes.resize(sz + 1);

  assert(ctsRouter.vertexMS.capacity() == sz + 1);
  assert(ctsRouter.vertexTRR.capacity() == sz + 1);
  assert(ctsRouter.vertexDistE.capacity() == sz + 1);
  assert(ctsRouter.vertexDistE_2.capacity() == sz + 1);
  assert(ctsRouter.treeNodes.capacity() == sz + 1);

  // bounding box
  for(const auto& cell : ff_list) {
    // lxm:原本是init
    // cluster_ll_x = std::min(cluster_ll_x, double(cell->init_x_coord));
    // cluster_ll_y = std::min(cluster_ll_y, double(cell->init_y_coord));
    // cluster_ur_x = std::max(cluster_ur_x, double(cell->init_x_coord));
    // cluster_ur_y = std::max(cluster_ur_y, double(cell->init_y_coord));

    cluster_ll_x = std::min(cluster_ll_x, double(cell->x_coord));
    cluster_ll_y = std::min(cluster_ll_y, double(cell->y_coord));
    cluster_ur_x = std::max(cluster_ur_x, double(cell->x_coord + cell->width));
    cluster_ur_y = std::max(cluster_ur_y, double(cell->y_coord + cell->height));
  }

  int num_id = 1;
  for(auto ff : ff_list) {
    double _x, _y;
    // if(ff->isPlaced) {
    _x = ff->x_coord;
    _y = ff->y_coord;
    // }
    // else {
    //   _x = ff->init_x_coord;
    //   _y = ff->init_y_coord;
    // }
    // sum_hpwl += ff->nets_hpwl;  // lxm
    // cout <<"docts:"<<num_id<<":("<< _x <<","<< _y<<")" <<endl;
    ffToTreeNode_map[ff->id] = num_id;  // ff_cell的id与treenode的id的映射
    ctsRouter.sinks[num_id].id = num_id;
    ctsRouter.sinks[num_id].x = round(_x, 5);
    ctsRouter.sinks[num_id].y = round(_y, 5);
    ctsRouter.sinks[num_id].cap = 35;  // 35fF
    num_id++;
  }
  // cout<<ctsRouter.sinks.size()<<endl;
  ctsRouter.num_sinks = ff_list.size();

  ctsRouter.layout.clear();
  ctsRouter.layout.resize(4);

  ctsRouter.layout[0] = GridPoint(cluster_ll_x, cluster_ll_y);
  ctsRouter.layout[1] = GridPoint(cluster_ll_x, cluster_ur_y);
  ctsRouter.layout[2] = GridPoint(cluster_ur_x, cluster_ur_y);
  ctsRouter.layout[3] = GridPoint(cluster_ur_x, cluster_ll_y);

  // ctsRouter.route();
  root_coordi = ctsRouter.RootDME();
  // cout<<"box:ll:("<<cluster_ll_x<<","<<cluster_ll_y<<")"<<"ur:("<<cluster_ur_x<<","<<cluster_ur_y<<")"<<endl;
  // cout<<"root:("<<root_coordi.first<<","<<root_coordi.second<<")"<<endl;

  _totalClkWL = ctsRouter.totalClkWL;
}

std::pair< double, double > MyNesterov::doCTS_root(
    std::vector< opendp::cell* > ff_list) {
  ctsRouter.sinks.clear();
  ctsRouter.sinks.reserve(ff_list.size() + 1);
  ffToTreeNode_map.clear();

  ctsRouter.vertexMS.clear();
  ctsRouter.vertexTRR.clear();
  ctsRouter.vertexDistE.clear();
  ctsRouter.vertexDistE_2.clear();
  ctsRouter.treeNodes.clear();
  ctsRouter.internal_num = 0;
  ctsRouter.gr_node_size = -1;
  ctsRouter.totalClkWL = 0;

  int sz = 2 * ff_list.size() - 1;
  if(ctsRouter.vertexMS.capacity() == 0) ctsRouter.vertexMS.resize(sz + 1);
  if(ctsRouter.vertexTRR.capacity() == 0) ctsRouter.vertexTRR.resize(sz + 1);
  if(ctsRouter.vertexDistE.capacity() == 0)
    ctsRouter.vertexDistE.resize(sz + 1);
  if(ctsRouter.vertexDistE_2.capacity() == 0)
    ctsRouter.vertexDistE_2.resize(sz + 1);
  if(ctsRouter.treeNodes.capacity() == 0) ctsRouter.treeNodes.resize(sz + 1);

  assert(ctsRouter.vertexMS.capacity() == sz + 1);
  assert(ctsRouter.vertexTRR.capacity() == sz + 1);
  assert(ctsRouter.vertexDistE.capacity() == sz + 1);
  assert(ctsRouter.vertexDistE_2.capacity() == sz + 1);
  assert(ctsRouter.treeNodes.capacity() == sz + 1);

  int num_id = 1;
  for(auto ff : ff_list) {
    double _x, _y;
    if(ff->isPlaced) {
      _x = ff->x_coord;
      _y = ff->y_coord;
    }
    else {
      _x = ff->init_x_coord;
      _y = ff->init_y_coord;
    }
    ffToTreeNode_map[ff->id] = num_id;  // ff_cell的id与treenode的id的映射
    ctsRouter.sinks[num_id].id = num_id;
    ctsRouter.sinks[num_id].x = round(_x, 5);
    ctsRouter.sinks[num_id].y = round(_y, 5);
    ctsRouter.sinks[num_id].cap = 35;
    num_id++;
  }
  ctsRouter.num_sinks = ff_list.size();

  ctsRouter.layout.clear();
  ctsRouter.layout.resize(4);

  // 这个layout应该是全局cell的左下右上坐标，但是gmin，gmax此处不可用，需要使用opendp上对应的变量

  ctsRouter.layout[0] = GridPoint(lower_left_x, lower_left_y);
  ctsRouter.layout[1] = GridPoint(lower_left_x, upper_right_y);
  ctsRouter.layout[2] = GridPoint(upper_right_x, upper_right_y);
  ctsRouter.layout[3] = GridPoint(upper_right_x, lower_left_y);

  double time_cts = 0;

  // time_start(&time_cts);
  std::pair< double, double > root_coord;
  root_coord = ctsRouter.RootDME();

  // time_end(&time_cts);
  return root_coord;
  // std::cout << "time_cts:" << time_cts << std::endl;
}

double MyNesterov::getTheCellToParWDist(opendp::cell* theCell) {
  int treeNode_id = ffToTreeNode_map.find(theCell->id)->second;
  // std::cout << "treenode_id:" << treeNode_id << std::endl;
  TreeNode theCell_self = ctsRouter.treeNodes[treeNode_id];
  TreeNode* theCell_par = ctsRouter.treeNodes[treeNode_id].par;
  TreeNode* theCell_par_par = ctsRouter.treeNodes[theCell_par->id].par;

  double disP1, disP2, beta = 0.5;
  disP1 = abs(theCell_self.x - theCell_par->x) +
          abs(theCell_self.y - theCell_par->y);
  disP2 = abs(theCell_self.x - theCell_par_par->x) +
          abs(theCell_self.y - theCell_par_par->y);

  return disP1 * beta + disP2 * (1 - beta);
}

pair< double, double > MyNesterov::getGrandParentCoord(opendp::cell* theCell) {
  int treeNode_id = ffToTreeNode_map.find(theCell->id)->second;
  // th
  //  cout << "cell_id:" << theCell->id << endl;
  //  for (auto it : ffToTreeNode_map) {
  //    cout << it.first << "->" << it.first << endl;
  //  }

  TreeNode* theCell_par = ctsRouter.treeNodes[treeNode_id].par;
  TreeNode* theCell_par_par = theCell_par ? theCell_par->par : nullptr;
  TreeNode* theCell_par_par_par =
      theCell_par_par ? theCell_par_par->par : nullptr;

  double x = theCell->x_coord;
  double y = theCell->y_coord;

  // 计算距离 disP1_x/y, disP2_x/y, disP3_x/y
  double disP1_x = abs(x - theCell_par->x);
  double disP1_y = abs(y - theCell_par->y);

  double disP2_x = theCell_par_par ? abs(x - theCell_par_par->x) : 0;
  double disP2_y = theCell_par_par ? abs(y - theCell_par_par->y) : 0;

  double disP3_x = theCell_par_par_par ? abs(x - theCell_par_par_par->x) : 0;
  double disP3_y = theCell_par_par_par ? abs(y - theCell_par_par_par->y) : 0;

  // 计算 sum_x 和 sum_y
  double sum_x = disP1_x + disP2_x + disP3_x;
  double sum_y = disP1_y + disP2_y + disP3_y;

  // 预防 sum_x 或 sum_y 为 0，避免除法错误
  if(sum_x == 0) sum_x = 1.0;  // 防止除数为0
  if(sum_y == 0) sum_y = 1.0;  // 防止除数为0

  // 计算 r_x 和 r_y
  double r_x =
      (disP1_x * disP1_x + disP2_x * disP2_x + disP3_x * disP3_x) / sum_x;
  double r_y =
      (disP1_y * disP1_y + disP2_y * disP2_y + disP3_y * disP3_y) / sum_y;

  return make_pair(r_x, r_y);

  // int treeNode_id = ffToTreeNode_map.find(theCell->id)->second;
  // TreeNode* theCell_par = ctsRouter.treeNodes[treeNode_id].par;
  // TreeNode* theCell_par_par = theCell_par ? theCell_par->par : nullptr;
  // TreeNode* theCell_par_par_par =
  //     theCell_par_par ? theCell_par_par->par : nullptr;

  // double x = theCell->x_coord;
  // double y = theCell->y_coord;

  // double disP_x = 0;
  // double disP_y = 0;

  // // 检查最远的祖先并计算距离
  // if(theCell_par_par_par) {
  //   disP_x = abs(x - theCell_par_par_par->x);
  //   disP_y = abs(y - theCell_par_par_par->y);
  // }
  // else if(theCell_par_par) {
  //   // 如果最远的祖先不存在，检查爷并计算距离
  //   disP_x = abs(x - theCell_par_par->x);
  //   disP_y = abs(y - theCell_par_par_par->y);
  // }
  // else if(theCell_par) {
  //   // 如果爷也不存在，使用父节点计算距离
  //   disP_x = abs(x - theCell_par->x);
  //   disP_y = abs(y - theCell_par->y);
  // }
  // return make_pair(disP_x, disP_y);
}

double MyNesterov::getTheCellToParWDist(opendp::cell* theCell, double x,
                                        double y) {
  int treeNode_id = ffToTreeNode_map.find(theCell->id)->second;
  TreeNode& theCell_self = ctsRouter.treeNodes[treeNode_id];
  TreeNode* theCell_par = theCell_self.par;
  TreeNode* theCell_par_par = nullptr;
  TreeNode* theCell_par_par_par = nullptr;

  // 提前检查父节点是否存在，减少多次访问
  if(theCell_par) {
    theCell_par_par = theCell_par->par;
    if(theCell_par_par) {
      theCell_par_par_par = theCell_par_par->par;
    }
  }

  // 提前计算并缓存 root 的坐标
  // double root_x = root_coordi.first;
  // double root_y = root_coordi.second;
  // double disP_root = abs(x - root_x) + abs(y - root_y);

  // 计算父节点的距离
  double disP1 =
      (theCell_par) ? abs(x - theCell_par->x) + abs(y - theCell_par->y) : 0;
  double disP2 = (theCell_par_par)
                     ? abs(x - theCell_par_par->x) + abs(y - theCell_par_par->y)
                     : 0;
  double disP3 = (theCell_par_par_par) ? abs(x - theCell_par_par_par->x) +
                                             abs(y - theCell_par_par_par->y)
                                       : 0;

  double eps = 1e-9;  // small constant to avoid division by zero
  double sum_dis = disP1 + disP2 + disP3;
  double inv_sum_dis = 1.0 / sum_dis;  // 预先计算 sum_dis 的倒数
  double distance =
      (disP1 * disP1 + disP2 * disP2 + disP3 * disP3) * inv_sum_dis;
  return distance;
}
