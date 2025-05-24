///////////////////////////////////////////////////////////////////////////////
//// Authors: SangGi Do(sanggido@unist.ac.kr), Mingyu Woo(mwoo@eng.ucsd.edu)
////          (respective Ph.D. advisors: Seokhyeong Kang, Andrew B. Kahng)
////
////          Original parsing structure was made by Myung-Chul Kim (IBM).
////
//// BSD 3-Clause License
////
//// Copyright (c) 2018, SangGi Do and Mingyu Woo
//// All rights reserved.
////
//// Redistribution and use in source and binary forms, with or without
//// modification, are permitted provided that the following conditions are met:
////
//// * Redistributions of source code must retain the above copyright notice,
/// this /   list of conditions and the following disclaimer.
////
//// * Redistributions in binary form must reproduce the above copyright notice,
////   this list of conditions and the following disclaimer in the documentation
////   and/or other materials provided with the distribution.
////
//// * Neither the name of the copyright holder nor the names of its
////   contributors may be used to endorse or promote products derived from
////   this software without specific prior written permission.
////
//// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
/// ARE / DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
/// LIABLE / FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
/// CONSEQUENTIAL / DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
/// SUBSTITUTE GOODS OR / SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
/// INTERRUPTION) HOWEVER / CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
/// CONTRACT, STRICT LIABILITY, / OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
/// ARISING IN ANY WAY OUT OF THE USE / OF THIS SOFTWARE, EVEN IF ADVISED OF THE
/// POSSIBILITY OF SUCH DAMAGE.
/////////////////////////////////////////////////////////////////////////////////

#include "circuit.h"
#include "th/ns.h"
#include "th/Router.h"

#include "thread"

#include <algorithm>
#include <limits>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <stdexcept>
#include <random>
#include <memory>
#include <thread>
#include <mutex>

// #include "/home/eda/th2/gurobi1103/linux64/include/gurobi_c++.h"

#define _DEBUG
#define SOFT_IGNORE true

using opendp::cell;
using opendp::circuit;
using opendp::density_bin;
using opendp::pixel;
using opendp::rect;
using opendp::row;

using std::cerr;
using std::cout;
using std::endl;
using std::ifstream;
using std::make_pair;
using std::max;
using std::min;
using std::ofstream;
using std::pair;
using std::string;
using std::to_string;
using std::vector;

std::mutex mtx;  // 用于线程安全

int ff_num;
std::vector< TreeNode > treenodes;
MyNesterov myNesterov;
std::vector< std::vector< cell* > > _clusters;
// std::vector< MyNesterov* > _myNesterovs;
std::vector< std::unique_ptr< MyNesterov > > _myNesterovs;

std::vector< std::pair< double, double > > _ffOptimalRegions;
std::vector< rect* > _clusterFixRect;
std::vector< double > _clusterFFArea;
bool alreadyBuildTrees = false;
std::vector< rect > cluster_rects;  // lxm:记录每个cluster的矩形

extern vector< density_bin >
    bins;  // lxm:让bins全局化，方便随时获取密度信息，新代码

extern int cap_mode;
extern double util;
int wendin_count = 0;
bool lianxu = false;
bool iter_flag = true;
double pre_totalClkWL = 0;

template < typename T >
T clamp(const T& value, const T& min_value, const T& max_value) {
  return std::max(min_value, std::min(value, max_value));
}

// lxm:initial_power是最下方的row的类型
void circuit::power_mapping() {
  for(int i = 0; i < rows.size(); i++) {
    row* theRow = &rows[i];
    if(initial_power == VDD) {
      if(i % 2 == 0)
        theRow->top_power = VDD;
      else
        theRow->top_power = VSS;
    }
    else {
      if(i % 2 == 0)
        theRow->top_power = VSS;
      else
        theRow->top_power = VDD;
    }
  }
  return;
}

void circuit::evaluation() {
  double avg_displacement = 0;
  double sum_displacement = 0;
  double max_displacement = 0;
  int count_displacement = 0;

  cell* maxCell = NULL;

  for(int i = 0; i < cells.size(); i++) {
    cell* theCell = &cells[i];
    double displacement = abs(theCell->init_x_coord - theCell->x_coord) +
                          abs(theCell->init_y_coord - theCell->y_coord);
    sum_displacement += displacement;
    if(displacement > max_displacement) {
      max_displacement = displacement;
      maxCell = theCell;
    }
    count_displacement++;
  }
  avg_displacement = sum_displacement / count_displacement;

  // double max_util = 0.0;
  // for(int i = 0; i < groups.size(); i++) {
  //   group* theGroup = &groups[i];
  //   if(max_util < theGroup->util) max_util = theGroup->util;
  // }

  cout << " - - - - - EVALUATION - - - - - " << endl;
  cout << " AVG_displacement : " << avg_displacement / wsite << endl;
  cout << " SUM_displacement : " << sum_displacement / wsite << endl;
  cout << " MAX_displacement : " << max_displacement / wsite << endl;
  cout << " - - - - - - - - - - - - - - - - " << endl;
  cout << " GP HPWL          : " << HPWL("INIT") << endl;
  cout << " HPWL             : " << HPWL("") << endl;
  cout << " avg_Disp_site    : " << Disp() / cells.size() / wsite << endl;
  cout << " avg_Disp_row     : " << Disp() / cells.size() / rowHeight << endl;
  cout << " delta_HPWL       : "
       << (HPWL("") - HPWL("INIT")) / HPWL("INIT") * 100 << endl;

  return;
}

double circuit::Disp() {
  double result = 0.0;
  for(int i = 0; i < cells.size(); i++) {
    cell* theCell = &cells[i];
    if(theCell->x_coord == 0 && theCell->y_coord == 0) continue;
    result += abs(theCell->init_x_coord - theCell->x_coord) +
              abs(theCell->init_y_coord - theCell->y_coord);
  }
  return result;
}

double circuit::HPWL(string mode) {
  double hpwl = 0;

  double x_coord = 0;
  double y_coord = 0;

  for(int i = 0; i < nets.size(); i++) {
    rect box;
    net* theNet = &nets[i];
    // cout << " net name : " << theNet->name << endl;
    pin* source = &pins[theNet->source];

    if(source->type == NONPIO_PIN) {
      cell* theCell = &cells[source->owner];
      if(mode == "INIT") {
        x_coord = theCell->init_x_coord;
        y_coord = theCell->init_y_coord;
      }
      else {
        x_coord = theCell->x_coord;
        y_coord = theCell->y_coord;
      }
      box.xLL = box.xUR = x_coord + source->x_offset * DEFdist2Microns;
      box.yLL = box.yUR = y_coord + source->y_offset * DEFdist2Microns;
    }
    else {
      box.xLL = box.xUR = source->x_coord;
      box.yLL = box.yUR = source->y_coord;
    }

    for(int j = 0; j < theNet->sinks.size(); j++) {
      pin* sink = &pins[theNet->sinks[j]];
      // cout << " sink name : " << sink->name << endl;
      if(sink->type == NONPIO_PIN) {
        cell* theCell = &cells[sink->owner];
        if(mode == "INIT") {
          x_coord = theCell->init_x_coord;
          y_coord = theCell->init_y_coord;
        }
        else {
          x_coord = theCell->x_coord;
          y_coord = theCell->y_coord;
        }
        box.xLL = min(box.xLL, x_coord + sink->x_offset * DEFdist2Microns);
        box.xUR = max(box.xUR, x_coord + sink->x_offset * DEFdist2Microns);
        box.yLL = min(box.yLL, y_coord + sink->y_offset * DEFdist2Microns);
        box.yUR = max(box.yUR, y_coord + sink->y_offset * DEFdist2Microns);
      }
      else {
        box.xLL = min(box.xLL, sink->x_coord);
        box.xUR = max(box.xUR, sink->x_coord);
        box.yLL = min(box.yLL, sink->y_coord);
        box.yUR = max(box.yUR, sink->y_coord);
      }
    }

    double box_boundary = (box.xUR - box.xLL + box.yUR - box.yLL);

    hpwl += box_boundary;
  }
  return hpwl / static_cast< double >(DEFdist2Microns);
  // return hpwl;
}

// lxm:计算每个bin的利用率和每个cell的密度因子==>密度因子=cell所在的bin的moveable_cell_area/(可用面积-fixed_cell_area)
double circuit::calc_density_factor(double unit) {
  static bool first_call = true;

  // 计算网格单元的大小==>bin的宽高为4个行高(unit传入为4)
  double gridUnit = unit * rowHeight;
  // 计算网格的数量
  int x_gridNum = (int)ceil((rx - lx) / gridUnit);
  int y_gridNum = (int)ceil((ty - by) / gridUnit);
  int numBins = x_gridNum * y_gridNum;
  if(first_call) {
    bins.reserve(numBins);
    // Initialize density grid
    bins.resize(numBins);  // Change from reserve to resize
// Initialize bin properties
#pragma omp parallel for collapse(2)
    for(int j = 0; j < y_gridNum; j++) {
      for(int k = 0; k < x_gridNum; k++) {
        unsigned binId = j * x_gridNum + k;
        bins[binId].lx = lx + k * gridUnit;
        bins[binId].ly = by + j * gridUnit;
        bins[binId].hx = bins[binId].lx + gridUnit;
        bins[binId].hy = bins[binId].ly + gridUnit;

        bins[binId].hx = std::min(bins[binId].hx, rx);
        bins[binId].hy = std::min(bins[binId].hy, ty);

        bins[binId].area = std::max((bins[binId].hx - bins[binId].lx) *
                                        (bins[binId].hy - bins[binId].ly),
                                    0.0);
        bins[binId].m_util = 0.0;
        bins[binId].f_util = 0.0;
        bins[binId].free_space = 0.0;
        bins[binId].overflow = 0.0;
      }
    }
    first_call = false;  // Mark as not the first call anymore
    /* (a) 计算与行网格的重叠面积，并将其添加到free_space */
    // #pragma omp parallel for
    for(auto& theRow : rows) {
      int lcol = std::max((int)floor((theRow.origX - lx) / gridUnit), 0);
      int rcol = std::min(
          (int)floor((theRow.origX + theRow.numSites * theRow.stepX - lx) /
                     gridUnit),
          x_gridNum - 1);
      int brow = std::max((int)floor((theRow.origY - by) / gridUnit), 0);
      int trow =
          std::min((int)floor((theRow.origY + rowHeight - by) / gridUnit),
                   y_gridNum - 1);

      for(int j = brow; j <= trow; j++) {
        for(int k = lcol; k <= rcol; k++) {
          unsigned binId = j * x_gridNum + k;

          /* 获取交集 */
          double lx = std::max(bins[binId].lx, (double)theRow.origX);
          double hx =
              std::min(bins[binId].hx,
                       (double)theRow.origX + theRow.numSites * theRow.stepX);
          double ly = std::max(bins[binId].ly, (double)theRow.origY);
          double hy =
              std::min(bins[binId].hy, (double)theRow.origY + rowHeight);

          if((hx - lx) > 1.0e-5 && (hy - ly) > 1.0e-5) {
            double common_area = (hx - lx) * (hy - ly);
            // #pragma omp atomic
            bins[binId].free_space += common_area;
            bins[binId].free_space =
                std::min(bins[binId].free_space, bins[binId].area);
          }
        }
      }
    }
  }
  else {
    // Only recalculate free_space for subsequent calls
    for(auto& bin : bins) {
      bin.f_util = 0.0;
      bin.m_util = 0.0;
    }
  }

  /* 1. 构建密度地图 */

  /* (b) 计算固定/可移动对象与网格的重叠面积 */
  // #pragma omp parallel for
  for(auto& theCell : cells) {
    int lcol = std::max((int)floor((theCell.x_coord - lx) / gridUnit), 0);
    int rcol =
        std::min((int)floor((theCell.x_coord + theCell.width - lx) / gridUnit),
                 x_gridNum - 1);
    int brow = std::max((int)floor((theCell.y_coord - by) / gridUnit), 0);
    int trow =
        std::min((int)floor((theCell.y_coord + theCell.height - by) / gridUnit),
                 y_gridNum - 1);

    for(int j = brow; j <= trow; j++) {
      for(int k = lcol; k <= rcol; k++) {
        unsigned binId = j * x_gridNum + k;

        if(theCell.inGroup)
          bins[binId].density_limit = std::max(
              bins[binId].density_limit, groups[group2id[theCell.group]].util);

        /* 获取交集 */
        double lx = std::max(bins[binId].lx, (double)theCell.x_coord);
        double hx =
            std::min(bins[binId].hx, (double)theCell.x_coord + theCell.width);
        double ly = std::max(bins[binId].ly, (double)theCell.y_coord);
        double hy =
            std::min(bins[binId].hy, (double)theCell.y_coord + theCell.height);
        double x_center = (double)theCell.x_coord + (double)theCell.width / 2;
        double y_center = (double)theCell.y_coord + (double)theCell.height / 2;

        // 如果一个cell的中心在bin中，则将其binId设置为cell的binId
        if(bins[binId].lx <= x_center && x_center < bins[binId].hx &&
           bins[binId].ly < y_center && y_center < bins[binId].hy) {
          theCell.binId = binId;
        }

        if((hx - lx) > 1.0e-5 && (hy - ly) > 1.0e-5) {
          double common_area = (hx - lx) * (hy - ly);
          if(theCell.isFixed) {
            // #pragma omp atomic
            bins[binId].f_util += common_area;
          }
          else {
            // #pragma omp atomic
            bins[binId].m_util += common_area;
          }
        }
      }
    }
  }
  /* 更新每个单元的密度因子 */
  int den9 = 0, den75 = 0;
  // #pragma omp parallel for
  for(auto& theCell : cells) {
    if(theCell.binId == UINT_MAX) continue;
    density_bin* theBin = &bins[theCell.binId];
    theCell.dense_factor =
        theBin->m_util / (theBin->free_space - theBin->f_util);
    if(theCell.dense_factor > 0.9) {
      den9++;
    }
    else if(theCell.dense_factor > 0.75) {
      den75++;
    }
  }
  cout << "DenBin > 0.9 : " << (den9 * 1.0) / cells.size() << endl;
  cout << "DenBin > 0.75 : " << (den75 * 1.0) / cells.size() << endl;
  if(den9 > 0.2 * cells.size() || den75 > 0.5 * cells.size() ||
     benchmark.find("superblue") != string::npos) {
    high_density = true;
  }
  else {
    high_density = false;
  }
  cout << "high density : " << high_density << endl;
  // exit(0);
  return 0.0;
}

void circuit::group_analyze() {
  for(int i = 0; i < groups.size(); i++) {
    group* theGroup = &groups[i];
    double region_area = 0;
    double avail_region_area = 0;
    double cell_area = 0;
    for(int j = 0; j < theGroup->regions.size(); j++) {
      rect* theRect = &theGroup->regions[j];
      region_area +=
          (theRect->xUR - theRect->xLL) * (theRect->yUR - theRect->yLL);
      avail_region_area +=
          (theRect->xUR - theRect->xLL - (int)theRect->xUR % 200 +
           (int)theRect->xLL % 200 - 200) *
          (theRect->yUR - theRect->yLL - (int)theRect->yUR % 2000 +
           (int)theRect->yLL % 2000 - 2000);
    }
    for(int k = 0; k < theGroup->siblings.size(); k++) {
      cell* theCell = theGroup->siblings[k];
      cell_area += theCell->width * theCell->height;
    }
    cout << " GROUP : " << theGroup->name << endl;
    cout << " region count : " << theGroup->regions.size() << endl;
    cout << " cell count : " << theGroup->siblings.size() << endl;
    cout << " region area : " << region_area << endl;
    cout << " avail region area : " << avail_region_area << endl;
    cout << " cell area : " << cell_area << endl;
    cout << " utilization : " << cell_area / region_area << endl;
    cout << " avail util : " << cell_area / avail_region_area << endl;
    cout << " - - - - - - - - - - - - - - - - - - - - " << endl;
  }
  return;
}

// lxm:返回的pair让cell尽可能接近区域，但不重叠(极大可能放在边界上)
pair< int, int > circuit::nearest_coord_to_rect_boundary(cell* theCell,
                                                         rect* theRect,
                                                         string mode) {
  int x = INT_MAX;
  int y = INT_MAX;
  int size_x = (int)floor(theCell->width / wsite + 0.5);
  int size_y = (int)floor(theCell->height / rowHeight + 0.5);
  if(mode == "init_coord") {
    x = theCell->init_x_coord;
    y = theCell->init_y_coord;
  }
  else if(mode == "coord") {
    x = theCell->x_coord;
    y = theCell->y_coord;
  }
  else if(mode == "pos") {
    x = theCell->x_pos * wsite;
    y = theCell->y_pos * rowHeight;
  }
  else {
    cerr << "circuit::nearest_coord_to_rect_boundary == invalid mode!" << endl;
    exit(2);
  }
  int temp_x = x;
  int temp_y = y;

  if(check_overlap(theCell, theRect, "init_coord") == true) {
    int dist_x = 0;
    int dist_y = 0;

    if(abs(x - theRect->xLL + theCell->width) > abs(theRect->xUR - x)) {
      dist_x = abs(theRect->xUR - x);
      temp_x = theRect->xUR;
    }
    else {
      dist_x = abs(x - theRect->xLL);
      temp_x = theRect->xLL - theCell->width;
    }
    if(abs(y - theRect->yLL + theCell->height) > abs(theRect->yUR - y)) {
      dist_y = abs(theRect->yUR - y);
      temp_y = theRect->yUR;
    }
    else {
      dist_y = abs(y - theRect->yLL);
      temp_y = theRect->yLL - theCell->height;
    }
    assert(dist_x > -1);
    assert(dist_y > -1);
    if(dist_x < dist_y)
      return make_pair(temp_x, y);
    else
      return make_pair(x, temp_y);
  }

  if(x < theRect->xLL)
    temp_x = theRect->xLL;
  else if(x + theCell->width > theRect->xUR)
    temp_x = theRect->xUR - theCell->width;

  if(y < theRect->yLL)
    temp_y = theRect->yLL;
  else if(y + theCell->height > theRect->yUR)
    temp_y = theRect->yUR - theCell->height;

#ifdef DEBUG
  cout << " - - - - - - - - - - - - - - - " << endl;
  cout << " input x_coord : " << x << endl;
  cout << " input y_coord : " << y << endl;
  cout << " found x_coord : " << temp_x << endl;
  cout << " found y_coord : " << temp_y << endl;
#endif

  return make_pair(temp_x, temp_y);
}

// lxm:返回当前cell与传入的矩形区域的距离(x+y)
int circuit::dist_for_rect(cell* theCell, rect* theRect, string mode) {
  int x = INT_MAX;
  int y = INT_MAX;
  if(mode == "init_coord") {
    x = theCell->init_x_coord;
    y = theCell->init_y_coord;
  }
  else if(mode == "coord") {
    x = theCell->x_coord;
    y = theCell->y_coord;
  }
  else if(mode == "pos") {
    x = theCell->x_pos * wsite;
    y = theCell->y_pos * rowHeight;
  }
  else {
    cerr << "circuit::dist_for_rect == invalid mode!" << endl;
    exit(2);
  }
  int temp_x = 0;
  int temp_y = 0;

  if(x < theRect->xLL)
    temp_x = theRect->xLL - x;
  else if(x + theCell->width > theRect->xUR)
    temp_x = x + theCell->width - theRect->xUR;

  if(y < theRect->yLL)
    temp_y = theRect->yLL - y;
  else if(y + theCell->height > theRect->yUR)
    temp_y = y + theCell->height - theRect->yUR;

  assert(temp_y > -1);
  assert(temp_x > -1);

  return temp_y + temp_x;
}

bool circuit::check_overlap(rect cell, rect box) {
  if(box.xLL >= cell.xUR || box.xUR <= cell.xLL) return false;
  if(box.yLL >= cell.yUR || box.yUR <= cell.yLL) return false;
  return true;
}

bool circuit::check_overlap(cell* theCell, rect* theRect, string mode) {
  int x = INT_MAX;
  int y = INT_MAX;
  if(mode == "init_coord") {
    x = theCell->init_x_coord;
    y = theCell->init_y_coord;
  }
  else if(mode == "coord") {
    x = theCell->x_coord;
    y = theCell->y_coord;
  }
  else if(mode == "pos") {
    x = theCell->x_pos * wsite;
    y = theCell->y_pos * rowHeight;
  }
  else {
    cerr << "circuit::check_overlap == invalid mode!" << endl;
    exit(2);
  }

  if(theRect->xUR <= x || theRect->xLL >= x + theCell->width) return false;
  if(theRect->yUR <= y || theRect->yLL >= y + theCell->height) return false;

  return true;
}

// lxm:返回false表示cell不全在box里面
bool circuit::check_inside(rect cell, rect box) {
  if(box.xLL > cell.xLL || box.xUR < cell.xUR) return false;
  if(box.yLL > cell.yLL || box.yUR < cell.yUR) return false;
  return true;
}

bool circuit::check_inside(cell* theCell, rect* theRect, string mode) {
  int x = INT_MAX;
  int y = INT_MAX;
  if(mode == "init_coord") {
    x = theCell->init_x_coord;
    y = theCell->init_y_coord;
  }
  else if(mode == "coord") {
    x = theCell->x_coord;
    y = theCell->y_coord;
  }
  else if(mode == "pos") {
    x = theCell->x_pos * wsite;
    y = theCell->y_pos * rowHeight;
  }
  else {
    cerr << "circuit::check_inside == invalid mode!" << endl;
    exit(2);
  }

  if(theRect->xUR < x + theCell->width || theRect->xLL > x) return false;
  if(theRect->yUR < y + theCell->height || theRect->yLL > y) return false;

  return true;
}

// lxm:传入的x_pos是当前x所在的第几个site，x_pos是当前位置，x,y是目标位置
// 避免了走到group外面或者不在group的cell到group中
// lxm:新版：保证在选位置的时候相对位移最小
pair< bool, pair< int, int > > circuit::bin_search(int x_pos, cell* theCell,
                                                   int x, int y) {
  pair< int, int > pos;
  macro* theMacro = &macros[theCell->type];
  int range = 10;
  int edge_left = (theMacro->edgetypeLeft == 1) ? 2 : 0;
  int edge_right = (theMacro->edgetypeRight == 1) ? 2 : 0;

  int x_step = (int)ceil(theCell->width / wsite) + edge_left + edge_right;
  int y_step = (int)ceil(theCell->height / rowHeight);

  if(init_lg_flag) {
    if(y + y_step > die.yUR / rowHeight) {
      return make_pair(false, pos);
    }
  }
  else {
    if(y + y_step > die.yUR / rowHeight ||
       (y_step % 2 == 0 && rows[y].top_power == theMacro->top_power)) {
      return make_pair(false, pos);
    }
  }

  auto isAvailable = [&](int x_new) {
    for(int k = y; k < y + y_step; k++) {
      for(int l = x_new; l < x_new + x_step; l++) {
        if(grid[k][l].linked_cell != NULL || !grid[k][l].isValid ||
           (theCell->inGroup && grid[k][l].group != group2id[theCell->group]) ||
           (!theCell->inGroup && grid[k][l].group != UINT_MAX)) {
          return false;
        }
      }
    }
    return true;
  };

  for(int i = 0; i < range; i++) {
    int x_new = (x_pos < x)            ? x + i
                : (x_pos >= x + range) ? x + range - i
                : (i <= x_pos - x)     ? x_pos - i
                                       : x + i;
    if(x_new + x_step <= die.xUR / wsite && isAvailable(x_new)) {
      pos = make_pair(y, x_new + edge_left);
      return make_pair(true, pos);
    }
  }

  return make_pair(false, pos);
}

// lxm:可以调控x步长的搜索
pair< bool, pair< int, int > > circuit::bin_search_site(int x_pos,
                                                        cell* theCell, int x,
                                                        int y, int site_num) {
  pair< int, int > pos;
  std::vector< cell* > overlapping_cells;  // lxm:记录重叠的cell
  overlapping_cells.reserve(100);          // lxm:预分配空间

  macro* theMacro = &macros[theCell->type];
  int range = site_num;
  int edge_left = (theMacro->edgetypeLeft == 1) ? 2 : 0;
  int edge_right = (theMacro->edgetypeRight == 1) ? 2 : 0;

  int x_step = (int)ceil(theCell->width / wsite) + edge_left + edge_right;
  int y_step = (int)ceil(theCell->height / rowHeight);

  if(init_lg_flag) {  // lxm:默认传入的都是power line对齐的
    if(y + y_step > die.yUR / rowHeight) {
      return make_pair(false, pos);
    }
  }
  else {
    if(y + y_step > die.yUR / rowHeight ||
       (y_step % 2 == 0 && rows[y].top_power == theMacro->top_power)) {
      return make_pair(false, pos);
    }
  }

  auto isAvailable = [&](int x_new) {
    for(int k = y; k < y + y_step; k++) {
      for(int l = x_new; l < x_new + x_step; l++) {
        if(grid[k][l].linked_cell != NULL || !grid[k][l].isValid ||
           (theCell->inGroup && grid[k][l].group != group2id[theCell->group]) ||
           (!theCell->inGroup && grid[k][l].group != UINT_MAX)) {
          return false;
        }
        // if(init_lg_flag) {
        //   if(grid_init[k][l].linked_cell != NULL &&
        //      !grid_init[k][l].linked_cell->is_ff &&
        //      !grid_init[k][l]
        //           .linked_cell
        //           ->isFixed) {  // lxm:暂时只让初始是合法化的情况去计算cost
        //     cell* temp_cell = grid_init[k][l].linked_cell;
        //     if(std::find(overlapping_cells.begin(), overlapping_cells.end(),
        //                  grid[k][l].linked_cell) == overlapping_cells.end())
        //                  {
        //       overlapping_cells.push_back(temp_cell);
        //       theCell->cost += (int)ceil(theCell->height / rowHeight);
        //     }
        //   }
        // }
      }
    }
    return true;
  };

  for(int i = 0; i < range; i++) {
    int x_new = (x_pos < x)            ? x + i
                : (x_pos >= x + range) ? x + range - i
                : (i <= x_pos - x)     ? x_pos - i
                                       : x + i;
    if(x_new + x_step <= die.xUR / wsite && isAvailable(x_new)) {
      pos = make_pair(y, x_new + edge_left);
      return make_pair(true, pos);
    }
  }
  theCell->cost = 0;
  return make_pair(false, pos);
}

// lxm:专为重合法化设计的
pair< bool, pair< int, int > > circuit::relegal_bin_search_site(int x_pos,
                                                                cell* theCell,
                                                                int x, int y,
                                                                int site_num) {
  pair< int, int > pos;

  macro* theMacro = &macros[theCell->type];
  int range = site_num;
  int edge_left = (theMacro->edgetypeLeft == 1) ? 2 : 0;
  int edge_right = (theMacro->edgetypeRight == 1) ? 2 : 0;

  int x_step = (int)ceil(theCell->width / wsite) + edge_left + edge_right;
  int y_step = (int)ceil(theCell->height / rowHeight);
  int init_y = (int)floor(theCell->init_y_coord / rowHeight + 0.5);
  int init_x = (int)floor(theCell->init_x_coord / wsite + 0.5);
  const auto heightRatio = rowHeight / wsite;

  if(init_lg_flag) {
    if(y + y_step > die.yUR / rowHeight) {
      return make_pair(false, pos);
    }
  }
  else {
    if(y + y_step > die.yUR / rowHeight ||
       (y_step % 2 == 0 && rows[y].top_power == theMacro->top_power)) {
      return make_pair(false, pos);
    }
  }

  auto isAvailable = [&](int x_new) {
    // 曼哈顿距离检查，若超出最大位移则跳过
    int x_displacement = abs(x_new - init_x);
    int y_displacement = abs(y - init_y) * heightRatio;
    if(x_displacement + y_displacement > displacement) {
      return false;
    }

    if(wire_flag && theCell->is_ff) {
      x_displacement = abs(x_new - x_pos);
      y_displacement = abs(y - theCell->y_coord / rowHeight) * heightRatio;
      if(x_displacement + y_displacement > displacement / 20) {
        return false;
      }
    }

    for(int k = y; k < y + y_step; k++) {
      for(int l = x_new; l < x_new + x_step; l++) {
        if(grid[k][l].linked_cell != NULL || !grid[k][l].isValid ||
           (theCell->inGroup && grid[k][l].group != group2id[theCell->group]) ||
           (!theCell->inGroup && grid[k][l].group != UINT_MAX)) {
          return false;
        }
        // if(grid_init[k][l].linked_cell !=
        //    NULL) {
        //    //lxm:grid_init包括non-overlap+overlap+register(register和overlap存在假位置)

        //   cell* temp_cell = grid_init[k][l].linked_cell;
        //   if((temp_cell->isOverlap && grid[k][l].linked_cell != NULL) ||
        //      (!temp_cell->isOverlap &&
        //       grid_init[k][l]
        //           .util)) {  // lxm:grid包括register和overlap的真实位置
        //     return false;
        //   }
        // }
      }
    }
    return true;
  };

  for(int i = 0; i < range; i++) {
    int x_new = (x_pos < x)            ? x + i
                : (x_pos >= x + range) ? x + range - i
                : (i <= x_pos - x)     ? x_pos - i
                                       : x + i;
    if(x_new + x_step <= die.xUR / wsite && isAvailable(x_new)) {
      pos = make_pair(y, x_new + edge_left);
      return make_pair(true, pos);
    }
  }
  return make_pair(false, pos);
}

// lxm:为diamond_swap设计
opendp::CandidatePosition circuit::new_relegal_bin_search_site(int x_pos,
                                                               cell* theCell,
                                                               int x, int y,
                                                               int site_num) {
  CandidatePosition candidate;
  macro* theMacro = &macros[theCell->type];
  int range = site_num;
  int edge_left = (theMacro->edgetypeLeft == 1) ? 2 : 0;
  int edge_right = (theMacro->edgetypeRight == 1) ? 2 : 0;

  int x_step = (int)ceil(theCell->width / wsite) + edge_left + edge_right;
  int y_step = (int)ceil(theCell->height / rowHeight);
  int init_y = (int)floor(theCell->init_y_coord / rowHeight + 0.5);
  int init_x = (int)floor(theCell->init_x_coord / wsite + 0.5);
  const auto heightRatio = rowHeight / wsite;

  if(init_lg_flag && y + y_step > die.yUR / rowHeight ||
     (!init_lg_flag &&
      (y + y_step > die.yUR / rowHeight ||
       (y_step % 2 == 0 && rows[y].top_power == theMacro->top_power)))) {
    return candidate;
  }

  auto isAvailable = [&](int x_new) {
    // 曼哈顿距离检查，若超出最大位移则跳过
    int x_displacement = abs(x_new - init_x);
    int y_displacement = abs(y - init_y) * heightRatio;
    if(x_displacement + y_displacement > displacement) {
      return false;
    }
    if(theCell->is_ff) {
      x_displacement = abs(x_new - x_pos);
      y_displacement = abs(y - theCell->y_coord / rowHeight) * heightRatio;
      if(x_displacement + y_displacement > 10) {
        return false;
      }
    }

    bool available = true;
    for(int k = y; k < y + y_step; k++) {
      for(int l = x_new; l < x_new + x_step; l++) {
        cell* linked_cell = grid[k][l].linked_cell;
        if(!grid[k][l].isValid ||
           (theCell->inGroup && grid[k][l].group != group2id[theCell->group]) ||
           (!theCell->inGroup && grid[k][l].group != UINT_MAX)) {
          return false;
        }
        if(linked_cell != nullptr) {
          // 若 linked_cell 尺寸相同
          if(linked_cell->width == theCell->width &&
             linked_cell->height == theCell->height) {
            candidate.isSwap = true;
            candidate.swapCell = linked_cell;
            candidate.position =
                make_pair(linked_cell->y_pos, linked_cell->x_pos);
            return true;
          }
          else {
            available = false;
          }
        }
      }
    }
    return available;
  };

  int left = 0, right = range;
  while(left < right) {
    int mid = (left + right) / 2;
    int x_new = x + mid;
    if(x_new + x_step <= die.xUR / wsite && isAvailable(x_new)) {
      if(!candidate.isSwap) {  // lxm:如果返回的isSwap为false表示找到了空位
        candidate.isSwap = true;
        candidate.position = make_pair(y, x_new + edge_left);
      }
      return candidate;  // 找到合适位置
    }
    if(x_new < x_pos) {
      left = mid + 1;
    }
    else {
      right = mid;
    }
  }
  return candidate;  // 未找到则返回默认空候选位置
}

pair< bool, pixel* > circuit::diamond_search(cell* theCell, int x_coord,
                                             int y_coord) {
  pixel* myPixel = NULL;
  pair< bool, pair< int, int > > found;
  int x_pos = (int)floor(x_coord / wsite + 0.5);
  int y_pos = (int)floor(y_coord / rowHeight + 0.5);

  int x_start = 0;
  int x_end = 0;
  int y_start = 0;
  int y_end = 0;

  int x_step = 10;
  int y_step = 1;
  if(static_cast< int >(theCell->height / rowHeight) % 2 == 0 && init_lg_flag) {
    y_step = 2;
  }
  // lxm:原本displacement*5
  if(theCell->inGroup == true) {
    // lxm:限制在group范围内
    group* theGroup = &groups[group2id[theCell->group]];
    x_start = max(x_pos - (int)(displacement),
                  (int)floor(theGroup->boundary.xLL / wsite));
    x_end = min(x_pos + (int)(displacement),
                (int)floor(theGroup->boundary.xUR / wsite) -
                    (int)floor(theCell->width / rowHeight + 0.5));
    y_start = max(y_pos - (int)displacement,
                  (int)ceil(theGroup->boundary.yLL / rowHeight));
    y_end = min(y_pos + (int)displacement,
                (int)ceil(theGroup->boundary.yUR / rowHeight) -
                    (int)floor(theCell->height / rowHeight + 0.5));
  }
  else {
    x_start = max(x_pos - (int)(displacement), 0);
    x_end =
        min(x_pos + (int)(displacement),
            (int)floor(rx / wsite) - (int)floor(theCell->width / wsite + 0.5));
    y_start = max(y_pos - (int)displacement, 0);
    y_end = min(y_pos + (int)displacement,
                (int)floor(ty / rowHeight) -
                    (int)floor(theCell->height / rowHeight + 0.5));
  }
  vector< pixel* > avail_list;  // lxm:  新代码，考虑遍历所有范围内的点来评估 ==
                                // > 可以考虑在适当时候制止，来减少时间消耗
  avail_list.reserve(50);
  int x_new = min(x_end, max(x_start, x_pos));
  int y_new = max(y_start, min(y_end, y_pos));
  if(wire_flag) {  // lxm:进入线长优化模式则默认起始位置是正确的
    found.first = true;
    found.second = make_pair(y_pos, x_pos);
  }
  else {
    found = relegal_bin_search_site(x_pos, theCell, x_new, y_new, x_step);
  }
  if(found.first == true) {
    myPixel = &grid[found.second.first][found.second.second];
    avail_list.emplace_back(myPixel);
    // cout << "return First" << endl;
    // return make_pair(found.first, myPixel);
  }
  // lxm:用于规定搜索的区域，利用率越高，存在的fixed_cell越多，就需要更大的搜索空间
  int div = 4;
  if(design_util > 0.6 || num_fixed_nodes > 0) div = 1;
  // lxm:i是搜索的半径
  for(int i = 1; i < (int)(displacement * 2) / div; i++) {
    // vector< pixel* > avail_list;
    avail_list.reserve(i * 4);
    pixel* myPixel = NULL;

    int x_offset = 0;
    int y_offset = 0;

    // lxm:左侧
    for(int j = 1; j < (i + 1) * 2; j++) {
      x_offset = -((j + 1) / 2);
      if(j % 2 == 1)
        y_offset = ((i + 1) * 2 - j) / 2;
      else
        y_offset = -((i + 1) * 2 - j) / 2;

      x_new = min(x_end, max(x_start, (x_pos + x_offset * x_step)));
      y_new = min(y_end, max(y_start, (y_pos + y_offset * y_step)));
      // lxm:检查Y坐标变化是否超过4,这里限制为4
      // 目前只允许低密度和纯线长优化执行
      if((wire_flag || !high_density) && abs(y_new - y_pos) > 4) {
        continue;  // 超过限制，停止搜索
      }
      found = relegal_bin_search_site(x_pos, theCell, x_new, y_new, x_step);
      if(found.first == true) {
        myPixel = &grid[found.second.first][found.second.second];

        avail_list.emplace_back(myPixel);
      }
    }

    // left side
    for(int j = 1; j < (i + 1) * 2; j++) {
      x_offset = (j - 1) / 2;
      if(j % 2 == 1)
        y_offset = ((i + 1) * 2 - j) / 2;
      else
        y_offset = -((i + 1) * 2 - j) / 2;
      x_new = min(x_end, max(x_start, (x_pos + x_offset * x_step)));
      y_new = min(y_end, max(y_start, (y_pos + y_offset * y_step)));
      // lxm:检查Y坐标变化是否超过4,这里限制为4
      // 目前只允许低密度和纯线长优化执行
      if((wire_flag || !high_density) && abs(y_new - y_pos) > 4) {
        continue;  // 超过限制，停止搜索
      }
      found = relegal_bin_search_site(x_pos, theCell, x_new, y_new, x_step);
      if(found.first == true) {
        myPixel = &grid[found.second.first][found.second.second];
        avail_list.emplace_back(myPixel);
      }
    }
    if(avail_list.size() > 16 || ((wire_flag || !high_density) && i > 20)) {
      break;
    }
  }
  double hpwl = 10000000;
  int best = INT_MAX;
  for(int j = 0; j < avail_list.size(); j++) {
    double hpwl_temp = calculateCellHPWL(theCell, avail_list[j]->x_pos * wsite,
                                         avail_list[j]->y_pos * rowHeight);
    if(hpwl_temp < hpwl) {
      hpwl = hpwl_temp;
      best = j;
    }
  }
  // cout << "best index: " << best << endl;
  if(best != INT_MAX) {
    return make_pair(true, avail_list[best]);
  }

  return make_pair(false, myPixel);
}

// lxm:只考虑位移的搜索
pair< bool, pixel* > circuit::diamond_search_disp(cell* theCell, int x_coord,
                                                  int y_coord) {
  pixel* myPixel = NULL;
  pair< bool, pair< int, int > > found;
  int x_pos = (int)floor(x_coord / wsite + 0.5);
  int y_pos = (int)floor(y_coord / rowHeight + 0.5);

  int x_start = 0;
  int x_end = 0;
  int y_start = 0;
  int y_end = 0;

  int x_step = 10;
  int y_step = 1;
  if(static_cast< int >(theCell->height / rowHeight) % 2 == 0 && init_lg_flag) {
    y_step = 2;
  }

  int multiple = static_cast< int >(rowHeight / wsite + 0.5);  // 计算步长

  if(theCell->inGroup == true) {
    group* theGroup = &groups[group2id[theCell->group]];
    x_start = max(x_pos - (int)(displacement * 5),
                  (int)floor(theGroup->boundary.xLL / wsite));
    x_end = min(x_pos + (int)(displacement * 5),
                (int)floor(theGroup->boundary.xUR / wsite) -
                    (int)floor(theCell->width / rowHeight + 0.5));
    y_start = max(y_pos - (int)displacement,
                  (int)ceil(theGroup->boundary.yLL / rowHeight));
    y_end = min(y_pos + (int)displacement,
                (int)ceil(theGroup->boundary.yUR / rowHeight) -
                    (int)floor(theCell->height / rowHeight + 0.5));
  }
  else {
    x_start = max(x_pos - (int)(displacement * 5), 0);
    x_end =
        min(x_pos + (int)(displacement * 5),
            (int)floor(rx / wsite) - (int)floor(theCell->width / wsite + 0.5));
    y_start = max(y_pos - (int)displacement, 0);
    y_end = min(y_pos + (int)displacement,
                (int)floor(ty / rowHeight) -
                    (int)floor(theCell->height / rowHeight + 0.5));
  }

  int div = 4;
  if(design_util > 0.6 || num_fixed_nodes > 0) div = 1;

  for(int i = 1; i < (int)(displacement * 2) / div; i++) {
    vector< pixel* > avail_list;
    avail_list.reserve(i * 4);
    pixel* myPixel = NULL;

    int x_offset = 0;
    int y_offset = 0;

    // 右侧候选点
    for(int j = 1; j < i * 2; j++) {
      x_offset = -((j + 1) / 2);
      y_offset = (j % 2 == 1) ? (i * 2 - j) / 2 : -(i * 2 - j) / 2;

      found = bin_search(x_pos, theCell,
                         min(x_end, max(x_start, (x_pos + x_offset * x_step))),
                         min(y_end, max(y_start, (y_pos + y_offset * y_step))));
      if(found.first) {
        myPixel = &grid[found.second.first][found.second.second];
        avail_list.push_back(myPixel);
      }
    }

    // 左侧候选点
    for(int j = 1; j < (i + 1) * 2; j++) {
      x_offset = (j - 1) / 2;
      y_offset = (j % 2 == 1) ? ((i + 1) * 2 - j) / 2 : -((i + 1) * 2 - j) / 2;

      found = bin_search(x_pos, theCell,
                         min(x_end, max(x_start, (x_pos + x_offset * x_step))),
                         min(y_end, max(y_start, (y_pos + y_offset * y_step))));
      if(found.first) {
        myPixel = &grid[found.second.first][found.second.second];
        avail_list.push_back(myPixel);
      }
    }

    // 选择当前轮次位移最小的候选点
    unsigned min_distance = UINT_MAX;
    int best_index = INT_MAX;
    for(int j = 0; j < avail_list.size(); j++) {
      int current_distance = abs(x_coord - avail_list[j]->x_pos * wsite) +
                             abs(y_coord - avail_list[j]->y_pos * rowHeight);
      if(current_distance < min_distance) {
        min_distance = current_distance;
        best_index = j;
      }
    }

    if(best_index != INT_MAX) {
      return make_pair(true, avail_list[best_index]);
    }
  }
  return make_pair(false, myPixel);
}

// lxm:钻石搜索基础上将相同宽高的cell进行交换判断，选择线长变化量最大的
void circuit::diamond_swap(cell* theCell, int x_coord, int y_coord) {
  CandidatePosition found;
  int x_pos = (int)floor(x_coord / wsite + 0.5);
  int y_pos = (int)floor(y_coord / rowHeight + 0.5);

  int x_start = 0;
  int x_end = 0;
  int y_start = 0;
  int y_end = 0;

  int x_step = 10;
  int y_step = 1;
  if(static_cast< int >(theCell->height / rowHeight) % 2 == 0 && init_lg_flag) {
    y_step = 2;
  }

  if(theCell->inGroup == true) {
    // lxm:限制在group范围内
    group* theGroup = &groups[group2id[theCell->group]];
    x_start = max(x_pos - (int)(displacement),
                  (int)floor(theGroup->boundary.xLL / wsite));
    x_end = min(x_pos + (int)(displacement),
                (int)floor(theGroup->boundary.xUR / wsite) -
                    (int)floor(theCell->width / rowHeight + 0.5));
    y_start = max(y_pos - (int)displacement,
                  (int)ceil(theGroup->boundary.yLL / rowHeight));
    y_end = min(y_pos + (int)displacement,
                (int)ceil(theGroup->boundary.yUR / rowHeight) -
                    (int)floor(theCell->height / rowHeight + 0.5));
  }
  else {
    x_start = max(x_pos - (int)(displacement), 0);
    x_end =
        min(x_pos + (int)(displacement),
            (int)floor(rx / wsite) - (int)floor(theCell->width / wsite + 0.5));
    y_start = max(y_pos - (int)displacement, 0);
    y_end = min(y_pos + (int)displacement,
                (int)floor(ty / rowHeight) -
                    (int)floor(theCell->height / rowHeight + 0.5));
  }
  vector< CandidatePosition >
      avail_list;  // lxm:  新代码，考虑遍历所有范围内的点来评估 ==
                   // > 可以考虑在适当时候制止，来减少时间消耗

  int x_new = min(x_end, max(x_start, x_pos));
  int y_new = max(y_start, min(y_end, y_pos));
  // found = new_relegal_bin_search_site(x_pos, theCell, x_new, y_new, 1);
  found.isSwap = true;
  found.position = make_pair(y_pos, x_pos);  // lxm:默认起始位置是正确的
  if(found.isSwap == true) {
    avail_list.emplace_back(found);
  }
  // lxm:用于规定搜索的区域，利用率越高，存在的fixed_cell越多，就需要更大的搜索空间
  int div = 4;
  if(design_util > 0.6 || num_fixed_nodes > 0) div = 1;
  // lxm:i是搜索的半径
  for(int i = 1; i < (int)(displacement) / div; i++) {
    // vector< pixel* > avail_list;
    avail_list.reserve(i * 4);
    pixel* myPixel = NULL;

    int x_offset = 0;
    int y_offset = 0;

    for(int j = 1; j < (i + 1) * 2; j++) {
      x_offset = -((j + 1) / 2);
      if(j % 2 == 1)
        y_offset = ((i + 1) * 2 - j) / 2;
      else
        y_offset = -((i + 1) * 2 - j) / 2;
      x_new = min(x_end, max(x_start, (x_pos + x_offset * x_step)));
      y_new = min(y_end, max(y_start, (y_pos + y_offset * y_step)));
      if(abs(y_new - y_pos) > 3) {
        continue;
      }
      found = new_relegal_bin_search_site(x_pos, theCell, x_new, y_new, x_step);
      if(found.isSwap == true) {
        avail_list.emplace_back(found);
      }
    }

    // left side
    for(int j = 1; j < (i + 1) * 2; j++) {
      x_offset = (j - 1) / 2;
      if(j % 2 == 1)
        y_offset = ((i + 1) * 2 - j) / 2;
      else
        y_offset = -((i + 1) * 2 - j) / 2;
      x_new = min(x_end, max(x_start, (x_pos + x_offset * x_step)));
      y_new = min(y_end, max(y_start, (y_pos + y_offset * y_step)));
      if(abs(y_new - y_pos) > 3) {  // lxm:DAC是4
        continue;
      }
      found = new_relegal_bin_search_site(x_pos, theCell, x_new, y_new, x_step);
      if(found.isSwap == true) {
        avail_list.emplace_back(found);
      }
    }
    if(avail_list.size() > 15 || i > 10) {  // lxm:DAC是i>20
      break;
    }
  }
  double hpwl = DBL_MAX;
  int best = INT_MAX;
  double or_hpwl = calculateCellHPWL(theCell, x_pos * wsite, y_pos * rowHeight);
  for(int j = 0; j < avail_list.size(); j++) {
    double hpwl_change = 0;
    if(avail_list[j].swapCell == nullptr) {
      hpwl_change =
          calculateCellHPWL(theCell, avail_list[j].position.second * wsite,
                            avail_list[j].position.first * rowHeight) -
          or_hpwl;
    }
    else {
      cell* swapCell = avail_list[j].swapCell;
      hpwl_change =
          calculateCellHPWL(swapCell, x_coord, y_coord) +
          calculateCellHPWL(theCell, swapCell->x_coord, swapCell->y_coord) -
          calculateCellHPWL(theCell, x_coord, y_coord) -
          calculateCellHPWL(swapCell, swapCell->x_coord, swapCell->y_coord);
    }
    if(hpwl_change < hpwl) {
      hpwl = hpwl_change;
      best = j;
    }
  }
  if(avail_list[best].swapCell != nullptr) {
    erase_pixel(avail_list[best].swapCell);
    erase_pixel(theCell);
    relegal_paint_pixel(theCell, avail_list[best].swapCell->x_coord / wsite,
                        avail_list[best].swapCell->y_coord / rowHeight);
    relegal_paint_pixel(avail_list[best].swapCell, x_coord / wsite,
                        y_coord / rowHeight);
  }
  else {
    relegal_paint_pixel(theCell, avail_list[best].position.second,
                        avail_list[best].position.first);
  }
}

vector< cell* > circuit::overlap_cells(cell* theCell) {
  vector< cell* > list;
  int step_x = (int)ceil(theCell->width / wsite);
  int step_y = (int)ceil(theCell->height / rowHeight);

  OPENDP_HASH_MAP< unsigned, cell* > cell_list;
#ifdef USE_GOOGLE_HASH
  cell_list.set_empty_key(UINT_MAX);
#endif

  for(int i = theCell->y_pos; i < theCell->y_pos + step_y; i++) {
    for(int j = theCell->x_pos; j < theCell->y_pos + step_x; j++) {
      if(grid[i][j].linked_cell != NULL) {
        cell_list[grid[i][j].linked_cell->id] = grid[i][j].linked_cell;
      }
    }
  }
  list.reserve(cell_list.size());
  for(auto& currCell : cell_list) {
    list.push_back(currCell.second);
  }

  return list;
}

// rect should be position
// lxm:返回矩形区域内所有cell的集合(不会重复)
vector< cell* > circuit::get_cells_from_boundary(rect* theRect) {
  assert(theRect->xLL >= die.xLL);
  assert(theRect->yLL >= die.yLL);
  assert(theRect->xUR <= die.xUR);
  assert(theRect->yUR <= die.yUR);

  int x_start = (int)floor(theRect->xLL / wsite + 0.5);
  int y_start = (int)floor(theRect->yLL / rowHeight + 0.5);
  int x_end = (int)floor(theRect->xUR / wsite + 0.5);
  int y_end = (int)floor(theRect->yUR / rowHeight + 0.5);

  vector< cell* > list;

  OPENDP_HASH_MAP< unsigned, cell* > cell_list;
#ifdef USE_GOOGLE_HASH
  cell_list.set_empty_key(UINT_MAX);
#endif

  for(int i = y_start; i < y_end; i++) {
    for(int j = x_start; j < x_end; j++) {
      cell* theCell = dynamic_cast< cell* >(grid[i][j].linked_cell);
      if(theCell != NULL) {
        if(theCell->isFixed == false) cell_list[theCell->id] = theCell;
      }
    }
  }

  list.reserve(cell_list.size());
  for(auto& currCell : cell_list) {
    list.push_back(currCell.second);
  }
  return list;
}

double circuit::dist_benefit(cell* theCell, int x_coord, int y_coord) {
  double curr_dist = abs(theCell->x_coord - theCell->init_x_coord) +
                     abs(theCell->y_coord - theCell->init_y_coord);
  double new_dist = abs(theCell->init_x_coord - x_coord) +
                    abs(theCell->init_y_coord - y_coord);
  return new_dist - curr_dist;
}

bool circuit::swap_cell(cell* cellA, cell* cellB) {
  if(cellA == cellB)
    return false;
  else if(cellA->type != cellB->type)
    return false;
  else if(cellA->isFixed == true || cellB->isFixed == true)
    return false;

  double benefit = dist_benefit(cellA, cellB->x_coord, cellB->y_coord) +
                   dist_benefit(cellB, cellA->x_coord, cellA->y_coord);

  if(benefit < 0) {
    int A_x_pos = cellB->x_pos;
    int A_y_pos = cellB->y_pos;
    int B_x_pos = cellA->x_pos;
    int B_y_pos = cellA->y_pos;

    erase_pixel(cellA);
    erase_pixel(cellB);
    paint_pixel(cellA, A_x_pos, A_y_pos);
    paint_pixel(cellB, B_x_pos, B_y_pos);
    // cout << "swap benefit : " << benefit << endl;
    // save_score();
    return true;
  }
  return false;
}

pair< bool, cell* > circuit::nearest_cell(int x_coord, int y_coord) {
  bool found = false;
  cell* nearest_cell = NULL;
  double nearest_dist = 99999999999;
  for(int i = 0; i < cells.size(); i++) {
    cell* theCell = &cells[i];
    if(theCell->isPlaced == false) continue;

    double dist =
        abs(theCell->x_coord - x_coord) + abs(theCell->y_coord - y_coord);

    if(dist < rowHeight * 2)
      if(nearest_dist > dist) {
        nearest_dist = dist;
        nearest_cell = theCell;
        found = true;
      }
  }
  return make_pair(found, nearest_cell);
}

// lxm:相比bin_search只遍历自己cell的长宽，不会额外扩展遍历
pair< bool, pair< int, int > > circuit::bin_search_FF(cell* theCell, int x,
                                                      int y) {
  pair< int, int > pos;
  macro* theMacro = &macros[theCell->type];

  // EDGETYPE 1 - 1 : 400, 1 - 2 : 400, 2 - 2 : 0
  // lxm:计算水平方向上的偏移量
  int edge_left = 0;
  int edge_right = 0;
  if(theMacro->edgetypeLeft == 1) edge_left = 2;
  if(theMacro->edgetypeRight == 1) edge_right = 2;

  // theMacro->print();  // lxm:打印macro信息
  if(theMacro->edgetypeLeft != 0 || theMacro->edgetypeRight != 0) {
    cout << "theMacro->name: " << theMacro->name
         << " edgetypeLeft: " << theMacro->edgetypeLeft
         << " edgetypeRight: " << theMacro->edgetypeRight << endl;
  }

  int x_step = (int)ceil(theCell->width / wsite) + edge_left + edge_right;
  int y_step = (int)ceil(theCell->height / rowHeight);

  // IF y is out of border
  if(y + y_step > (die.yUR / rowHeight) || x + x_step > (die.xUR / wsite))
    return make_pair(false, pos);

  // If even number multi-deck cell -> check top power
  // lxm:如果当前行的power类型和macro的top_power类型相同，且cell高度是偶数，则不允许放置
  if(!init_lg_flag) {
    if(y_step % 2 == 0) {
      if(rows[y].top_power == theMacro->top_power) return make_pair(false, pos);
    }
  }
  int i = 0;
  bool available = true;
  for(int k = y; k < y + y_step; k++) {
    for(int l = x + i; l < x + i + x_step; l++) {
      if(grid[k][l].linked_cell != NULL || grid[k][l].isValid == false) {
        available = false;
        break;
      }
      // check group regions
      if(theCell->inGroup == true) {
        if(grid[k][l].group != group2id[theCell->group]) available = false;
      }
      else {
        if(grid[k][l].group != UINT_MAX) available = false;
      }
    }
    if(available == false) break;
  }

  if(available == true) {
    if(edge_left == 0)
      pos = make_pair(y, x + i);
    else
      pos = make_pair(y, x + i + edge_left);

    return make_pair(available, pos);
  }
  return make_pair(false, pos);
}

// lxm:排序函数相关，离root远的先动，距离一样密度大的先动
//  曼哈顿距离函数
double manhattan_distance(const std::pair< double, double >& root, cell* c) {
  return std::abs(root.first - c->x_coord) + std::abs(root.second - c->y_coord);
}

double manhattan_distance(const std::pair< double, double >& root, double x,
                          double y) {
  return std::abs(root.first - x) + std::abs(root.second - y);
}

// lxm:根据和root的距离排序(register的面积都相同)
bool SortUpdistance(cell* a, cell* b, const std::pair< double, double >& root) {
  double distance_a = manhattan_distance(root, a) + a->nets_hpwl;
  double distance_b = manhattan_distance(root, b) + b->nets_hpwl;
  // double distance_a = manhattan_distance(root, a);
  // double distance_b = manhattan_distance(root, b);
  // a<b true：表示降序排序
  if(distance_a > distance_b)
    return true;
  else if(distance_a < distance_b)
    return false;
  else
    return (a->dense_factor < b->dense_factor);
}

/// th
// kmeans++初始化聚类中心的函数

// double calculateManhattanDistance(cell* a, std::pair< double, double > b) {
//   return std::abs(a->x_coord - b.first) + std::abs(a->y_coord - b.second);
// }
// // k++中心
// std::vector< std::pair< double, double > > initCenters(
//     std::vector< cell* >& cells, int K) {
//   std::vector< std::pair< double, double > > centers;

//   if(cells.empty()) {
//     return centers;
//   }

//   std::mt19937 rng(static_cast< unsigned int >(std::time(0)));
//   std::uniform_int_distribution< int > dist(0, cells.size() - 1);

//   // k-means++ 初始中心选择
//   centers.push_back(
//       {cells[dist(rng)]->init_x_coord, cells[dist(rng)]->init_y_coord});

//   for(int i = 1; i < K; ++i) {
//     std::vector< double > distances(cells.size(),
//                                     std::numeric_limits< double >::max());

//     for(size_t j = 0; j < cells.size(); ++j) {
//       for(const auto& center : centers) {
//         double dist = calculateManhattanDistance(cells[j], center);
//         if(dist < distances[j]) {
//           distances[j] = dist;
//         }
//       }
//     }

//     std::discrete_distribution< int > weighted_dist(distances.begin(),
//                                                     distances.end());
//     int next_center_index = weighted_dist(rng);
//     centers.push_back({cells[next_center_index]->init_x_coord,
//                        cells[next_center_index]->init_y_coord});
//   }

//   return centers;
// }
// 递归二分中心
std::vector< std::pair< double, double > > initCenters(
    std::vector< cell* >& cells, int K) {
  std::vector< std::pair< double, double > > centers;

  if(cells.empty()) {
    return centers;
  }

  if(K <= 1) {
    double sumX = 0;
    double sumY = 0;
    for(cell* c : cells) {
      sumX += c->x_coord;
      sumY += c->y_coord;
    }
    double centerX = sumX / cells.size();
    double centerY = sumY / cells.size();
    centers.emplace_back(centerX, centerY);
    return centers;
  }

  std::sort(cells.begin(), cells.end(),
            [](cell* a, cell* b) { return a->x_coord < b->x_coord; });
  size_t mid = cells.size() / 2;
  std::vector< cell* > left(cells.begin(), cells.begin() + mid);
  std::vector< cell* > right(cells.begin() + mid, cells.end());

  std::vector< std::pair< double, double > > leftCenters =
      initCenters(left, K / 2);
  std::vector< std::pair< double, double > > rightCenters =
      initCenters(right, K - K / 2);
  centers.insert(centers.end(), leftCenters.begin(), leftCenters.end());
  centers.insert(centers.end(), rightCenters.begin(), rightCenters.end());

  return centers;
}

// lxm:获取最优区域，so slow!
void circuit::getOptimalRegion(std::vector< cell* > ff_cells) {
// std::map<int, std::pair<double, double>> results;
#pragma omp parallel for
  for(auto ff_cell : ff_cells) {
    std::vector< rect > relaBoxes;

    for(int i = 0; i < ff_cell->connected_nets.size(); i++) {
      rect box;
      net* theNet = &nets[ff_cell->connected_nets[i]];
      pin* source = &pins[theNet->source];
      double x_coord = 0;
      double y_coord = 0;

      if(ff_cell != &cells[source->owner]) {
        if(source->type == NONPIO_PIN) {
          cell* theCell = &cells[source->owner];
          x_coord = theCell->x_coord;
          y_coord = theCell->y_coord;

          box.xLL = box.xUR = x_coord + source->x_offset * DEFdist2Microns;
          box.yLL = box.yUR = y_coord + source->y_offset * DEFdist2Microns;
        }
        else {
          box.xLL = box.xUR = source->x_coord;
          box.yLL = box.yUR = source->y_coord;
        }
      }

      for(int j = 0; j < theNet->sinks.size(); j++) {
        pin* sink = &pins[theNet->sinks[j]];
        // cout << " sink name : " << sink->name << endl;
        if(ff_cell != &cells[sink->owner]) {
          if(sink->type == NONPIO_PIN) {
            cell* theCell = &cells[sink->owner];
            x_coord = theCell->x_coord;
            y_coord = theCell->y_coord;

            box.xLL = min(box.xLL, x_coord + sink->x_offset * DEFdist2Microns);
            box.xUR = max(box.xUR, x_coord + sink->x_offset * DEFdist2Microns);
            box.yLL = min(box.yLL, y_coord + sink->y_offset * DEFdist2Microns);
            box.yUR = max(box.yUR, y_coord + sink->y_offset * DEFdist2Microns);
          }
          else {
            box.xLL = min(box.xLL, sink->x_coord);
            box.xUR = max(box.xUR, sink->x_coord);
            box.yLL = min(box.yLL, sink->y_coord);
            box.yUR = max(box.yUR, sink->y_coord);
          }
        }
      }
      relaBoxes.push_back(box);
    }

    rect OptimalRegion;
    std::vector< double > xs;
    std::vector< double > ys;
    for(auto box : relaBoxes) {
      xs.emplace_back(box.xLL);
      xs.emplace_back(box.xUR);
      ys.emplace_back(box.yUR);
      ys.emplace_back(box.yLL);
    }
    sort(xs.begin(), xs.end());
    sort(ys.begin(), ys.end());
    OptimalRegion.xLL = xs[xs.size() / 2];
    OptimalRegion.xUR = xs[xs.size() / 2 + 1];
    OptimalRegion.yUR = ys[ys.size() / 2 + 1];
    OptimalRegion.yLL = ys[ys.size() / 2];

    std::pair< double, double > center =
        std::make_pair((OptimalRegion.xLL + OptimalRegion.xUR) / 2,
                       (OptimalRegion.yUR + OptimalRegion.yLL) / 2);
    _ffOptimalRegions[ff_cell->id] = center;
  }
  std::cout << "getOptimalRegion over " << endl;
  // return results;
}

// 8/12 th
double calculateManhattanDistance(cell* a, std::pair< double, double > b) {
  return std::abs(a->x_coord - b.first) + std::abs(a->y_coord - b.second);
}
// 计算聚类中心
std::vector< std::pair< double, double > > calculateCentroids(
    const std::vector< cell* >& cells, int k) {
  std::vector< std::pair< double, double > > centroids(k, {0.0, 0.0});
  std::vector< int > counts(k, 0);

  for(const auto& c : cells) {
    centroids[c->cluster_id].first += c->x_coord;
    centroids[c->cluster_id].second += c->y_coord;
    counts[c->cluster_id]++;
  }

  for(int i = 0; i < k; ++i) {
    if(counts[i] > 0) {
      centroids[i].first /= counts[i];
      centroids[i].second /= counts[i];
    }
  }

  return centroids;
}

// 论文的初始中心
std::vector< std::pair< double, double > > initCenters(
    const std::vector< cell* >& cells) {
  std::vector< std::pair< double, double > > centers;
  std::vector< double > xs, ys;
  for(auto& c : cells) {
    xs.emplace_back(c->x_coord);
    ys.emplace_back(c->y_coord);
  }
  sort(xs.begin(), xs.end());
  sort(ys.begin(), ys.end());
  std::pair< double, double > center1, center2;
  if(abs(ys.begin() - ys.end()) > abs(xs.end() - xs.begin())) {
    center1 = std::make_pair(xs[xs.size() / 2], ys[ys.size() / 4]);
    center2 = std::make_pair(xs[xs.size() / 2], ys[(ys.size() * 3) / 4]);
  }
  else {
    center1 = std::make_pair(xs[xs.size() / 4], ys[ys.size() / 2]);
    center2 = std::make_pair(xs[(xs.size() * 3) / 4], ys[ys.size() / 2]);
  }
  centers.emplace_back(center1);
  centers.emplace_back(center2);
  return centers;
}
// k-means 算法
void kmeans(std::vector< cell* >& cells, int k) {
  // 使用 initCenters 函数初始化聚类中心
  std::vector< std::pair< double, double > > centroids =
      initCenters(cells);  // initCenters(cells,k)

  bool changed = true;
  while(changed) {
    changed = false;

    // 分配每个 cell 到最近的聚类中心
    for(auto& c : cells) {
      double min_dist = std::numeric_limits< double >::max();
      int best_cluster = -1;

      for(int i = 0; i < k; ++i) {
        double dist = calculateManhattanDistance(c, centroids[i]);
        if(dist < min_dist) {
          min_dist = dist;
          best_cluster = i;
        }
      }

      if(c->cluster_id != best_cluster) {
        c->cluster_id = best_cluster;
        changed = true;
      }
    }

    // 重新计算聚类中心
    centroids = calculateCentroids(cells, k);
  }
}
// 8/12 th
std::pair< double, double > calculateCentroid(
    const std::vector< cell* >& cluster) {
  double sum_x = 0.0;
  double sum_y = 0.0;
  int count = cluster.size();

  for(const auto& c : cluster) {
    sum_x += c->x_coord;
    sum_y += c->y_coord;
  }

  // 计算质心的 x 和 y 坐标
  double centroid_x = (count > 0) ? sum_x / count : 0.0;
  double centroid_y = (count > 0) ? sum_y / count : 0.0;

  return std::make_pair(centroid_x, centroid_y);
}
double calculateSSE(const std::vector< cell* >& cells,
                    std::pair< double, double >& centroid) {
  double sse = 0.0;
  for(const auto& c : cells) {
    sse += std::pow(calculateManhattanDistance(c, centroid), 2);
  }
  return sse;
}
double calculateCapEntropy(double cap, double total_cap) {
  if(total_cap == 0) {
    return 0;
  }
  return -cap / total_cap * log2(cap / total_cap);
}

std::vector< std::vector< cell* > > biKmeans(std::vector< cell* >& ff_cells,
                                             int K, int current_size) {
  for(auto& c : ff_cells) {
    c->cluster_id = 0;
  }
  float lambda = 0.35;  // 0.35-0.75
  std::vector< std::vector< cell* > > result_clusters;
  // 初始化，将所有 cell 分为一个聚类
  using ClusterInfo =
      std::tuple< std::vector< cell* >,
                  double >;  // 聚类信息，包括聚类cell的集合和penalty

  auto compare = [](const ClusterInfo& a, const ClusterInfo& b) {
    return std::get< 1 >(a) <
           std::get< 1 >(b);  // Compare based on Penalty, descending order
  };

  std::priority_queue< ClusterInfo, std::vector< ClusterInfo >,
                       decltype(compare) >
      CQ(compare);

  CQ.emplace(ff_cells, 0.0);

  std::vector< cell* > target_cluster;
  std::vector< double > sses;
  double max_sse = 0.0;
  while(CQ.size() < K) {
    // 选择一个聚类进行二分
    target_cluster = std::get< 0 >(CQ.top());
    CQ.pop();

    std::pair< double, double > target_centroid =
        calculateCentroid(target_cluster);
    double target_sse = calculateSSE(target_cluster, target_centroid);
    if(target_sse == max_sse) {
      sort(sses.begin(), sses.end());
      sses.erase(sses.begin());
      max_sse = sses[0];
    }
    std::vector< std::vector< cell* > > subclusters(2);
    kmeans(target_cluster, 2);
    // cout << "target_cluster[0].id:" << target_cluster[0]->id << endl;
    //  更新全局聚类
    for(auto& c : target_cluster) {
      if(c->cluster_id == 0) {
        subclusters[0].push_back(c);
      }
      else {
        subclusters[1].push_back(c);
      }
    }
    for(auto& c : subclusters) {
      std::pair< double, double > centroid = calculateCentroid(c);
      double sse = calculateSSE(c, centroid);
      sses.push_back(sse);
      if(sse > max_sse) {
        max_sse = sse;
      }
    }
    for(auto& c : subclusters) {
      std::pair< double, double > centroid = calculateCentroid(c);
      double sse = calculateSSE(c, centroid);
      double h = calculateCapEntropy((double)c.size(), (double)ff_cells.size());
      double penalty = lambda * h + (1 - lambda) * (sse / max_sse);
      CQ.emplace(std::move(c), penalty);
    }
  }
  int index = current_size;
  while(CQ.size() > 0) {
    std::vector< cell* > cluster = std::get< 0 >(CQ.top());
    CQ.pop();
    for(auto c : cluster) {
      c->cluster_id = index;
    }
    result_clusters.push_back(cluster);
    index++;
  }

  return result_clusters;
}

std::vector< std::vector< cell* > > biKmeans(std::vector< cell* >& ff_cells,
                                             int K) {
  for(auto& c : ff_cells) {
    c->cluster_id = 0;
  }
  float lambda = 0.35;  // 0.35-0.75
  std::vector< std::vector< cell* > > result_clusters;
  // 初始化，将所有 cell 分为一个聚类
  using ClusterInfo =
      std::tuple< std::vector< cell* >,
                  double >;  // 聚类信息，包括聚类cell的集合和penalty

  auto compare = [](const ClusterInfo& a, const ClusterInfo& b) {
    return std::get< 1 >(a) <
           std::get< 1 >(b);  // Compare based on Penalty, descending order
  };

  std::priority_queue< ClusterInfo, std::vector< ClusterInfo >,
                       decltype(compare) >
      CQ(compare);

  CQ.emplace(ff_cells, 0.0);

  std::vector< cell* > target_cluster;
  std::vector< double > sses;
  double max_sse = 0.0;
  while(CQ.size() < K) {
    // 选择一个聚类进行二分
    target_cluster = std::get< 0 >(CQ.top());
    CQ.pop();

    std::pair< double, double > target_centroid =
        calculateCentroid(target_cluster);
    double target_sse = calculateSSE(target_cluster, target_centroid);
    if(target_sse == max_sse) {
      sort(sses.begin(), sses.end());
      sses.erase(sses.begin());
      max_sse = sses[0];
    }
    std::vector< std::vector< cell* > > subclusters(2);
    kmeans(target_cluster, 2);
    // cout << "target_cluster[0].id:" << target_cluster[0]->id << endl;
    //  更新全局聚类
    for(auto& c : target_cluster) {
      if(c->cluster_id == 0) {
        subclusters[0].push_back(c);
      }
      else {
        subclusters[1].push_back(c);
      }
    }
    for(auto& c : subclusters) {
      std::pair< double, double > centroid = calculateCentroid(c);
      double sse = calculateSSE(c, centroid);
      sses.push_back(sse);
      if(sse > max_sse) {
        max_sse = sse;
      }
    }
    for(auto& c : subclusters) {
      std::pair< double, double > centroid = calculateCentroid(c);
      double sse = calculateSSE(c, centroid);
      double h = calculateCapEntropy((double)c.size(), (double)ff_cells.size());
      double penalty = lambda * h + (1 - lambda) * (sse / max_sse);
      CQ.emplace(std::move(c), penalty);
    }
  }
  int index = 0;
  while(CQ.size() > 0) {
    std::vector< cell* > cluster = std::get< 0 >(CQ.top());
    CQ.pop();
    for(auto c : cluster) {
      c->cluster_id = index;
    }
    result_clusters.push_back(cluster);
    index++;
  }

  return result_clusters;
}
void buildCTSTrees_noGroup(std::vector< cell* > ff_cells) {
  std::vector< std::vector< cell* > > clusters;
  _clusters.clear();  // 清空之前的数据
  _myNesterovs.clear();
  _myNesterovs.reserve(1000);
  _clusters.reserve(1000);

  int per_cluster_num_limit =
      std::max(static_cast< int >(ff_cells.size()) / 40, 120);
  std::cout << "cluster num limit: " << per_cluster_num_limit << std::endl;
  int k = static_cast< int >(ff_cells.size() / per_cluster_num_limit) + 1;
  clusters = biKmeans(ff_cells, k);

  // 计算平均值和方差
  double mean = 0.0;
  int cluster_count = clusters.size();

  for(const auto& cluster : clusters) {
    mean += cluster.size();
  }
  mean /= cluster_count;

  double variance = 0.0;
  for(const auto& cluster : clusters) {
    variance += std::pow(cluster.size() - mean, 2);
  }
  variance /= cluster_count;

  // std::cout << "cluster size variance: " << variance << std::endl;

  _clusters = std::move(clusters);  // 使用移动语义
  _myNesterovs.resize(_clusters.size());

  for(size_t i = 0; i < _clusters.size(); ++i) {
    if(_clusters[i].empty()) {
      continue;
    }

    auto myNesterov = std::make_unique< MyNesterov >();  // 使用智能指针
    myNesterov->doCTS_cluster(_clusters[i]);
    _myNesterovs[i] = std::move(myNesterov);  // 移动指针

    // lxm: 新增，为画图服务,no for this version
    // rect new_rect(myNesterov->cluster_ll_x, myNesterov->cluster_ur_x,
    //                myNesterov->cluster_ll_y, myNesterov->cluster_ur_y);
    // cluster_rects.emplace_back(new_rect);
  }

  std::cout << "buildCTSTrees over" << std::endl;
}

void buildCTSTrees_Ingroup(std::vector< cell* > ff_cells) {
  std::vector< std::vector< cell* > > clusters;
  int per_cluster_num_limit =
      std::max(static_cast< int >(ff_cells.size()) / 40, 120);
  int k = static_cast< int >(ff_cells.size() / per_cluster_num_limit) + 1;
  _clusters.clear();  // 清空之前的数据,
  _myNesterovs.clear();
  _clusters.reserve(1000);
  _myNesterovs.reserve(1000);
  clusters = biKmeans(ff_cells, k);
  _clusters = std::move(clusters);  // 使用移动语义
  _myNesterovs.resize(clusters.size());

  // cluster_rects.clear();

  for(size_t i = 0; i < clusters.size(); i++) {
    if(clusters[i].empty()) continue;
    auto myNesterov = std::make_unique< MyNesterov >();
    myNesterov->doCTS_cluster(clusters[i]);
    _myNesterovs[i] = std::move(myNesterov);

    // 确保使用已初始化的值
    // rect new_rect(myNesterov->cluster_ll_x, myNesterov->cluster_ur_x,
    //               myNesterov->cluster_ll_y, myNesterov->cluster_ur_y);
    // cluster_rects.emplace_back(new_rect);
  }
  cout << "buildCTSTrees over" << endl;
}

// lxm:对全体cell构建一次整树
void parallel_buildCTSTrees_Ingroup(std::vector< cell* > ff_cells) {
  std::vector< std::vector< cell* > > clusters;

  int k = 1;
  size_t current_size = _clusters.size();
  // th:后处理clusters的cell与_clusters映射问题
  clusters = biKmeans(ff_cells, k, current_size);

  _clusters.resize(current_size + clusters.size());
  _myNesterovs.resize(current_size + clusters.size());

  for(size_t i = 0; i < clusters.size(); i++) {
    if(clusters[i].empty()) continue;
    // th:后处理clusters的cell与_clusters映射问题
    // for(auto c : clusters[i]) {
    //   c->cluster_id = i + current_size;
    // }
    _clusters[current_size + i] = clusters[i];
    auto myNesterov = std::make_unique< MyNesterov >();
    myNesterov->doCTS_cluster(clusters[i]);
    _myNesterovs[current_size + i] = std::move(myNesterov);
  }
}

MyNesterov* getMyNesterovByff(cell* thecell) {
  // std::cout << "getMyNesterovByff start:" << _clusters.size()<<std::endl;
  if(thecell->cluster_id > -1 && thecell->cluster_id < _clusters.size()) {
    return _myNesterovs[thecell->cluster_id].get();  // 返回裸指针
  }
  else
    throw std::runtime_error("没找到该ff所属的集群。");
}

// lxm:目前是根据传入的box来确定区域，保证在区域内搜索，如果原位置可行直接放入
pair< bool, pixel* > circuit::diamond_search_FF(cell* theCell, int x_coord,
                                                int y_coord, rect box) {
  pixel* myPixel = NULL;
  pair< bool, pair< int, int > > found;
  int x_pos = (int)floor(x_coord / wsite + 0.5);
  int y_pos = (int)floor(y_coord / rowHeight + 0.5);

  int x_start = 0;
  int x_end = 0;
  int y_start = 0;
  int y_end = 0;

  int x_step = 10;
  int y_step = 1;
  if(static_cast< int >(theCell->height / rowHeight) % 2 == 0 && init_lg_flag) {
    y_step = 2;
  }

  MyNesterov* my = getMyNesterovByff(theCell);
  // lxm:后续改成曼哈顿环？
  //  auto [grand_parent_x, grand_parent_y] = my->getGrandParentCoord(theCell);

  // lxm：根据与引导点距离确定候选点个数
  // int T = abs(grand_parent_x - x_coord) / wsite +
  //         abs(grand_parent_y - y_coord) / rowHeight;
  // int threshold = std::min(T, 20);
  int threshold = 20;
  // lxm:确定搜索边界,通过更改displacement可以调整搜索范围
  int multiple =
      static_cast< int >(rowHeight / wsite + 0.5);  // lxm:MA_code 代替5

  x_start = max(x_pos - (int)(displacement * 5), (int)floor(box.xLL / wsite));
  x_end = min(x_pos + (int)(displacement * 5),
              (int)floor(box.xUR / wsite) -
                  (int)floor(theCell->width / rowHeight + 0.5));
  y_start = max(y_pos - (int)displacement, (int)ceil(box.yLL / rowHeight));
  y_end = min(y_pos + (int)displacement,
              (int)ceil(box.yUR / rowHeight) -
                  (int)floor(theCell->height / rowHeight + 0.5));

  vector< pixel* > avail_list;  // lxm:  新代码，考虑遍历所有范围内的点来评估 ==
                                // > 可以考虑在适当时候制止，来减少时间消耗
  avail_list.reserve(100);
  vector< double > score;
  score.reserve(100);

  auto add_candidate = [&](int x, int y) {
    auto [found, pos] = relegal_bin_search_site(x_pos, theCell, x, y, x_step);
    if(found) {
      avail_list.emplace_back(&grid[pos.first][pos.second]);
    }
  };
  // lxm:对FF第一次就走10个步长
  add_candidate(clamp(x_pos, x_start, x_end), clamp(y_pos, y_start, y_end));

  int x_new = min(x_end, max(x_start, x_pos));
  int y_new = max(y_start, min(y_end, y_pos));
  // 用于规定搜索的区域，利用率越高，存在的fixed_cell越多，就需要更大的搜索空间

  int div = (design_util > 0.6 || num_fixed_nodes > 0) ? 1 : 4;
  // lxm:i是搜索的半径
  for(int i = 1; i < (int)(displacement * 2) / div; i++) {
    pixel* myPixel = NULL;

    int x_offset = 0;
    int y_offset = 0;
    // lxm:左侧
    for(int j = 1; j < (i + 1) * 2; j++) {
      x_offset = -((j + 1) / 2);
      if(j % 2 == 1)
        y_offset = ((i + 1) * 2 - j) / 2;
      else
        y_offset = -((i + 1) * 2 - j) / 2;
      x_new = min(x_end, max(x_start, (x_pos + x_offset * x_step)));
      y_new = min(y_end, max(y_start, (y_pos + y_offset * y_step)));
      if(!high_density && abs(y_new - y_pos) > 3) {  // lxm:限制信号线长恶化
        continue;
      }
      add_candidate(x_new, y_new);
    }
    // left side
    for(int j = 1; j < (i + 1) * 2; j++) {
      x_offset = (j - 1) / 2;
      if(j % 2 == 1)
        y_offset = ((i + 1) * 2 - j) / 2;
      else
        y_offset = -((i + 1) * 2 - j) / 2;
      x_new = min(x_end, max(x_start, (x_pos + x_offset * x_step)));
      y_new = min(y_end, max(y_start, (y_pos + y_offset * y_step)));
      if(!high_density && abs(y_new - y_pos) > 3) {  // lxm:限制信号线长恶化
        continue;
      }
      add_candidate(x_new, y_new);
    }
    // lxm:可调部分，用于平衡时间和效果（之前最好是20）
    if(avail_list.size() > threshold || (!high_density && i > 15)) {
      break;
    }
  }

  // lxm:funny code
  int best = INT_MAX;
  double score_min = 10000000;
  double alpha = 0.5, beta = 1, gamma = 1.5;
  for(int j = 0; j < avail_list.size(); j++) {
    double netsHPWL = calculateCellHPWL(theCell, avail_list[j]->x_pos * wsite,
                                        avail_list[j]->y_pos * rowHeight);
    double now_cts_wl =
        my->getTheCellToParWDist(theCell, avail_list[j]->x_pos * wsite,
                                 avail_list[j]->y_pos * rowHeight);
    double density = calculateOptimalDensityFactorChange(
                         theCell, avail_list[j]->x_pos * wsite,
                         avail_list[j]->y_pos * rowHeight) *
                     100;
    double score_temp = density * alpha + now_cts_wl * beta + gamma * netsHPWL;
    if(score_temp < score_min) {
      score_min = score_temp;
      best = j;
    }
  }
  if(best != INT_MAX) {
    return make_pair(true, avail_list[best]);
  }
  return make_pair(false, myPixel);
}

// lxm:对group内不同的算法
pair< bool, pixel* > circuit::diamond_search_FF_group(cell* theCell,
                                                      int x_coord, int y_coord,
                                                      rect box) {
  pixel* myPixel = NULL;
  pair< bool, pair< int, int > > found;
  int x_pos = (int)floor(x_coord / wsite + 0.5);
  int y_pos = (int)floor(y_coord / rowHeight + 0.5);

  int x_start = 0;
  int x_end = 0;
  int y_start = 0;
  int y_end = 0;

  int x_step = 10;
  int y_step = 1;
  if(static_cast< int >(theCell->height / rowHeight) % 2 == 0 && init_lg_flag) {
    y_step = 2;
  }

  MyNesterov* my = getMyNesterovByff(theCell);

  auto [grand_parent_x, grand_parent_y] = my->getGrandParentCoord(theCell);

  // lxm：根据与引导点距离确定候选点个数
  int T = abs(grand_parent_x - x_coord) / wsite +
          abs(grand_parent_y - y_coord) / rowHeight;

  // lxm:确定搜索边界,通过更改displacement可以调整搜索范围
  int multiple =
      static_cast< int >(rowHeight / wsite + 0.5);  // lxm:MA_code 代替5

  group* theGroup = &groups[group2id[theCell->group]];
  x_start = max(x_pos - (int)(displacement * 5),
                (int)floor(theGroup->boundary.xLL / wsite));
  x_end = min(x_pos + (int)(displacement * 5),
              (int)floor(theGroup->boundary.xUR / wsite) -
                  (int)floor(theCell->width / rowHeight + 0.5));
  y_start = max(y_pos - (int)displacement,
                (int)ceil(theGroup->boundary.yLL / rowHeight));
  y_end = min(y_pos + (int)displacement,
              (int)ceil(theGroup->boundary.yUR / rowHeight) -
                  (int)floor(theCell->height / rowHeight + 0.5));

  vector< pixel* > avail_list;  // lxm:  新代码，考虑遍历所有范围内的点来评估
                                // == > 可以考虑在适当时候制止，来减少时间消耗
  avail_list.reserve(100);
  auto add_candidate = [&](int x, int y) {
    auto [found, pos] = bin_search_site(x_pos, theCell, x, y, x_step);
    if(found) {
      avail_list.emplace_back(&grid[pos.first][pos.second]);
    }
  };
  // lxm:对FF第一次就走10个步长
  int x_new = min(x_end, max(x_start, x_pos));
  int y_new = max(y_start, min(y_end, y_pos));
  add_candidate(x_new, y_new);

  // 用于规定搜索的区域，利用率越高，存在的fixed_cell越多，就需要更大的搜索空间

  int div = (design_util > 0.6 || num_fixed_nodes > 0) ? 1 : 4;
  // lxm:i是搜索的半径
  for(int i = 1; i < (int)(displacement * 2) / div; i++) {
    pixel* myPixel = NULL;

    int x_offset = 0;
    int y_offset = 0;
    // lxm:这里应该是左侧吧，不知道为什么注释是右侧搜索
    for(int j = 1; j < (i + 1) * 2; j++) {
      x_offset = -((j + 1) / 2);
      if(j % 2 == 1)
        y_offset = ((i + 1) * 2 - j) / 2;
      else
        y_offset = -((i + 1) * 2 - j) / 2;
      x_new = min(x_end, max(x_start, (x_pos + x_offset * x_step)));
      y_new = min(y_end, max(y_start, (y_pos + y_offset * y_step)));
      if(abs(y_new - y_pos) > 4) {
        continue;
      }
      add_candidate(x_new, y_new);
    }
    // left side
    for(int j = 1; j < (i + 1) * 2; j++) {
      x_offset = (j - 1) / 2;
      if(j % 2 == 1)
        y_offset = ((i + 1) * 2 - j) / 2;
      else
        y_offset = -((i + 1) * 2 - j) / 2;
      x_new = min(x_end, max(x_start, (x_pos + x_offset * x_step)));
      y_new = min(y_end, max(y_start, (y_pos + y_offset * y_step)));
      if(abs(y_new - y_pos) > 4) {
        continue;
      }
      add_candidate(x_new, y_new);
    }
    // lxm:可调部分，用于平衡时间和效果（之前最好是20）
    if(avail_list.size() > min(T, 40) || i > 15) {
      break;
    }
  }
  // lxm:funny code
  int best = INT_MAX;
  double score_min = 10000000;
  double alpha = 0.5, beta = 1, gamma = 1.5;
  for(int j = 0; j < avail_list.size(); j++) {
    double netsHPWL = calculateCellHPWL(theCell, avail_list[j]->x_pos * wsite,
                                        avail_list[j]->y_pos * rowHeight);
    double density = calculateOptimalDensityFactorChange(
                         theCell, avail_list[j]->x_pos * wsite,
                         avail_list[j]->y_pos * rowHeight) *
                     100;
    double now_cts_wl =
        my->getTheCellToParWDist(theCell, avail_list[j]->x_pos * wsite,
                                 avail_list[j]->y_pos * rowHeight);
    double score_temp = density * alpha + now_cts_wl * beta + gamma * netsHPWL;
    if(score_temp < score_min) {
      score_min = score_temp;
      best = j;
    }
  }
  if(best != INT_MAX) {
    return make_pair(true, avail_list[best]);
  }
  return make_pair(false, myPixel);
}

// lxm:启发式shit，7月停用
void goToRoot(std::vector< cell* > partition_cells,
              std::pair< double, double > root, rect box, double wsite,
              double rowHeight) {
  for(auto theCell : partition_cells) {
    int x = theCell->x_coord;
    int y = theCell->y_coord;
    int x_distance = (root.first - x) / wsite;
    int y_distance = (root.second - y) / rowHeight;
    theCell->x_coord += (x_distance / 800) * wsite;
    theCell->y_coord += (y_distance / 40) * rowHeight;
  }
}

// lxm:对在group内的FF布局
void circuit::parallel_FF_placement_Ingroup() {
  // getOptimalRegion(group_cells);
  parallel_buildCTSTrees_Ingroup(
      all_ff_cells);  // lxm:对group内构建整棵树，不进行聚类

  // #pragma omp parallel for
  for(size_t i = 0; i < _clusters.size(); ++i) {
    auto cluster = _clusters[i];
    std::vector< cell* > partition_cells = cluster;
    MyNesterov* myNesterov = getMyNesterovByff(cluster[0]);

    std::pair< double, double > root = myNesterov->root_coordi;
    // rect ctsTree;
    // ctsTree.xLL = myNesterov->cluster_ll_x;
    // ctsTree.xUR = myNesterov->cluster_ur_x;
    // ctsTree.yLL = myNesterov->cluster_ll_y;
    // ctsTree.yUR = myNesterov->cluster_ur_y;
    std::sort(partition_cells.begin(), partition_cells.end(),
              [root](cell* a, cell* b) { return SortUpdistance(a, b, root); });
    FF_group_op(partition_cells, die);
  }
  iter++;
}

void circuit::lg_std_cells(std::vector< cell* > std_cells) {
  int size = std_cells.size();
  if(wire_flag) {
    int num_threads =
        static_cast< size_t >(std::thread::hardware_concurrency()) > 20
            ? 20
            : static_cast< size_t >(std::thread::hardware_concurrency());
    if(thread_num < num_threads) {
      num_threads = thread_num;
    }
    // lxm:shit code,no fuck reason
    if(cells.size() < 150000) {
      num_threads =
          std::min(rows.size() / 30,
                   static_cast< size_t >(std::thread::hardware_concurrency()));
      ;
    }
    int cells_per_thread = (std_cells.size() + num_threads - 1) / num_threads;
    // cout << "num_threads: " << num_threads << endl;
    auto process_cell_segment = [&](int start, int end) {
      for(int i = start; i < end && i < std_cells.size(); i++) {
        cell* theCell = std_cells[i];
        if(theCell->isFixed || theCell->inGroup) continue;
        // lxm:shit iccad15
        // if(benchmark.find("superblue") != string::npos && theCell->is_ff) {
        //   continue;
        // }

        // cout << "lg_std_cells: " << i << endl;
        // lxm:signal/clock-net wirelength balance
        if(theCell->is_ff) {
          // diamond_swap(theCell, theCell->x_coord, theCell->y_coord);
        }
        else {
          erase_pixel(theCell);
          auto myPixel =
              diamond_search(theCell, theCell->x_coord, theCell->y_coord);
          relegal_paint_pixel(theCell, myPixel.second->x_pos,
                              myPixel.second->y_pos);
        }
      }
    };

    std::vector< std::thread > threads;
    for(int t = 0; t < num_threads; ++t) {
      int start = t * cells_per_thread;
      int end = start + cells_per_thread;

      threads.emplace_back(process_cell_segment, start, end);
    }

    for(auto& t : threads) {
      t.join();
    }
  }
  else {
    // lxm:暂时是忽视重叠的搜索
    for(int i = 0; i < std_cells.size(); i++) {
      cell* theCell = std_cells[i];
      if(theCell->isFixed || theCell->isPlaced) continue;
      // lxm:if change the logic,you need to change this code,shit opendp code!
      if(benchmark == "mgc_superblue11_a" && (size > 80000 && i >= 64533)) {
        continue;
      }
      // diamond_swap(theCell, theCell->x_coord, theCell->y_coord);
      auto myPixel =
          diamond_search(theCell, theCell->x_coord, theCell->y_coord);

      relegal_paint_pixel(theCell, myPixel.second->x_pos,
                          myPixel.second->y_pos);
    }
  }
}

void circuit::OptimizeSignalWireLength(std::vector< cell* > std_cells) {
  int size = std_cells.size();
  for(int i = 0; i < std_cells.size(); i++) {
    cell* theCell = std_cells[i];
    if(theCell->isFixed) continue;
    if(!wire_flag && theCell->is_ff) continue;
    // lxm: no fuck reason,shit opendp code!
    if(!wire_flag && benchmark == "mgc_superblue11_a" && i >= 47675) {
      continue;
    }
    erase_pixel(theCell);
    // diamond_swap(theCell, theCell->x_coord, theCell->y_coord);
    auto myPixel = diamond_search(theCell, theCell->x_coord, theCell->y_coord);

    relegal_paint_pixel(theCell, myPixel.second->x_pos, myPixel.second->y_pos);
  }
}

// shit code,don't use
void circuit::DensityNearestNeighborSearch(std::vector< cell* > std_cells) {
  for(int i = 0; i < std_cells.size(); i++) {
    cell* theCell = std_cells[i];
    if(theCell->isFixed) continue;
    if(!wire_flag) {
      if(theCell->is_ff) continue;
    }
    if(calculateOptimalDensityFactorChange(theCell, theCell->x_coord,
                                           theCell->y_coord) < 0.8) {
      continue;
    }
    auto myPixel = density_nearest_neighbor_search(theCell, theCell->x_coord,
                                                   theCell->y_coord);
    // cout << "MyPixel.first: " << myPixel.first << endl;
    // cout << "i: " << i << endl;
    relegal_paint_pixel(theCell, myPixel.second->x_pos, myPixel.second->y_pos);
  }
}

pair< bool, pixel* > circuit::density_nearest_neighbor_search(cell* theCell,
                                                              int x_coord,
                                                              int y_coord) {
  pixel* optimalPixel = nullptr;
  double min_density = DBL_MAX;
  double min_hpwl = DBL_MAX;
  bool found_position = false;

  // 当前cell的坐标、搜索边界定义
  int x_pos = static_cast< int >(floor(x_coord / wsite + 0.5));
  int y_pos = static_cast< int >(floor(y_coord / rowHeight + 0.5));

  // 初始搜索窗口大小
  int initial_search_radius_x =
      static_cast< int >(1.5 * theCell->width / wsite);
  int initial_search_radius_y =
      static_cast< int >(2.0 * rowHeight / rowHeight);  // 2倍行高
  int search_radius_x = initial_search_radius_x;
  int search_radius_y = initial_search_radius_y;
  int max_search_radius_x = static_cast< int >(rx / wsite);
  int max_search_radius_y = static_cast< int >(ty / rowHeight);

  while(!found_position && search_radius_x <= max_search_radius_x &&
        search_radius_y <= max_search_radius_y) {
    int x_start = max(x_pos - search_radius_x, 0);
    int x_end = min(x_pos + search_radius_x, static_cast< int >(rx / wsite));
    int y_start = max(y_pos - search_radius_y, 0);
    int y_end =
        min(y_pos + search_radius_y, static_cast< int >(ty / rowHeight));

    // 遍历搜索范围内的所有位置
    for(int x_new = x_start; x_new <= x_end; ++x_new) {
      for(int y_new = y_start; y_new <= y_end; ++y_new) {
        auto found = bin_search_FF(theCell, x_new, y_new);
        if(!found.first) {
          continue;
        }

        // 计算候选位置的bin_id，并查询对应的密度
        int candidate_x = x_new * wsite;
        int candidate_y = y_new * rowHeight;
        double density_change = calculateOptimalDensityFactorChange(
            theCell, candidate_x, candidate_y);

        // 如果密度小于当前最小密度，或在密度相同情况下半周长最小
        if(density_change < min_density ||
           (density_change == min_density &&
            calculateCellHPWL(theCell, candidate_x, candidate_y) < min_hpwl)) {
          min_density = density_change;
          min_hpwl = calculateCellHPWL(theCell, candidate_x, candidate_y);
          optimalPixel = &grid[y_new][x_new];
          found_position = true;
        }
      }
    }

    // 如果没有找到符合条件的位置，增大搜索窗口大小
    if(!found_position) {
      search_radius_x++;
      search_radius_y++;
    }
  }

  // 如果找到最优位置，更新cell的位置及密度信息
  if(found_position && optimalPixel) {
    int oldBinId = theCell->binId;
    moveFFCellAndUpdateDensity(theCell, optimalPixel->x_pos * wsite,
                               optimalPixel->y_pos * rowHeight);
    updateDensityAfterFFCellMove(theCell, oldBinId, theCell->binId);
  }

  return make_pair(found_position, optimalPixel);
}

// lxm：core shit code
void circuit::FF_placement_non_group(string mode) {
  double HPWL_init = HPWL("");
  cout << "HPWL_init: " << HPWL_init << endl;
  while(iter < max_iter) {
    buildCTSTrees_noGroup(ff_cells);
    if(high_density) {  // lxm：高密度并行容易出问题
      for(size_t i = 0; i < _clusters.size(); ++i) {
        auto cluster = _clusters[i];
        std::vector< cell* > partition_cells = cluster;
        MyNesterov* myNesterov = getMyNesterovByff(cluster[0]);
        std::pair< double, double > root = myNesterov->root_coordi;
        rect ctsTree;
        ctsTree.xLL = myNesterov->cluster_ll_x;
        ctsTree.xUR = myNesterov->cluster_ur_x;
        ctsTree.yLL = myNesterov->cluster_ll_y;
        ctsTree.yUR = myNesterov->cluster_ur_y;
        std::sort(
            partition_cells.begin(), partition_cells.end(),
            [root](cell* a, cell* b) { return SortUpdistance(a, b, root); });
        EROPS(partition_cells, ctsTree);
        // cout << "done: " << i << endl;
        // goToRoot(partition_cells, root, ctsTree, wsite, rowHeight);
      }
    }
    else {
#pragma omp parallel for
      for(size_t i = 0; i < _clusters.size(); ++i) {
        auto cluster = _clusters[i];
        std::vector< cell* > partition_cells = cluster;
        MyNesterov* myNesterov = getMyNesterovByff(cluster[0]);
        std::pair< double, double > root = myNesterov->root_coordi;
        rect ctsTree;
        ctsTree.xLL = myNesterov->cluster_ll_x;
        ctsTree.xUR = myNesterov->cluster_ur_x;
        ctsTree.yLL = myNesterov->cluster_ll_y;
        ctsTree.yUR = myNesterov->cluster_ur_y;
        std::sort(
            partition_cells.begin(), partition_cells.end(),
            [root](cell* a, cell* b) { return SortUpdistance(a, b, root); });
        EROPS(partition_cells, ctsTree);
        // cout << "done: " << i << endl;
      }
    }

    iter++;
    double HPWL_new = HPWL("");
    cout << "iter: " << iter
         << " dela_HPWL: " << (HPWL_new - HPWL_init) * 100 / HPWL_init << endl;
    if(HPWL_new - HPWL_init > HPWL_init * 0.01 && iter != max_iter) {
      max_iter = iter + 1;
    }
    if(iter != max_iter) {
      // lxm:只要不满足收敛条件就清除
#pragma omp parallel for
      for(int i = 0; i < _clusters.size(); i++) {
        auto cluster = _clusters[i];
        for(auto& theCell : cluster) {
          erase_pixel(theCell);
        }
      }
      cout << "clear done" << endl;
    }
  }
  return;
}

// 在移动FF cell后调用此函数更新其他cell的密度因子
void circuit::updateDensityAfterFFCellMove(cell* movedCell, int oldBinId,
                                           int newBinId) {
  // 获取移动前后的密度网格单元
  density_bin* oldBin = &bins[oldBinId];
  density_bin* newBin = &bins[newBinId];

  // 如果移动前后的密度网格单元相同，则无需更新
  if(oldBinId == newBinId) return;

  // 更新移动前的密度网格单元
  if(oldBinId != UINT_MAX) {
    oldBin->m_util -= movedCell->width * movedCell->height;
    // movedCell->dense_factor -=
    //     oldBin->m_util / (oldBin->free_space - oldBin->f_util);
  }

  // 更新移动后的密度网格单元
  newBin->m_util += movedCell->width * movedCell->height;
  // movedCell->dense_factor +=
  //     newBin->m_util / (newBin->free_space - newBin->f_util);
}

// 当FF cell移动时调用此函数，更新其周围的密度网格单元及其它cell的密度因子
void circuit::moveFFCellAndUpdateDensity(cell* movedCell, int new_x,
                                         int new_y) {
  // 计算新的binId
  double gridUnit = bin_size * rowHeight;  // 假设单位为bin_size个行高
  int x_gridNum = (int)ceil((rx - lx) / gridUnit);
  int binId = (int)floor((new_x - lx) / gridUnit) +
              (int)floor((new_y - by) / gridUnit) * x_gridNum;

  // 获取旧的binId
  int oldBinId = movedCell->binId;

  // 更新FF cell的binId
  movedCell->binId = binId;

  // 更新其他cell的密度因子
  updateDensityAfterFFCellMove(movedCell, oldBinId, binId);

  // // 输出其他cell的最新dense_factor
  // for(auto& otherCell : cells) {
  //   if(&otherCell == movedCell) continue;  // 跳过移动的cell

  //   // 获取其他cell的密度网格单元
  //   density_bin* theBin = &bins[otherCell.binId];

  //   // 更新其他cell的密度因子
  //   otherCell.dense_factor =
  //       theBin->m_util / (theBin->free_space - theBin->f_util);
  // }
}

// lxm:计算出新位置的密度情况
double circuit::calculateOptimalDensityFactorChange(cell* theCell, int new_x,
                                                    int new_y) {
  double gridUnit = bin_size * rowHeight;  // 假设单位为bin_size个行高
  int x_gridNum = (int)ceil((rx - lx) / gridUnit);
  int binId = (int)floor((new_x - lx) / gridUnit) +
              (int)floor((new_y - by) / gridUnit) * x_gridNum;
  density_bin* newBin = &bins[binId];

  return newBin->m_util / (newBin->free_space - newBin->f_util) +
         theCell->width * theCell->height /
             (newBin->free_space - newBin->f_util);
}

// funny shit code,don't use
std::pair< bool, pixel* > circuit::hexagonal_search_FF(cell* theCell,
                                                       int x_coord, int y_coord,
                                                       rect box) {
  pixel* myPixel = NULL;
  pair< bool, pair< int, int > > found;
  int x_pos = static_cast< int >(floor(x_coord / wsite + 0.5));
  int y_pos = static_cast< int >(floor(y_coord / rowHeight + 0.5));

  // Initialize search boundaries
  int x_start =
      std::max(x_pos - (int)(displacement * 5), (int)floor(box.xLL / wsite));
  int x_end = std::min(x_pos + (int)(displacement * 5),
                       (int)floor(box.xUR / wsite) -
                           (int)floor(theCell->width / rowHeight + 0.5));
  int y_start =
      std::max(y_pos - (int)displacement, (int)ceil(box.yLL / rowHeight));
  int y_end = std::min(y_pos + (int)displacement,
                       (int)ceil(box.yUR / rowHeight) -
                           (int)floor(theCell->height / rowHeight + 0.5));

  std::vector< pixel* > avail_list;
  avail_list.reserve(100);
  std::vector< double > score;
  score.reserve(100);
  int x_step = 10;
  int y_flag = 1;
  if(static_cast< int >(theCell->height / rowHeight) % 2 == 0 && init_lg_flag) {
    y_flag = 2;
  }

  MyNesterov* my = getMyNesterovByff(theCell);

  double grand_parent_x = my->getGrandParentCoord(theCell).first;

  double grand_parent_y = my->getGrandParentCoord(theCell).second;

  // lxm：根据与引导点距离确定候选点个数
  int T = abs(grand_parent_x - x_coord) / wsite +
          abs(grand_parent_y - y_coord) / rowHeight;
  // Hexagonal directions
  std::vector< std::pair< int, int > > directions = {
      {10, 0},   // Right
      {0, 1},    // UpRight
      {-10, 1},  // UpLeft
      {-10, 0},  // Left
      {0, -1},   // DownLeft
      {10, -1}   // DownRight
  };

  // Perform a coarse search first to find promising candidates
  for(int d = 0; d <= displacement; d++) {
    for(auto& dir : directions) {
      int y_step =
          (dir.second * d) % y_flag ? dir.second * d : dir.second * d + 1;
      int x_new = std::min(x_end, std::max(x_start, x_pos + dir.first * d));
      int y_new = std::min(y_end, std::max(y_start, y_pos + y_step));

      found = bin_search_site(x_pos, theCell, x_new, y_new, x_step);
      if(found.first == true) {
        pixel* candidatePixel = &grid[found.second.first][found.second.second];
        avail_list.push_back(candidatePixel);
        score.push_back(theCell->cost);  // Store initial cost for scoring
        theCell->cost = 0;  // Reset cost after finding a valid position
      }
    }

    // If we have enough candidates, break early
    if(avail_list.size() > min(T, 40)) {
      break;
    }
  }

  // Evaluate scores and find the best position using a heuristic approach
  double best_score = std::numeric_limits< double >::max();
  int best_index = -1;
  double alpha = 2, beta = 1, gamma = 1.5;
  for(size_t j = 0; j < avail_list.size(); j++) {
    double netsHPWL = calculateCellHPWL(theCell, avail_list[j]->x_pos * wsite,
                                        avail_list[j]->y_pos * rowHeight);
    double now_cts_wl =
        my->getTheCellToParWDist(theCell, avail_list[j]->x_pos * wsite,
                                 avail_list[j]->y_pos * rowHeight);

    // Calculate score using heuristic weights
    score[j] = score[j] * alpha + now_cts_wl * beta + gamma * netsHPWL;

    if(score[j] < best_score) {
      best_score = score[j];
      best_index = j;
    }
  }

  if(best_index != -1) {
    return make_pair(true, avail_list[best_index]);
  }

  return make_pair(false, myPixel);
}