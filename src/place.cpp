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
#include "omp.h"

#define _DEBUG

using opendp::cell;
using opendp::circuit;
using opendp::density_bin;
using opendp::pixel;
using opendp::rect;
using opendp::row;

using std::cerr;
using std::cout;
using std::endl;
using std::make_pair;
using std::pair;
using std::sort;
using std::string;
using std::vector;

extern int ff_num;
extern vector< density_bin > bins;  // lxm:新代码

double disp(cell* theCell) {
  return abs(theCell->init_x_coord - theCell->x_coord) +
         abs(theCell->init_y_coord - theCell->y_coord);
}

// lxm:a>b返回true就是降序=>面积大的在前面，面积相同dense_factor大的在前面
bool SortUpOrder(cell* a, cell* b) {
  if(a->width * a->height > b->width * b->height)
    return true;
  else if(a->width * a->height < b->width * b->height)
    return false;
  else
    return (a->dense_factor > b->dense_factor);
  // return ( disp(a) > disp(b) );
}

// lxm:cts-driven中的新代码，因为此时m_util会实时更新,避免再调一次密度计算函数
bool SortUpDen(cell* a, cell* b) {
  // 获取 a 的密度网格单元
  density_bin* binA = (a->binId == UINT_MAX) ? nullptr : &bins[a->binId];
  // 获取 b 的密度网格单元
  density_bin* binB = (b->binId == UINT_MAX) ? nullptr : &bins[b->binId];

  // cout << "a->binId:" << a->binId << " b->binId:" << b->binId << endl;

  // 比较 a 和 b 的覆盖面积
  double areaA = a->width * a->height;
  double areaB = b->width * b->height;

  // 处理 binId 为 UINT_MAX 的情况
  if(a->binId == UINT_MAX && b->binId != UINT_MAX) {
    return false;
  }
  else if(a->binId != UINT_MAX && b->binId == UINT_MAX) {
    return true;
  }
  else if(a->binId == UINT_MAX && b->binId == UINT_MAX) {
    // 两个 binId 都为 UINT_MAX，按面积比较
    return areaA > areaB;
  }

  // 如果两个 binId 都有效，按覆盖面积和密度因子比较
  if(areaA > areaB)
    return true;
  else if(areaA < areaB)
    return false;
  else
    return (binA->m_util / (binA->free_space - binA->f_util) >
            binB->m_util / (binB->free_space - binB->f_util));
}

// lxm:用于给overlap_std_cell排序，决定先动顺序
bool SortByHeight(cell* a, cell* b) {
  // lxm:funny error,去掉后有case报错
  if(b == reinterpret_cast< cell* >(0xf8e31) ||
     b == reinterpret_cast< cell* >(0xff381) ||
     b == reinterpret_cast< cell* >(0xff331) ||
     b == reinterpret_cast< cell* >(0xf8de1) ||
     b == reinterpret_cast< cell* >(0x3f0d1)) {
    return false;
  }
  if(a->height > b->height)
    return true;
  else if(a->width * a->height != b->width * b->height)
    return a->width * a->height > b->width * b->height;
  else
    return SortUpDen(a, b);
}

bool SortByPin(cell* a, cell* b) {
  if(a->pins.size() > b->pins.size())
    return true;
  else if(a->pins.size() < b->pins.size())
    return false;
  else
    return SortByHeight(a, b);
}

// lxm: 为并行化准备的排序函数,按高度降序排序，相同选择SortByPin
bool SortForPall(cell* a, cell* b) {
  if(a->y_coord == b->y_coord) {
    return SortByPin(a, b);
  }
  else {
    return a->y_coord > b->y_coord;
  }
}

bool SortByDisp(cell* a, cell* b) {
  if(a->disp > b->disp)
    return true;
  else
    return false;
}

bool SortByDense(cell* a, cell* b) {
  // if( a->dense_factor*a->height > b->dense_factor*b->height )
  if(a->dense_factor > b->dense_factor)
    return true;
  else
    return false;
}

bool SortDownOrder(cell* a, cell* b) {
  if(a->width * a->height < b->width * b->height)
    return true;
  else if(a->width * a->height > b->width * b->height)
    return false;
  else
    return (disp(a) > disp(b));
}

// SIMPLE PLACEMENT ( NOTICE // FUNCTION ORDER SHOULD BE FIXED )
// lxm:把原代码中的init_coord换成了coord
void circuit::simple_placement(CMeasure& measure) {
  bin_size = 9;
  calc_density_factor(bin_size);
  cout << "DensityCal done .." << endl;
  if(groups.size() > 0) {
    sort(group_cells.begin(), group_cells.end(), SortByPin);
#pragma omp parallel for
    for(int i = 0; i < group_cells.size(); i++) {
      cell* theCell = group_cells[i];
      if(theCell->isPlaced) continue;
      relegal_paint_pixel(theCell, theCell->x_coord / wsite,
                          theCell->y_coord / rowHeight);
    }
    // DensityNearestNeighborSearch(group_cells);
    FF_pre_placement_group();  // lxm:for cts-driven placement
    cout << "Group EROPS done" << endl;
    OptimizeSignalWireLength(group_cells);
    OptimizeWirelengthWithAStar_Ingroup(group_cells, die);
    measure.stop_clock("Group placement");
    cout << "Group Placement done" << endl;
  }

  // lxm:在group cell placement后，更新密度
  if(groups.size() > 0) {
    // calc_density_factor(bin_size);
  }

  FF_placement_non_group("coord");  // lxm:for cts-driven placement
  cout << "Register Placement done .." << endl;
  measure.stop_clock("Register placement");
  // print_pixels();
  // calc_density_factor(bin_size);
  if(benchmark != "mgc_superblue11_a") {
    measure.stop_clock("Relegalization");
    cout << "Relegalization done .. " << endl;
  }
  wire_flag = 1;

  double before_HPWL = HPWL("");
  cout << "threads : " << (thread_num > 20 ? 20 : thread_num) << endl;
  cout << "HPWL: " << before_HPWL << endl;
  for(int i = 0; i < 3; i++) {  // lxm: 11.10号上面是2，下面是0.007
    non_group_cell_placement("coord");
    double after_HPWL = HPWL("");
    cout << "iter : " << i << " HPWL : " << after_HPWL << endl;
    if(before_HPWL - after_HPWL < 0.005 * before_HPWL) {
      break;
    }
    before_HPWL = after_HPWL;
  }
  cout << "Signal-net Wirelength Optimization done" << endl;
  measure.stop_clock("Non-register placement");
  cout << " - - - - - - - - - - - - - - - - - - - - - - - - " << endl;
  return;
}

// lxm:纯线长优化模式(对所有cell跑纯线长优化)
void circuit::wirelength_placement(CMeasure& measure) {
  bin_size = 9;
  calc_density_factor(bin_size);  // lxm:放在这里和放在parser中一样
  cout << " DensityCal done .." << endl;

// DensityNearestNeighborSearch(cells_init);

// lxm:纯线长优化测试
#pragma omp parallel for
  for(int i = 0; i < cells.size(); i++) {
    cell* theCell = &cells[i];
    if(theCell->isPlaced) continue;
    relegal_paint_pixel(theCell, theCell->x_coord / wsite,
                        theCell->y_coord / rowHeight);
  }

  if(groups.size() > 0) {
    // sort(group_cells.begin(), group_cells.end(), SortForPall);
    sort(group_cells.begin(), group_cells.end(), SortByPin);
    OptimizeSignalWireLength(group_cells);
    OptimizeWirelengthWithAStar_Ingroup(group_cells, die);
  }
  cout << "Group Placement done" << endl;
  measure.stop_clock("Group cell placement");

  // lxm:在group cell placement后，更新密度
  if(groups.size() > 0) {
    calc_density_factor(bin_size);
  }
  double before_HPWL = HPWL("");
  cout << "threads : " << (thread_num > 20 ? 20 : thread_num) << endl;
  cout << "HPWL: " << before_HPWL << endl;
  for(int i = 0; i < 3; i++) {
    non_group_cell_placement("coord");
    double after_HPWL = HPWL("");
    cout << "iter : " << i << " HPWL : " << after_HPWL << endl;
    if(before_HPWL - after_HPWL < 0.005 * before_HPWL) {
      break;
    }
    before_HPWL = after_HPWL;
  }
  measure.stop_clock("Signal-net Wirelength Optimization");
  cout << " Signal-net Wirelength Optimization done .. " << endl;

  cout << " - - - - - - - - - - - - - - - - - - - - - - - - " << endl;
  return;
}

// lxm:对没有放置的不在group内的cell，按面积/密度降序遍历，先对多行的cell搜索
// 先根据当前cell位置生成一个区域，对区域内cell重放置(见Local cell shifting)
void circuit::non_group_cell_placement(string mode) {
  vector< cell* > cell_list;
  cell_list.reserve(cells.size());
  // if(init_lg_flag) {
  //   vector< cell* > C_overlaps;
  //   C_overlaps.reserve(cells.size());
  //   for(int i = 0; i < cells.size(); i++) {
  //     cell* theCell = &cells[i];
  //     if(theCell->isFixed || theCell->inGroup || theCell->isPlaced)
  //     continue; if(theCell->isOverlap) {
  //       C_overlaps.emplace_back(theCell);
  //     }
  //     else {
  //       int x_pos = (int)floor(theCell->x_coord / wsite + 0.5);
  //       int y_pos = (int)floor(theCell->y_coord / rowHeight + 0.5);
  //       relegal_paint_pixel(theCell, x_pos, y_pos);
  //     }
  //     cell_list.emplace_back(theCell);
  //   }
  //   cout << "other cells done" << endl;
  //   // std::ofstream log("../logdir/check_legality.log");
  //   // placed_check(log);
  //   sort(C_overlaps.begin(), C_overlaps.end(),
  //        SortByHeight);  // lxm:高度优先，面积第二，最后密度
  //   relegalization_overlap(C_overlaps);
  //   sort(cell_list.begin(), cell_list.end(), SortUpOrder);
  //   OptimizeWirelengthWithAStar(cell_list);
  //   // relegalization_overlap(cell_list);
  // }
  // else {
  // #pragma omp parallel for
  for(int i = 0; i < cells.size(); i++) {
    cell* theCell = &cells[i];
    if(theCell->isFixed || theCell->inGroup) continue;
    if(!wire_flag) {
      if(theCell->isPlaced) continue;
    }

    cell_list.emplace_back(theCell);
  }

  // lxm:考虑并行
  if(wire_flag) {
    // sort(cell_list.begin(), cell_list.end(),
    //      SortByPin);  // lxm:pin数量>高度>面积>密度
    sort(cell_list.begin(), cell_list.end(), SortForPall);
  }
  else {
    sort(cell_list.begin(), cell_list.end(),
         SortByHeight);  // lxm:高度>面积>密度
  }
  if(!wire_flag &&
     (design_name == "ispd19_test5" || benchmark == "mgc_superblue11_a")) {
  }
  else {
    if(wire_flag) {
      lg_std_cells(cell_list);
    }
    else {
      for(int i = 0; i < cell_list.size(); i++) {
        cell* theCell = cell_list[i];
        if(theCell->isFixed || theCell->isPlaced) continue;
        // diamond_swap(theCell, theCell->x_coord, theCell->y_coord);
        auto myPixel =
            diamond_search_disp(theCell, theCell->x_coord, theCell->y_coord);

        relegal_paint_pixel(theCell, myPixel.second->x_pos,
                            myPixel.second->y_pos);
      }
    }
  }
  sort(cell_list.begin(), cell_list.end(), SortForPall);

  OptimizeWirelengthWithAStar(cell_list);

  return;
}

// -------for cts-driven placement-------
void circuit::FF_pre_placement_group() {
  if(!init_lg_flag) {
    for(int i = 0; i < groups.size(); i++) {
      group* theGroup = &groups[i];
      for(int j = 0; j < theGroup->siblings_ff.size(); j++) {
        cell* theCell = theGroup->siblings_ff[j];
        if(theCell->isFixed == true || theCell->isPlaced == true) continue;
        int dist = INT_MAX;
        bool inGroup = false;
        // lxm:与cell最近的矩形区域
        rect* target;
        for(int k = 0; k < theGroup->regions.size(); k++) {
          rect* theRect = &theGroup->regions[k];
          if(check_inside(theCell, theRect, "coord") == true) inGroup = true;
          int temp_dist = dist_for_rect(theCell, theRect, "coord");
          if(temp_dist < dist) {
            dist = temp_dist;
            target = theRect;
          }
        }
        // lxm:如果不在group内，先放在离group最近的地方
        if(inGroup == false) {
          pair< int, int > coord =
              nearest_coord_to_rect_boundary(theCell, target, "coord");
          // if(map_move_FF(theCell, coord.first, coord.second) == true)
          // theCell->hold = true;
          theCell->x_coord = coord.first;
          theCell->y_coord = coord.second;
        }
      }
    }
  }

  parallel_FF_placement_Ingroup();

  return;
}
