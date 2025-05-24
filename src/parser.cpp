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
#include <iomanip>
#include <iostream>
#include <queue>
#include <vector>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <cmath>
#include <limits>
#include <algorithm>
#include <execution>
#include <limits>

#include "th/ns.h"
#include "th/Router.h"

// #include "/home/eda/th2/gurobi1103/linux64/include/gurobi_c++.h"
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
using std::fixed;
using std::ifstream;
using std::make_pair;
using std::max;
using std::min;
using std::numeric_limits;
using std::ofstream;
using std::pair;
using std::setprecision;
using std::string;
using std::to_string;
using std::vector;

const char* DEFCommentChar = "#";
const char* DEFLineEndingChar = ";";
const char* LEFCommentChar = "#";
const char* LEFLineEndingChar = ";";
const char* FFClkPortName = "ck";

extern double lower_left_x, lower_left_y, upper_right_x, upper_right_y;
extern int cap_mode;
extern double util;
extern vector< density_bin > bins;

extern std::mutex mtx;

inline bool operator<(const row& a, const row& b) {
  return (a.origY < b.origY) || (a.origY == b.origY && a.origX < b.origX);
}

void circuit::print_usage() {
  cout << "Incorrect arguments. exiting .." << endl;
  cout << "Usage1 : opendp -lef tech.lef -lef cell.lef -def "
          "placed.def -cpu 4 -placement_constraints placement.constraints "
          "-output_def lg.def"
       << endl;
  cout << "Usage2 : opendp -lef design.lef -def placed.def -cpu 4 "
          "-placement_constraints placement.constraints -output_def lg.def"
       << endl;

  return;
}

void circuit::read_files(int argc, char* argv[]) {
  vector< string > lefStor;
  string defLoc = "";

  char *cpu = NULL, *constraints = NULL, *out_def = NULL, *size = NULL;

  if(argc < 5) {
    print_usage();
    exit(1);
  }
  for(int i = 1; i < argc; i++) {
    if(strncmp(argv[i], "-cap", 4) == 0) cap_mode = 1;
    if(strncmp(argv[i], "-wire", 5) == 0) wire_flag = true;

    if(i + 1 != argc) {
      if(strncmp(argv[i], "-threads", 8) == 0) thread_num = atoi(argv[++i]);
      if(strncmp(argv[i], "-lef", 4) == 0)
        lefStor.emplace_back(argv[++i]);
      else if(strncmp(argv[i], "-def", 4) == 0)
        defLoc = argv[++i];
      else if(strncmp(argv[i], "-cpu", 4) == 0)
        cpu = argv[++i];
      else if(strncmp(argv[i], "-placement_constraints", 22) == 0)
        constraints = argv[++i];
      else if(strncmp(argv[i], "-output_def", 11) == 0)
        out_def = argv[++i];
      else if(strncmp(argv[i], "-group_ignore", 13) == 0)
        GROUP_IGNORE = true;
    }
  }

  if(lefStor.size() == 0 || defLoc == "") {
    print_usage();
    exit(1);
  }

  string tech_str, cell_lef_str, lef_str;
  string constraints_str;

  // Below should be modified!!
  /*
  if( lefStor.size() == 1 ) {
    read_lef(lefStor[0]);
  }
  else {
    read_tech_lef(lefStor[0]);
    read_cell_lef(lefStor[1]);
  }
  */

  size_t def_found = defLoc.find_last_of("/\\");
  string dir_bench = defLoc.substr(0, def_found);
  string dir = dir_bench.substr(0, dir_bench.find_last_of("/\\"));
  string bench = dir_bench.substr(dir_bench.find_last_of("/\\") + 1);

  // 根据lef文件名设置benchmark名称
  // size_t pos = lefStor[0].find_last_of("/\\");
  // if(pos != std::string::npos) {
  //   std::string filename = lefStor[0].substr(pos + 1);  // 提取最后的文件名
  //   size_t ext_pos = filename.find_last_of(".");
  //   if(ext_pos != std::string::npos) {
  //     benchmark = filename.substr(0, ext_pos);  // 去掉扩展名
  //   }
  // }
  benchmark = bench;

  ReadLef(lefStor);

  if(constraints != NULL) constraints_str = constraints;

  if(benchmark == "b19" || benchmark == "vga_lcd") {
    displacement = 200;
  }
  else if(benchmark == "leon3mp" || benchmark == "mgc_edit_dist" ||
          benchmark == "mgc_matrix_mult") {
    displacement = 300;
  }
  else if(benchmark == "superblue7" || benchmark == "superblue10") {
    displacement = 2631;  // 500*2000/380
  }
  else if(benchmark == "superblue1" || benchmark == "superblue3" ||
          benchmark == "superblue4" || benchmark == "superblue5" ||
          benchmark == "superblue16" || benchmark == "superblue18") {
    displacement = 2105;  // 400*2000/380
  }

  in_def_name = defLoc;

  if(out_def == NULL)
    out_def_name = "./" + bench + "_cts_lg.def";
  else
    out_def_name = out_def;

  cout << endl;
  cout << "-------------------- INPUT FILES ----------------------------------"
       << endl;
  cout << " benchmark name    : " << bench << endl;
  cout << " directory         : " << dir << endl;

  for(auto& curLefLoc : lefStor) {
    cout << " lef               : " << curLefLoc << endl;
  }
  cout << " def               : " << defLoc << endl;
  cout << "displacement       : " << displacement << endl;
  if(constraints != NULL)
    cout << " constraints       : " << constraints_str << endl;
  cout << "-------------------------------------------------------------------"
       << endl;

  // read_def shuld after read_lef
  //  read_def(defLoc, INIT);

  ReadDef(defLoc);
  //  exit(1);

  if(size != NULL) {
    string size_file = size;
    read_def_size(size_file);
  }
  power_mapping();

  if(constraints != NULL) read_constraints(constraints_str);

  // summary of benchmark
  calc_design_area_stats();

  // dummy cell generation
  dummy_cell.name = "FIXED_DUMMY";
  dummy_cell.isFixed = true;
  dummy_cell.isPlaced = true;

  // calc row / site offset
  int row_offset = rows[0].origY;
  int site_offset = rows[0].origX;

  // construct pixel grid
  int row_num = ty / rowHeight;
  int col = rx / wsite;
  grid = new pixel*[row_num];
  grid_init = new pixel*[row_num];
  for(int i = 0; i < row_num; i++) {
    grid[i] = new pixel[col];
    grid_init[i] = new pixel[col];
  }

  for(int i = 0; i < row_num; i++) {
    for(int j = 0; j < col; j++) {
      grid[i][j].name = "pixel_" + to_string(i) + "_" + to_string(j);
      grid[i][j].y_pos = i;
      grid[i][j].x_pos = j;
      grid[i][j].linked_cell = NULL;
      grid[i][j].isValid = false;
      if(init_lg_flag) {
        grid_init[i][j].name = "pixel_" + to_string(i) + "_" + to_string(j);
        grid_init[i][j].y_pos = i;
        grid_init[i][j].x_pos = j;
        grid_init[i][j].linked_cell = NULL;
        grid_init[i][j].isValid = false;
      }
    }
  }

  // Fragmented Row Handling
  for(auto& curFragRow : prevrows) {
    int x_start = IntConvert((1.0 * curFragRow.origX - core.xLL) / wsite);
    int y_start = IntConvert((1.0 * curFragRow.origY - core.yLL) / rowHeight);

    int x_end = x_start + curFragRow.numSites;
    int y_end = y_start + 1;

    // cout << "x_start: " << x_start << endl;
    // cout << "y_start: " << y_start << endl;
    // cout << "x_end: " << x_end << endl;
    // cout << "y_end: " << y_end << endl;
    for(int i = x_start; i < x_end; i++) {
      for(int j = y_start; j < y_end; j++) {
        grid[j][i].isValid = true;
      }
    }
  }

  /*
  for(int i = 0; i < rows.size(); i++) {
    // original rows : Fragmented ROWS
    row* myRow = &rows[i];

    int col_size = myRow->numSites;
    for(int j = 0; j < col_size; j++) {
      int y_pos = (myRow->origY-core.yLL) / rowHeight;
      int x_pos = j + (myRow->origX-core.xLL) / wsite;
      grid[y_pos][x_pos].isValid = true;
    }
  }
  */
  {
    lower_left_x = lx;
    lower_left_y = by;
    upper_right_x = rx;
    upper_right_y = ty;
  }  // th
  // cout << "lx: " << lx << " by: " << by << " rx: " << rx << " ty: " << ty
  //      << endl;

  // fixed cell marking
  fixed_cell_assign();
  init_ff_cell();  // lxm:初始Flip-Flop cell
  if(fix_cells.size() > 0) {
    // move_cells_to_nearest_fix();
  }

  // group id mapping & x_axis dummycell insertion
  // group_pixel_assign_2();  // lxm:会被group_pixel_assign覆盖结果
  // y axis dummycell insertion
  if(groups.size() > 0) {
    group_pixel_assign();
  }
  return;
}

//---------- lxm:for cts-driven ----------------

// 这个函数接受输入数字，并将其映射到0到1之间
double scale_to_unit_interval(int input) {
  // 假设输入在1到1000之间，将其映射到0到1之间
  // 使用对数函数使映射更加平滑
  const double max_input = 10.0;
  const double min_input = 1.0;
  const double log_max = std::log(max_input);
  const double log_min = std::log(min_input);

  // 计算对数值并进行归一化
  double log_input = std::log(input);
  double normalized_value = (log_input - log_min) / (log_max - log_min);

  return normalized_value;
}

// lxm:将所有在fix cell内的非fixed cell移动到最近的边界
void circuit::move_cells_to_nearest_fix() {
  rect* target;
  int dist = INT_MAX;  // lxm:尝试找最近的区域
  pair< int, int > coord;
  bool inFix = false;
#pragma omp parallel for shared(cells, fix_rects) private(target, dist, coord, \
                                                              inFix)
  for(int i = 0; i < cells.size(); ++i) {
    // cell* theCell = ff_cells[i];
    cell* theCell = &cells[i];
    rect* target = nullptr;
    int dist = INT_MAX;
    pair< int, int > coord;
    bool inFix = false;

    if(!theCell->isFixed) {
      for(int k = 0; k < fix_rects.size(); k++) {
        rect* theRect = &fix_rects[k];
        if(check_overlap(theCell, theRect, "coord")) {
          inFix = true;
          int current_dist = dist_for_rect(theCell, theRect, "coord");
          if(current_dist < dist) {
            dist = current_dist;
            target = theRect;
          }
        }
      }
    }

    if(inFix) {
      coord = nearest_coord_to_rect_boundary(theCell, target, "coord");
      theCell->x_coord = coord.first;
      theCell->y_coord = coord.second;
    }
  }
}
struct AStarNode {
  int x, y;
  double cost, heuristic, total;
  AStarNode* parent;

  AStarNode(int x, int y, double cost, double heuristic,
            AStarNode* parent = nullptr)
      : x(x),
        y(y),
        cost(cost),
        heuristic(heuristic),
        total(cost + heuristic),
        parent(parent) {}
};

struct CompareNode {
  bool operator()(AStarNode* a, AStarNode* b) { return a->total > b->total; }
};

// lxm:计算给定的cell的net的总线长和
double circuit::calculateCLKHPWL(cell* theCell, int x, int y) {
  double hpwl = 0;

  double x_coord = 0;
  double y_coord = 0;
  for(int i : theCell->connected_nets) {
    rect box;
    net* theNet = &nets[i];
    if(theNet->name != "ispd_clk" || theNet->name != "iccad_clk") {
      continue;
    }
    // cout << " net name : " << theNet->name << endl;
    pin* source = &pins[theNet->source];

    if(source->type == NONPIO_PIN) {
      cell* Cell = &cells[source->owner];
      if(Cell == theCell) {
        x_coord = x;
        y_coord = y;
      }
      else {
        x_coord = Cell->x_coord;
        y_coord = Cell->y_coord;
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
        cell* Cell = &cells[sink->owner];
        if(Cell == theCell) {
          x_coord = x;
          y_coord = y;
        }
        else {
          x_coord = Cell->x_coord;
          y_coord = Cell->y_coord;
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
    // cout << "box.xLL: " << box.xLL << " box.xUR: " << box.xUR
    //      << "box.yLL: " << box.xLL << " box.yUR: " << box.xUR << endl;
    double box_boundary = (box.xUR - box.xLL + box.yUR - box.yLL);

    hpwl += box_boundary;
  }
  return hpwl / static_cast< double >(DEFdist2Microns);
  // return HPWL("");
}

double circuit::calculateCellHPWL(cell* theCell, int x, int y) {
  double hpwl = 0;

  double x_coord = 0;
  double y_coord = 0;
  for(int i : theCell->connected_nets) {
    rect box;
    net* theNet = &nets[i];
    if(theNet->name == "ispd_clk" || theNet->name == "iccad_clk") {
      continue;
    }
    // cout << " net name : " << theNet->name << endl;
    pin* source = &pins[theNet->source];

    if(source->type == NONPIO_PIN) {
      cell* Cell = &cells[source->owner];
      if(Cell == theCell) {
        x_coord = x;
        y_coord = y;
      }
      else {
        x_coord = Cell->x_coord;
        y_coord = Cell->y_coord;
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
        cell* Cell = &cells[sink->owner];
        if(Cell == theCell) {
          x_coord = x;
          y_coord = y;
        }
        else {
          x_coord = Cell->x_coord;
          y_coord = Cell->y_coord;
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
    // cout << "box.xLL: " << box.xLL << " box.xUR: " << box.xUR
    //      << "box.yLL: " << box.xLL << " box.yUR: " << box.xUR << endl;
    double box_boundary = (box.xUR - box.xLL + box.yUR - box.yLL);

    hpwl += box_boundary;
  }
  return hpwl / static_cast< double >(DEFdist2Microns);
  // return HPWL("");
}

// lxm:对cell的当前位置进行HPWL优化
opendp::point circuit::findOptimalStdCellPosition(cell* theCell,
                                                  double initial_hpwl) {
  auto heuristic = [theCell, this](int x, int y) {
    return std::abs((x - theCell->init_x_coord) / wsite) +
           std::abs((y - theCell->init_y_coord) / rowHeight);
  };
  auto get_hpwl_change = [&](int x, int y) {
    double hpwl = calculateCellHPWL(theCell, x, y);
    return hpwl - initial_hpwl;
  };
  auto getDensityMultiplier = [&](int x, int y) {
    double densityFactorChange =
        calculateOptimalDensityFactorChange(theCell, x, y);
    return densityFactorChange;
  };

  std::priority_queue< AStarNode*, std::vector< AStarNode* >, CompareNode >
      open_set;
  std::unordered_map< int, AStarNode* > all_nodes;
  int original_x = theCell->x_coord, original_y = theCell->y_coord;
  int initial_x = theCell->init_x_coord, initial_y = theCell->init_y_coord;

  AStarNode* start_node =
      new AStarNode(original_x, original_y, 0, heuristic(initial_x, initial_y));
  open_set.push(start_node);
  all_nodes[original_x * 10000 + original_y] = start_node;

  int count = 0;
  AStarNode* best_node = start_node;
  int x_step = theCell->width / wsite;
  int y_step = std::ceil(theCell->height / rowHeight);
  y_step = y_step % 2 == 0 ? 2 : 1;
  while(!open_set.empty()) {
    AStarNode* current = open_set.top();
    open_set.pop();

    if(current->total > best_node->total) {
      if(++count >= 2) {
        break;
      }
    }

    std::array< opendp::point, 14 > neighbor_offsets = {
        opendp::point{wsite * x_step, 0},
        opendp::point{-wsite * x_step, 0},
        opendp::point{0, rowHeight * y_step * 2},
        opendp::point{0, -rowHeight * y_step * 2},
        opendp::point{wsite * x_step / 2, 0},
        opendp::point{-wsite * x_step / 2, 0},
        opendp::point{0, rowHeight * y_step},
        opendp::point{0, -rowHeight * y_step},
        opendp::point{wsite * x_step, rowHeight * y_step},
        opendp::point{wsite * x_step, -rowHeight * y_step},
        opendp::point{wsite * x_step * 2, 0},
        opendp::point{-wsite * x_step, rowHeight * y_step},
        opendp::point{-wsite * x_step, -rowHeight * y_step},
        opendp::point{-wsite * x_step * 2, 0}};

    for(const auto& offset : neighbor_offsets) {
      int nx = current->x + offset.x, ny = current->y + offset.y;
      if(nx < 0 || nx > die.xUR || ny < 0 || ny > die.yUR) continue;
      int manhattan_distance =
          (std::abs(nx - initial_x) + std::abs(ny - initial_y)) / wsite;
      if(manhattan_distance >= displacement) continue;

      auto found = bin_search_FF(theCell, nx / wsite, ny / rowHeight);
      if(!found.first) continue;

      double new_cost = current->cost + 1;
      double hpwl_change = get_hpwl_change(nx, ny);
      double score = hpwl_change * 2000 + heuristic(nx, ny);

      AStarNode* neighbor_node = all_nodes[nx * 10000 + ny];
      if(!neighbor_node || score < neighbor_node->total) {
        if(neighbor_node) {
          neighbor_node->cost = new_cost;
          neighbor_node->heuristic = heuristic(nx, ny);
          neighbor_node->total = score;
          neighbor_node->parent = current;
        }
        else {
          neighbor_node =
              new AStarNode(nx, ny, new_cost, heuristic(nx, ny), current);
          open_set.push(neighbor_node);
          all_nodes[nx * 10000 + ny] = neighbor_node;
        }

        if(score < best_node->total) best_node = neighbor_node;
      }
    }
  }

  opendp::point optimal_position(best_node->x, best_node->y);

  for(auto& [_, node] : all_nodes) delete node;

  return optimal_position;
}

// lxm:对在group内的cell的当前位置进行HPWL优化
opendp::point circuit::findOptimalStdCellPosition_Ingroup(cell* theCell,
                                                          double initial_hpwl,
                                                          rect box) {
  auto heuristic = [theCell, this](int x, int y) {
    return std::abs((x - theCell->init_x_coord) / wsite) +
           std::abs((y - theCell->init_y_coord) / rowHeight);
  };
  auto get_hpwl_change = [&](int x, int y) {
    double hpwl = calculateCellHPWL(theCell, x, y);
    return hpwl - initial_hpwl;
  };
  auto getDensityMultiplier = [&](int x, int y) {
    double densityFactorChange =
        calculateOptimalDensityFactorChange(theCell, x, y);
    return densityFactorChange;
  };

  std::priority_queue< AStarNode*, std::vector< AStarNode* >, CompareNode >
      open_set;
  std::unordered_map< int, AStarNode* > all_nodes;
  int original_x = theCell->x_coord, original_y = theCell->y_coord;
  int initial_x = theCell->init_x_coord, initial_y = theCell->init_y_coord;
  AStarNode* start_node =
      new AStarNode(original_x, original_y, 0, heuristic(initial_x, initial_y));
  open_set.push(start_node);
  all_nodes[original_x * 10000 + original_y] = start_node;
  int count = 0;
  AStarNode* best_node = start_node;
  while(!open_set.empty()) {
    AStarNode* current = open_set.top();
    open_set.pop();

    // 停止条件：当前节点的得分没有比最优节点更好
    if(current->total > best_node->total) {
      if(++count >= 2) {
        break;
      }
    }
    // int manhattan_distance = std::abs(current->x - initial_x) / wsite +
    //                          std::abs(current->y - initial_y) / wsite;
    // // // lxm:加上了最大位移限制
    // if(manhattan_distance >= displacement) {
    //   continue;
    // }
    std::vector< opendp::point > neighbors;
    int x_step = theCell->width / wsite;  // lxm:一次动2个site
    int y_step = std::ceil(theCell->height / rowHeight);
    y_step = y_step % 2 == 0 ? 2 : 1;

    // lxm:新代码，考虑到多行高和单行高拼接的问题，可能会导致没办法搜到空缺
    neighbors.push_back({current->x + wsite * x_step, current->y});
    neighbors.push_back({current->x - wsite * x_step, current->y});
    neighbors.push_back({current->x, current->y + rowHeight * y_step * 2});
    neighbors.push_back({current->x, current->y - rowHeight * y_step * 2});

    neighbors.push_back({current->x + wsite * x_step / 2, current->y});
    neighbors.push_back({current->x - wsite * x_step / 2, current->y});
    neighbors.push_back({current->x, current->y + rowHeight * y_step});
    neighbors.push_back({current->x, current->y - rowHeight * y_step});

    neighbors.push_back({current->x + wsite * x_step,
                         current->y + rowHeight * y_step});  // 右上
    neighbors.push_back({current->x + wsite * x_step,
                         current->y - rowHeight * y_step});              // 右下
    neighbors.push_back({current->x + wsite * x_step * 2, current->y});  // 右
    neighbors.push_back({current->x - wsite * x_step,
                         current->y + rowHeight * y_step});  // 左上
    neighbors.push_back({current->x - wsite * x_step,
                         current->y - rowHeight * y_step});              // 左下
    neighbors.push_back({current->x - wsite * x_step * 2, current->y});  // 左

    for(const auto& neighbor : neighbors) {
      int nx = neighbor.x, ny = neighbor.y;
      if(nx < 0 || nx > die.xUR || ny < 0 || ny > die.yUR) continue;
      int manhattan_distance = (std::abs(current->x - initial_x) +
                                std::abs(current->y - initial_y)) /
                               wsite;
      if(manhattan_distance >= displacement) {
        continue;
      }
      auto found = bin_search_FF(theCell, nx / wsite, ny / rowHeight);
      if(!found.first) continue;

      double new_cost = current->cost + 1;
      double hpwl_change = get_hpwl_change(nx, ny);
      double score =
          hpwl_change * 2000 +
          heuristic(nx, ny);  // lxm:加上了步数和线长变化(加上密度后时间会爆炸)

      AStarNode* neighbor_node = all_nodes[nx * 10000 + ny];
      if(!neighbor_node || score < neighbor_node->total) {
        if(neighbor_node) {
          neighbor_node->cost = new_cost;
          neighbor_node->heuristic = heuristic(nx, ny);
          neighbor_node->total = score;
          neighbor_node->parent = current;
        }
        else {
          neighbor_node =
              new AStarNode(nx, ny, new_cost, heuristic(nx, ny), current);
          open_set.push(neighbor_node);
          all_nodes[nx * 10000 + ny] = neighbor_node;
        }

        if(score < best_node->total) best_node = neighbor_node;
      }
    }
  }

  opendp::point optimal_position(best_node->x, best_node->y);
  for(auto& [_, node] : all_nodes) delete node;

  return optimal_position;
}

// lxm:对在group内的cell的当前位置进行HPWL优化
opendp::point circuit::findOptimalFFPosition_Ingroup(cell* theCell,
                                                     double initial_hpwl,
                                                     rect box) {
  auto heuristic = [theCell, this](int x, int y) {
    return std::abs((x - theCell->init_x_coord) / wsite) +
           std::abs((y - theCell->init_y_coord) / rowHeight);
  };
  auto get_hpwl_change = [&](int x, int y) {
    double hpwl =
        calculateCLKHPWL(theCell, x, y) + calculateCellHPWL(theCell, x, y);
    return hpwl - initial_hpwl;
  };
  auto getDensityMultiplier = [&](int x, int y) {
    double densityFactorChange =
        calculateOptimalDensityFactorChange(theCell, x, y);
    return densityFactorChange;
  };

  std::priority_queue< AStarNode*, std::vector< AStarNode* >, CompareNode >
      open_set;
  std::unordered_map< int, AStarNode* > all_nodes;
  int original_x = theCell->x_coord, original_y = theCell->y_coord;

  AStarNode* start_node = new AStarNode(original_x, original_y, 0,
                                        heuristic(original_x, original_y));
  open_set.push(start_node);
  all_nodes[original_x * 10000 + original_y] = start_node;

  AStarNode* best_node = start_node;
  while(!open_set.empty()) {
    AStarNode* current = open_set.top();
    open_set.pop();

    // 停止条件：当前节点的得分没有比最优节点更好
    if(current->total > best_node->total) {
      break;
    }
    int manhattan_distance = std::abs(current->x - original_x) / wsite +
                             std::abs(current->y - original_y) / rowHeight;
    // lxm:加上了最大位移限制
    // if(manhattan_distance >= displacement / 20) {
    //   best_node = current;
    //   break;
    // }
    std::vector< opendp::point > neighbors;
    int x_step = 2;  // lxm:一次动2个site
    int y_step = std::ceil(theCell->height / rowHeight);
    y_step = y_step % 2 == 0 ? 2 : 1;

    // lxm:新代码，考虑到多行高和单行高拼接的问题，可能会导致没办法搜到空缺
    neighbors.push_back({current->x + wsite, current->y});
    neighbors.push_back({current->x - wsite, current->y});
    neighbors.push_back({current->x, current->y + rowHeight * y_step * 2});
    neighbors.push_back({current->x, current->y - rowHeight * y_step * 2});

    neighbors.push_back({current->x + wsite * x_step, current->y});
    neighbors.push_back({current->x - wsite * x_step, current->y});
    neighbors.push_back({current->x, current->y + rowHeight * y_step});
    neighbors.push_back({current->x, current->y - rowHeight * y_step});

    for(const auto& neighbor : neighbors) {
      int nx = neighbor.x, ny = neighbor.y;
      if(nx < box.xLL || nx > box.xUR || ny < box.yLL || ny > box.yUR) continue;

      auto found = bin_search_FF(theCell, nx / wsite, ny / rowHeight);
      if(!found.first) continue;

      double new_cost = current->cost + 1;
      double hpwl_change = get_hpwl_change(nx, ny);
      double score =
          hpwl_change * 10 +
          heuristic(nx, ny);  // lxm:加上了步数和线长变化(加上密度后时间会爆炸)

      AStarNode* neighbor_node = all_nodes[nx * 10000 + ny];
      if(!neighbor_node || score < neighbor_node->total) {
        if(neighbor_node) {
          neighbor_node->cost = new_cost;
          neighbor_node->heuristic = heuristic(nx, ny);
          neighbor_node->total = score;
          neighbor_node->parent = current;
        }
        else {
          neighbor_node =
              new AStarNode(nx, ny, new_cost, heuristic(nx, ny), current);
          open_set.push(neighbor_node);
          all_nodes[nx * 10000 + ny] = neighbor_node;
        }

        if(score < best_node->total) best_node = neighbor_node;
      }
    }
  }

  opendp::point optimal_position(best_node->x, best_node->y);
  for(auto& [_, node] : all_nodes) delete node;

  return optimal_position;
}

// lxm:高效(搞笑)的最优位置搜索==>后续加上每次迭代不重叠
double circuit::EROPS(std::vector< cell* > partition_cells, rect box) {
  int i = 0;
  for(cell* theCell : partition_cells) {
    i++;
    if(theCell->isFixed || theCell->isPlaced) continue;
    pair< bool, pixel* > myPixel;
    // cout << "i: " << i << endl;
    myPixel =
        diamond_search_FF(theCell, theCell->x_coord, theCell->y_coord, box);
    if(iter != max_iter - 1) {
      // theCell->x_coord = myPixel.second->x_pos * wsite;
      // theCell->y_coord = myPixel.second->y_pos * rowHeight;
      relegal_paint_pixel(theCell, myPixel.second->x_pos,
                          myPixel.second->y_pos);
    }
    else {
      relegal_paint_pixel(theCell, myPixel.second->x_pos,
                          myPixel.second->y_pos);
    }

    // opendp::point optimalPosition;
    // optimalPosition.x = myPixel.second->x_pos * wsite;
    // optimalPosition.y = myPixel.second->y_pos * rowHeight;
    moveFFCellAndUpdateDensity(theCell, theCell->x_coord, theCell->y_coord);
  }

  return 0;
}

// lxm:对group内的FF进行处理
void circuit::FF_group_op(std::vector< cell* > partition_cells, rect box) {
  int i = 0;
  for(cell* theCell : partition_cells) {
    i++;

    if(theCell->isFixed || theCell->isPlaced) continue;
    pair< bool, pixel* > myPixel;
    if(!theCell->inGroup) {
      myPixel =
          diamond_search_FF(theCell, theCell->x_coord, theCell->y_coord, box);
      theCell->x_coord = myPixel.second->x_pos * wsite;
      theCell->y_coord = myPixel.second->y_pos * rowHeight;
    }
    else {
      erase_pixel(theCell);
      myPixel = diamond_search_FF_group(theCell, theCell->x_coord,
                                        theCell->y_coord, box);
      relegal_paint_pixel(theCell, myPixel.second->x_pos,
                          myPixel.second->y_pos);
    }

    moveFFCellAndUpdateDensity(theCell, theCell->x_coord, theCell->y_coord);
  }
}

double circuit::OptimizeWirelengthWithAStar(std::vector< cell* >& std_cells) {
  int num_threads = std::thread::hardware_concurrency() > 20
                        ? 20
                        : std::thread::hardware_concurrency();
  if(thread_num < num_threads) {
    num_threads = thread_num;
  }
  // 选取每百分之多少的 cell 进行并行处理
  int cells_per_thread = (std_cells.size() + num_threads - 1) / num_threads;
  // cout << "num_threads: " << num_threads << endl;
  auto process_cell_segment = [&](int start, int end) {
    for(int i = start; i < end && i < std_cells.size(); i++) {
      cell* theCell = std_cells[i];
      if(theCell->isFixed || theCell->inGroup || theCell->is_ff) continue;
      double original_hpwl =
          calculateCellHPWL(theCell, theCell->x_coord, theCell->y_coord);
      opendp::point optimalPosition =
          findOptimalStdCellPosition(theCell, original_hpwl);
      std::lock_guard< std::mutex > lock(mtx);
      update_pixel(theCell, optimalPosition.x / wsite,
                   optimalPosition.y / rowHeight);
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
  return 0;
}

double circuit::OptimizeWirelengthWithAStar_Ingroup(
    std::vector< cell* >& std_cells, rect box) {
  for(int i = 0; i < std_cells.size(); i++) {
    cell* theCell = std_cells[i];
    if(theCell->isFixed || theCell->is_ff) continue;
    group* theGroup = &groups[group2id[theCell->group]];
    box = theGroup->boundary;
    double original_hpwl =
        calculateCellHPWL(theCell, theCell->x_coord, theCell->y_coord);
    std::pair< bool, opendp::pixel* > myPixel;

    opendp::point optimalPosition =
        findOptimalStdCellPosition_Ingroup(theCell, original_hpwl, box);
    update_pixel(theCell, optimalPosition.x / wsite,
                 optimalPosition.y / rowHeight);
  }

  return 0;
}

// lxm:找出所有的FF并存储
void circuit::init_ff_cell() {
  int j = 0;
#pragma omp paragma parallel for shared(cells, macros, ff_cells, groups, \
                                            group2id, j)
  for(int i = 0; i < cells.size(); i++) {
    cell* theCell = &cells[i];
    macro* theMacro = &macros[theCell->type];
    if(!theCell->isFixed) {
      cells_init.emplace_back(theCell);
    }
    if(theMacro->isFlop == true &&
       !theCell->isFixed) {  // lxm:加上了fix的判断，fix的不认为是FF

      theCell->nets_hpwl =
          calculateCellHPWL(theCell, theCell->x_coord, theCell->y_coord);
      if(cap_mode)
        theCell->capacitance = (theCell->width * theCell->height) * 0.16;
      theCell->cluster_id = -1;
      theCell->is_ff = true;
      // cout << "FF cell: " << theCell->name << endl;
      // cout << "FF cell type: " << theMacro->name << endl;
      if(theCell->inGroup) {
        groups[group2id[theCell->group]].siblings_ff.emplace_back(theCell);
        j++;
      }
      else {
        ff_cells.emplace_back(theCell);
      }
      all_ff_cells.emplace_back(theCell);
    }
  }
  cout << "ALL FF cells num: " << all_ff_cells.size() << endl;
  cout << "FF cells num in groups: " << j << endl;
}

//----------cts-end------------

// lxm:计算可移动、fixed cell的面积、最高的cell高度
void circuit::calc_design_area_stats() {
  num_fixed_nodes = 0;
  int macro_num = 0;
  max_height = 0;
  int move_cell_num = 0;
  avg_cell_area = 0.0;
  // total_mArea : total movable cell area that need to be placed
  // total_fArea : total fixed cell area.
  // designArea : total available place-able area.
  //
  total_mArea = total_fArea = designArea = 0.0;
  for(vector< cell >::iterator theCell = cells.begin(); theCell != cells.end();
      ++theCell) {
    if(theCell->isFixed) {
      total_fArea += theCell->width * theCell->height;
      num_fixed_nodes++;
    }
    else if(theCell->height / rowHeight > 4) {
      // cout << theCell->name << " is fixed" << endl;
      total_fArea += theCell->width * theCell->height;
      macro_num++;
      theCell->isFixed = true;
      theCell->isPlaced = true;
    }
    else {
      total_mArea += theCell->width * theCell->height;
      max_cell_width =
          max_cell_width > theCell->width ? max_cell_width : theCell->width;
      max_height = max_height > theCell->height ? max_height : theCell->height;
      // avg_cell_area += theCell->width * theCell->height;
      move_cell_num++;
      // cout << theCell->name << " is movable"
      //      << "isPlaced: " << theCell->isPlaced << endl;
    }
  }
  for(vector< row >::iterator theRow = rows.begin(); theRow != rows.end();
      ++theRow)
    designArea += theRow->stepX * theRow->numSites *
                  sites[theRow->site].height *
                  static_cast< double >(DEFdist2Microns);

  unsigned multi_num = 0;
  for(int i = 0; i < cells.size(); i++) {
    cell* theCell = &cells[i];
    macro* theMacro = &macros[theCell->type];
    if(theCell->height / rowHeight > 1 && !theCell->isFixed) {  // lxm
      theMacro->isMulti = true;
    }
    else {
      theMacro->isMulti = false;
    }
    // cout << theCell->name << " " << theMacro->name << " " <<
    // theMacro->isMulti
    //      << endl;
    if(theMacro->isMulti == true) {
      multi_num++;
    }
  }

  for(int i = 0; i < cells.size(); i++) {
    cell* theCell = &cells[i];
    macro* theMacro = &macros[theCell->type];
    if(theCell->isFixed == false && theMacro->isMulti == true &&
       theMacro->type == "CORE") {
      if(max_cell_height <
         static_cast< int >(theMacro->height * DEFdist2Microns / rowHeight +
                            0.5))
        max_cell_height = static_cast< int >(
            theMacro->height * DEFdist2Microns / rowHeight + 0.5);
    }
  }
  // avg_cell_area /= move_cell_num;
  // avg_cell_area /= max_utilization;  // lxm:参考replace，获取bin的理想面积
  design_util = total_mArea / (designArea - total_fArea);
  multi_number = multi_num;  // lxm:用来存放多行高cell的数量

  cout << "-------------------- DESIGN ANALYSIS ------------------------------"
       << endl;
  cout << "  total cells              : " << cells.size() << endl;
  cout << "  multi cells              : " << multi_num << endl;
  cout << "  fixed cells              : " << num_fixed_nodes << endl;
  cout << "  total nets               : " << nets.size() << endl;
  cout << "  move macros              : " << macro_num << endl;
  cout << "  design area              : " << designArea << endl;
  cout << "  total f_area             : " << total_fArea << endl;
  cout << "  total m_area             : " << total_mArea << endl;
  cout << "  design util              : " << design_util * 100.00 << endl;
  cout << "  num rows                 : " << rows.size() << endl;
  cout << "  row height               : " << rowHeight << endl;
  if(max_cell_height > 1)
    cout << "  max multi_cell height    : " << max_cell_height << endl;
  if(max_disp_const > 0)
    cout << "  max disp const           : " << max_disp_const << endl;
  if(groups.size() > 0)
    cout << "  group num                : " << groups.size() << endl;
  cout << "-------------------------------------------------------------------"
       << endl;
  if(groups.size() > 0) {
    max_iter = 10;
  }
  else {
    max_iter = 10;
  }
  iter = 0;
  //
  // design_utilization error handling.
  //
  if(design_util >= 1.001) {
    cout << "ERROR:  Utilization exceeds 100%. (" << fixed << setprecision(2)
         << design_util * 100.00 << "%)! ";
    cout << "        Please double check your input files!" << endl;
    exit(1);
  }

  return;
}

void circuit::read_constraints(const string& input) {
  //    cout << " .constraints file : " << input << endl;
  ifstream dot_constraints(input.c_str());
  if(!dot_constraints.good()) {
    cerr << "read_constraints:: cannot open '" << input << "' for reading"
         << endl;
  }

  string context;

  while(!dot_constraints.eof()) {
    dot_constraints >> context;
    if(dot_constraints.eof()) break;
    if(strncmp(context.c_str(), "maximum_utilization", 19) == 0) {
      string temp = context.substr(0, context.find_last_of("%"));
      string max_util = temp.substr(temp.find_last_of("=") + 1);
      max_utilization = atof(max_util.c_str());
      util = max_utilization;
    }
    else if(strncmp(context.c_str(), "maximum_movement", 16) == 0) {
      string temp = context.substr(0, context.find_last_of("rows"));
      string max_move = temp.substr(temp.find_last_of("=") + 1);
      // displacement = atoi(max_move.c_str()) * 20;
      displacement = atoi(max_move.c_str()) * rowHeight / wsite;
      max_disp_const = atoi(max_move.c_str());
    }
    else {
#ifdef DEBUG
      cerr << "read_constraints:: unsupported keyword " << endl;
#endif
    }
  }

  if(max_disp_const == 0.0) max_disp_const = rows.size();

  dot_constraints.close();
  return;
}

void circuit::read_def(const string& input, bool mode) {
  ifstream dot_def(input.c_str());
  if(!dot_def.good()) {
    cerr << "read_def:: cannot open `" << input << "' for reading." << endl;
    exit(1);
  }

  vector< string > tokens(1);
  while(!dot_def.eof()) {
    get_next_token(dot_def, tokens[0], DEFCommentChar);

    if(tokens[0] == DEFLineEndingChar) continue;

    if(tokens[0] == "VERSION") {
      get_next_token(dot_def, tokens[0], DEFCommentChar);
      DEFVersion = tokens[0];
#ifdef DEBUG
      cout << "def version: " << DEFVersion << endl;
#endif
    }
    else if(tokens[0] == "DIVIDERCHAR") {
      get_next_token(dot_def, tokens[0], DEFCommentChar);
      unsigned index1 = tokens[0].find_first_of("\"");
      unsigned index2 = tokens[0].find_last_of("\"");
      assert(index1 != string::npos);
      assert(index2 != string::npos);
      assert(index2 > index1);
      DEFDelimiter = tokens[0].substr(index1 + 1, index2 - index1 - 1);
#ifdef DEBUG
      cout << "divide character: " << DEFDelimiter << endl;
#endif
    }
    else if(tokens[0] == "BUSBITCHARS") {
      get_next_token(dot_def, tokens[0], DEFCommentChar);
      unsigned index1 = tokens[0].find_first_of("\"");
      unsigned index2 = tokens[0].find_last_of("\"");
      assert(index1 != string::npos);
      assert(index2 != string::npos);
      assert(index2 > index1);
      DEFBusCharacters = tokens[0].substr(index1 + 1, index2 - index1 - 1);
#ifdef DEBUG
      cout << "bus bit characters: " << DEFBusCharacters << endl;
#endif
    }
    else if(tokens[0] == "DESIGN") {
      get_next_token(dot_def, tokens[0], DEFCommentChar);
      design_name = tokens[0];
#ifdef DEBUG
      cout << "design name: " << design_name << endl;
#endif
    }
    else if(tokens[0] == "UNITS") {
      get_next_n_tokens(dot_def, tokens, 3, DEFCommentChar);
      assert(tokens.size() == 3);
      assert(tokens[0] == "DISTANCE");
      assert(tokens[1] == "MICRONS");
      DEFdist2Microns = atoi(tokens[2].c_str());
      assert(DEFdist2Microns <= DEFdist2Microns);
#ifdef DEBUG
      cout << "unit distance to microns: " << DEFdist2Microns << endl;
#endif
    }
    else if(tokens[0] == "DIEAREA" && mode == INIT) {
      get_next_n_tokens(dot_def, tokens, 8, DEFCommentChar);
      assert(tokens.size() == 8);
      assert(tokens[0] == "(" && tokens[3] == ")");
      assert(tokens[4] == "(" && tokens[7] == ")");
      lx = atof(tokens[1].c_str());
      by = atof(tokens[2].c_str());
      rx = atof(tokens[5].c_str());
      ty = atof(tokens[6].c_str());
      die.xLL = lx;
      die.xUR = rx;
      die.yLL = by;
      die.yUR = ty;
    }
    else if(tokens[0] == "ROW" && mode == INIT) {
      get_next_n_tokens(dot_def, tokens, 5, DEFCommentChar);

      // if( true ) {
      if(tokens[0][0] != 'h') {
        row* myRow = locateOrCreateRow(tokens[0]);
        myRow->name = tokens[0];
        myRow->site = site2id[tokens[1]];
        myRow->origX = atoi(tokens[2].c_str());
        myRow->origY = atoi(tokens[3].c_str());
        myRow->siteorient = tokens[4];

        if(fabs(rowHeight - 0.0f) <= numeric_limits< double >::epsilon()) {
          rowHeight = sites[myRow->site].height *
                      static_cast< double >(DEFdist2Microns);
        }

        if(wsite == 0) {
          wsite = static_cast< int >(
              sites[myRow->site].width * DEFdist2Microns + 0.5);
        }
        // NOTE: this contest does not allow flipping/rotation
        // assert(myRow->siteorient == "N");
        /*
           if( tokens[4] == "N" )
           myRow->top_power = VDD;
           else if ( tokens[4] == "FS" )
           myRow->top_power = VSS;
           else {
           cerr << "read_def : unsupported site orient !!" << endl;
           }
           */
        get_next_token(dot_def, tokens[0], DEFCommentChar);
        if(tokens[0] == "DO") {
          get_next_n_tokens(dot_def, tokens, 3, DEFCommentChar);
          assert(tokens[1] == "BY");
          myRow->numSites =
              max(atoi(tokens[0].c_str()), atoi(tokens[2].c_str()));
          // NOTE: currenlty we only handle horizontal row sites
          assert(tokens[2] == "1");
          get_next_token(dot_def, tokens[0], DEFCommentChar);
          if(tokens[0] == "STEP") {
            get_next_n_tokens(dot_def, tokens, 2, DEFCommentChar);
            myRow->stepX = atoi(tokens[0].c_str());
            myRow->stepY = atoi(tokens[1].c_str());
            // if( wsite == 0 )
            //    wsite = atoi(tokens[0].c_str());
            // else
            //    assert(wsite == atoi(tokens[0].c_str()) );

            // NOTE: currenlty we only handle horizontal row sites & spacing = 0
            assert(myRow->stepX == sites[myRow->site].width * DEFdist2Microns);
            assert(myRow->stepY == 0);
            // assert(wsite > 0);
          }
        }
      }
      else {
        while(tokens[0] != DEFLineEndingChar)
          get_next_token(dot_def, tokens[0], DEFCommentChar);
      }
      // else we do not currently store properties
    }
    else if(tokens[0] == "TRACKS") {
      track temp;
      get_next_n_tokens(dot_def, tokens, 6, DEFCommentChar);
      assert(tokens[4] == "STEP");
      temp.axis = tokens[0].c_str();
      temp.start = atoi(tokens[1].c_str());
      temp.num_track = atoi(tokens[3].c_str());
      while(tokens[0] == ";") {
        get_next_token(dot_def, tokens[0], DEFCommentChar);
        layer* temp_layer = &layers[layer2id[tokens[0].c_str()]];
        temp.layers.emplace_back(temp_layer);
      }
      tracks.emplace_back(temp);
    }
    else if(tokens[0] == "VIAS") {
      read_def_vias(dot_def);
    }
    else if(tokens[0] == "NONDEFAULTRULES") {  // Shold Make Later
      get_next_token(dot_def, tokens[0], DEFCommentChar);
      while(tokens[0] != "NONDEFAULTRULES")
        get_next_token(dot_def, tokens[0], DEFCommentChar);
    }
    else if(tokens[0] == "REGIONS") {
      if(GROUP_IGNORE == true || mode == FINAL) {
        get_next_token(dot_def, tokens[0], DEFCommentChar);
        while(tokens[0] != "REGIONS")
          get_next_token(dot_def, tokens[0], DEFCommentChar);
      }
      else
        read_def_regions(dot_def);
    }
    else if(tokens[0] == "COMPONENTS") {
      if(mode == INIT)
        read_init_def_components(dot_def);
      else if(mode == FINAL)
        read_final_def_components(dot_def);
    }
    else if(tokens[0] == "PINS" && mode == INIT) {
      read_def_pins(dot_def);
    }
    else if(tokens[0] == "NETS" && mode == INIT)  // Shold Make Later
    {
      read_def_nets(dot_def);
    }
    else if(tokens[0] == "GROUPS") {
      if(GROUP_IGNORE == true) {
        get_next_token(dot_def, tokens[0], DEFCommentChar);
        while(tokens[0] != "GROUPS")
          get_next_token(dot_def, tokens[0], DEFCommentChar);
      }
      else
        read_def_groups(dot_def);
    }
    else if(tokens[0] == "PROPERTYDEFINITIONS") {
      get_next_token(dot_def, tokens[0], DEFCommentChar);
      while(tokens[0] != "PROPERTYDEFINITIONS")
        get_next_token(dot_def, tokens[0], DEFCommentChar);
    }
    else if(tokens[0] == "BLOCKAGES") {
      get_next_token(dot_def, tokens[0], DEFCommentChar);
      while(tokens[0] != "BLOCKAGES")
        get_next_token(dot_def, tokens[0], DEFCommentChar);
    }
    else if(tokens[0] == "SPECIALNETS")  // Shold Make Later !!!! Important
    {
      read_def_special_nets(dot_def);
    }
    else if(tokens[0] == "END") {
      get_next_token(dot_def, tokens[0], DEFCommentChar);
#ifdef DEBUG
      cout << "Next letter of END : " << tokens[0] << endl;
#endif
      assert(tokens[0] == "DESIGN");
      break;
    }
    else {
      get_next_token(dot_def, tokens[0], DEFCommentChar);
    }
  }
#ifdef DEBUG
  cout << "Reading def Done ..." << endl;
#endif
  dot_def.close();
  return;
}

void circuit::read_def_size(const string& input) {
#ifdef DEBUG
  cout << " read def size start " << endl;
#endif
  ifstream dot_size(input.c_str());
  vector< string > tokens(1);

  if(!dot_size.good()) {
    cerr << "read_def_size:: there is no '" << input << "' for reading."
         << endl;
    return;
  }

  while(true) {
    get_next_n_tokens(dot_size, tokens, 3, DEFCommentChar);
    if(dot_size.eof()) break;
    cell* theCell = locateOrCreateCell(tokens[0]);
    theCell->width = static_cast< double >(atof(tokens[1].c_str()) * wsite);
    theCell->height =
        static_cast< double >(atof(tokens[2].c_str()) * rowHeight);

    macro* theMacro = &macros[theCell->type];
    theMacro->width = static_cast< double >(atof(tokens[1].c_str()) *
                                            (double)wsite / DEFdist2Microns);
    theMacro->height = static_cast< double >(
        atof(tokens[2].c_str()) * (double)rowHeight / DEFdist2Microns);

    theMacro->top_power = VDD;

    if(atof(tokens[2].c_str()) > 1) {
      theMacro->isMulti = true;
      // cout << " multi cell height : " << theMacro->height * DEFdist2Microns /
      // rowHeight << endl;
    }
  }
  dot_size.close();
  return;
}

// assumes the COMPONENTS keyword has already been read in
void circuit::read_init_def_components(ifstream& is) {
#ifdef DEBUG
  cout << "read_init_def_component start " << endl;
#endif

  cell* myCell = NULL;
  vector< string > tokens(1);

  get_next_n_tokens(is, tokens, 2, DEFCommentChar);

  cells.reserve(atoi(tokens[0].c_str()));

  assert(tokens[1] == DEFLineEndingChar);

  unsigned countComponents = 0;
  unsigned numComponents = atoi(tokens[0].c_str());

  // can do with while(1)
  while(countComponents <= numComponents) {
    get_next_token(is, tokens[0], DEFCommentChar);
    if(tokens[0] == "-") {
      ++countComponents;
      get_next_n_tokens(is, tokens, 2, DEFCommentChar);
      // assert(cell2id.find(tokens[0]) != cell2id.end());
      if(cell2id.find(tokens[0]) == cell2id.end()) {
        //        cout << "tokens[0]: " << tokens[0] << endl;
        myCell = locateOrCreateCell(tokens[0]);
        myCell->type = macro2id[tokens[1]];
        macro* myMacro = &macros[macro2id[tokens[1]]];
        myCell->width = myMacro->width * static_cast< double >(DEFdist2Microns);
        myCell->height =
            myMacro->height * static_cast< double >(DEFdist2Microns);
      }
      else
        myCell = locateOrCreateCell(tokens[0]);
    }
    else if(tokens[0] == "+") {
      assert(myCell != NULL);
      get_next_token(is, tokens[0], DEFCommentChar);

      if(tokens[0] == "PLACED" || tokens[0] == "FIXED") {
        myCell->isFixed = (tokens[0] == "FIXED");
        // myCell->isPlaced = (tokens[0] == "PLACED");
        get_next_n_tokens(is, tokens, 5, DEFCommentChar);
        assert(tokens[0] == "(");
        assert(tokens[3] == ")");
        myCell->init_x_coord = atoi(tokens[1].c_str());
        myCell->init_y_coord = atoi(tokens[2].c_str());
        if(myCell->isFixed == true) {
          myCell->x_coord = atoi(tokens[1].c_str());
          myCell->y_coord = atoi(tokens[2].c_str());
          myCell->isPlaced = true;
        }
        myCell->cellorient = tokens[4];
      }
    }
    else if(!strcmp(tokens[0].c_str(), DEFLineEndingChar)) {
      myCell = NULL;
    }
    else if(tokens[0] == "END") {
      get_next_token(is, tokens[0], DEFCommentChar);
      assert(tokens[0] == "COMPONENTS");
      break;
    }
  }
#ifdef DEBUG
  cout << "read_init_def_component: done" << endl;
  cout << "tokens[0] : " << tokens[0] << endl;
  cout << "countComponents : " << countComponents << endl;
#endif
  return;
}

// assumes the COMPONENTS keyword has already been read in
void circuit::read_final_def_components(ifstream& is) {
  cell* myCell = NULL;
  vector< string > tokens(1);

  get_next_n_tokens(is, tokens, 2, DEFCommentChar);
  assert(tokens[1] == DEFLineEndingChar);

  unsigned countComponents = 0;
  unsigned numComponents = atoi(tokens[0].c_str());

  // can do with while(1)
  while(countComponents <= numComponents) {
    get_next_token(is, tokens[0], DEFCommentChar);
    if(tokens[0] == "-") {
      ++countComponents;
      get_next_n_tokens(is, tokens, 2, DEFCommentChar);
      assert(cell2id.find(tokens[0]) != cell2id.end());
      myCell = locateOrCreateCell(tokens[0]);
    }
    else if(tokens[0] == "+") {
      assert(myCell != NULL);
      get_next_token(is, tokens[0], DEFCommentChar);

      if(tokens[0] == "PLACED" || tokens[0] == "FIXED") {
        myCell->isFixed = (tokens[0] == "FIXED");
        myCell->isPlaced = (tokens[0] == "PLACED");
        get_next_n_tokens(is, tokens, 5, DEFCommentChar);
        assert(tokens[0] == "(");
        assert(tokens[3] == ")");
        myCell->x_coord = atoi(tokens[1].c_str());
        myCell->y_coord = atoi(tokens[2].c_str());
        myCell->x_pos = myCell->x_coord / wsite;
        myCell->y_pos = myCell->y_coord / rowHeight;
        myCell->cellorient = tokens[4];
        // NOTE: this contest does not allow flipping/rotation
        // assert(myCell->cellorient == "N");
      }
    }
    else if(!strcmp(tokens[0].c_str(), DEFLineEndingChar)) {
      myCell = NULL;
    }
    else if(tokens[0] == "END") {
      get_next_token(is, tokens[0], DEFCommentChar);
      assert(tokens[0] == "COMPONENTS");
      break;
    }
  }
  return;
}

// assumes the PINS keyword has already been read in
// we already read pins from .verilog,
// thus this update locations / performs sanity checks
void circuit::read_def_pins(ifstream& is) {
  pin* myPin = NULL;
  vector< string > tokens(1);

  get_next_n_tokens(is, tokens, 2, DEFCommentChar);

  pins.reserve(atoi(tokens[0].c_str()));

  assert(tokens[1] == DEFLineEndingChar);

  unsigned countPins = 0;
  unsigned numPins = atoi(tokens[0].c_str());

  while(countPins <= numPins) {
    get_next_token(is, tokens[0], DEFCommentChar);
    if(tokens[0] == "-") {
      ++countPins;
      get_next_token(is, tokens[0], DEFCommentChar);
      // assert(pin2id.find(tokens[0]) != pin2id.end());
      myPin = locateOrCreatePin(tokens[0]);
      // NOTE: pins in .def are only for PI/POs that are fixed/placed
      // assert(myPin->type == PI_PIN || myPin->type == PO_PIN);
    }
    else if(tokens[0] == "+") {
      assert(myPin != NULL);
      get_next_token(is, tokens[0], DEFCommentChar);

      // NOTE: currently, we just store NET, DIRECTION, LAYER, FIXED/PLACED
      if(tokens[0] == "NET")  // Shold Make later
      {
        get_next_token(is, tokens[0], DEFCommentChar);
        // assert(net2id.find(tokens[0]) != net2id.end());
      }
      else if(tokens[0] == "DIRECTION") {
        assert(myPin != NULL);
        get_next_token(is, tokens[0], DEFCommentChar);
        if(myPin->type == PI_PIN)
          assert(tokens[0] == "INPUT");
        else if(myPin->type == PO_PIN)
          assert(tokens[0] == "OUTPUT");
      }
      else if(tokens[0] == "FIXED" || tokens[0] == "PLACED") {
        assert(myPin != NULL);
        myPin->isFixed = (tokens[0] == "FIXED");
        get_next_n_tokens(is, tokens, 5, DEFCommentChar);
        assert(tokens[0] == "(");
        assert(tokens[3] == ")");
        myPin->x_coord = atof(tokens[1].c_str());
        myPin->y_coord = atof(tokens[2].c_str());
        // NOTE: this contest does not allow flipping/rotation
        assert(tokens[4] == "N");
      }
      else if(tokens[0] == "LAYER") {
        assert(myPin != NULL);
        get_next_token(is, tokens[0], DEFCommentChar);
        // NOTE: we assume the layer is previously defined from .lef
        // we don't save layer for a pin instance.
        assert(layer2id.find(tokens[0]) != layer2id.end());
        get_next_n_tokens(is, tokens, 8, DEFCommentChar);
        assert(tokens[0] == "(");
        assert(tokens[3] == ")");
        assert(tokens[4] == "(");
        assert(tokens[7] == ")");
        myPin->x_coord +=
            0.5 * (atof(tokens[1].c_str()) + atof(tokens[5].c_str()));
        myPin->y_coord +=
            0.5 * (atof(tokens[2].c_str()) + atof(tokens[6].c_str()));
      }
      else if(!strcmp(tokens[0].c_str(), DEFLineEndingChar)) {
        myPin = NULL;
      }
    }
    else if(tokens[0] == "END") {
      get_next_token(is, tokens[0], DEFCommentChar);
      assert(tokens[0] == "PINS");
      break;
    }
  }
  return;
}

void circuit::read_def_special_nets(ifstream& is) {
  vector< string > tokens(1);
  get_next_n_tokens(is, tokens, 2, DEFCommentChar);
  assert(tokens[1] == DEFLineEndingChar);

  unsigned countNets = 0;
  unsigned numNets = atoi(tokens[0].c_str());

  while(countNets <= numNets) {
    get_next_n_tokens(is, tokens, 2, DEFCommentChar);
    if(tokens[0] == "END") return;

    assert(tokens[0] == "-");
    if(tokens[1] == "vdd") {
      get_next_token(is, tokens[0], DEFCommentChar);
      while(tokens[0] != DEFLineEndingChar) {
        if(tokens[0] == "metal1") {
          get_next_n_tokens(is, tokens, 9, DEFCommentChar);
          if(tokens[8] == "(") {
            // cout << tokens[6] << endl;
            if(atoi(tokens[6].c_str()) == 6000) {
              // cout << " power found !!!!! " << endl;
              initial_power = VDD;
            }
            else if(atoi(tokens[6].c_str()) == 8000) {
              initial_power = VSS;
              // cout << " power found !!!!! " << endl;
            }
          }
        }

        get_next_token(is, tokens[0], DEFCommentChar);
      }
    }
    else if(tokens[1] == "vss") {
      get_next_token(is, tokens[0], DEFCommentChar);
      while(tokens[0] != DEFLineEndingChar)
        get_next_token(is, tokens[0], DEFCommentChar);
    }
    else {
      cout << "tokens[1] == " << tokens[1] << endl;
      cerr << "circuit::read_def_spacial_nets ==> invalid special net. ( vdd / "
              "vss only ) "
           << endl;
      return;
    }
    countNets++;
  }

  return;
}

// assumes the NETS keyword has already been read in
// we already read nets from .verilog,
// thus this only performs sanity checks
void circuit::read_def_nets(ifstream& is) {
#ifdef DEBUG
  cout << " read_def_nets:: begin" << endl;
#endif

  net* myNet = NULL;
  pin* myPin = NULL;
  vector< string > tokens(1);

  get_next_n_tokens(is, tokens, 2, DEFCommentChar);

  nets.reserve(atoi(tokens[0].c_str()));

  assert(tokens[1] == DEFLineEndingChar);

  unsigned countNets = 0;
  unsigned numNets = atoi(tokens[0].c_str());

  while(countNets <= numNets) {
    get_next_token(is, tokens[0], DEFCommentChar);
    if(tokens[0] == "-") {
      get_next_token(is, tokens[0], DEFCommentChar);

      // Shold make later --> net , pin should build on def
      // assert(net2id.find(tokens[0]) != net2id.end());
      myNet = locateOrCreateNet(tokens[0]);
      unsigned myNetId = net2id.find(myNet->name)->second;

#ifdef DEBUG
      cout << myNet->name << endl;
#endif
      // first is always source, rest are sinks
      get_next_n_tokens(is, tokens, 4, DEFCommentChar);
      assert(tokens[0] == "(");
      assert(tokens[3] == ")");
      // ( PIN PI/PO ) or ( cell_instance internal_pin )
      string pinName =
          tokens[1] == "PIN" ? tokens[2] : tokens[1] + ":" + tokens[2];
      // assert(pin2id.find(pinName) != pin2id.end());
      myPin = locateOrCreatePin(pinName);
      myPin->net = myNetId;
      myNet->source = myPin->id;
      if(tokens[1] != "PIN") {
        myPin->owner = cell2id[tokens[1]];

#ifdef DEBUG
        cout << "owner name : " << cells[myPin->owner].name << endl;
        cout << "mypin name : " << myPin->name << endl;
#endif
        myPin->type = NONPIO_PIN;
        macro* theMacro = &macros[cells[myPin->owner].type];
        macro_pin* myMacroPin = &theMacro->pins[tokens[2]];
        myPin->x_offset =
            myMacroPin->port[0].xLL / 2 + myMacroPin->port[0].xUR / 2;
        myPin->y_offset =
            myMacroPin->port[0].yLL / 2 + myMacroPin->port[0].yUR / 2;
      }
      // assert(myPin->net == myNetId);

      do {
        get_next_token(is, tokens[0], DEFCommentChar);
        if(tokens[0] == DEFLineEndingChar) break;

        if(tokens[0] == "+") break;

        assert(tokens[0] == "(");
        get_next_n_tokens(is, tokens, 3, DEFCommentChar);
        assert(tokens.size() == 3);
        assert(tokens[2] == ")");
        if(tokens[2] == DEFLineEndingChar) break;

        pinName = tokens[0] == "PIN" ? tokens[1] : tokens[0] + ":" + tokens[1];
        // assert(pin2id.find(pinName) != pin2id.end());
        myPin = locateOrCreatePin(pinName);
        myPin->net = myNetId;

        if(tokens[0] != "PIN") {
          myPin->owner = cell2id[tokens[0]];
#ifdef DEBUG
          cout << "owner name : " << cells[myPin->owner].name << endl;
          cout << "mypin name : " << myPin->name << endl;
#endif
          myPin->type = NONPIO_PIN;
          macro* theMacro = &macros[cells[myPin->owner].type];
          macro_pin* myMacroPin = &theMacro->pins[tokens[1]];
          myPin->x_offset =
              myMacroPin->port[0].xLL / 2 + myMacroPin->port[0].xUR / 2;
          myPin->y_offset =
              myMacroPin->port[0].yLL / 2 + myMacroPin->port[0].yUR / 2;
        }
        myNet->sinks.emplace_back(myPin->id);

        assert(myPin->net == myNetId);
      } while(tokens[2] == ")");
    }
    else if(!strcmp(tokens[0].c_str(), DEFLineEndingChar)) {
      myNet = NULL;
      myPin = NULL;
    }
    else if(tokens[0] == "END") {
      get_next_token(is, tokens[0], DEFCommentChar);
      assert(tokens[0] == "NETS");
      break;
    }
  }
#ifdef DEBUG
  cout << " read_def_nets:: end" << endl;
#endif
  return;
}

void circuit::read_def_regions(ifstream& is) {
#ifdef DEBUG
  cout << "start read def regions" << endl;
#endif
  vector< string > tokens(1);
  get_next_n_tokens(is, tokens, 2, DEFCommentChar);
  assert(tokens[1] == DEFLineEndingChar);

  unsigned countRegions = 0;
  unsigned numRegions = atoi(tokens[0].c_str());

  while(tokens[0] != "END") {
    get_next_n_tokens(is, tokens, 2, DEFCommentChar);
    if(tokens[0] == "-") {
      countRegions++;
      group* myGroup = locateOrCreateGroup(tokens[1].c_str());
      get_next_token(is, tokens[0], DEFCommentChar);
      while(tokens[0] != "+") {
        get_next_n_tokens(is, tokens, 7, DEFCommentChar);
        rect myRect;
        myRect.xLL = max(lx, atof(tokens[0].c_str()));
        myRect.yLL = max(by, atof(tokens[1].c_str()));
        myRect.xUR = min(rx, atof(tokens[4].c_str()));
        myRect.yUR = min(ty, atof(tokens[5].c_str()));

        myGroup->boundary.xLL =
            max(lx, min(myGroup->boundary.xLL, atof(tokens[0].c_str())));
        myGroup->boundary.yLL =
            max(by, min(myGroup->boundary.yLL, atof(tokens[1].c_str())));
        myGroup->boundary.xUR =
            min(rx, max(myGroup->boundary.xUR, atof(tokens[4].c_str())));
        myGroup->boundary.yUR =
            min(ty, max(myGroup->boundary.yUR, atof(tokens[5].c_str())));
        myGroup->regions.emplace_back(myRect);

        get_next_token(is, tokens[0], DEFCommentChar);
      }
      if(tokens[0] == "+") {
        get_next_n_tokens(is, tokens, 3, DEFCommentChar);
        assert(tokens[0] == "TYPE");
        myGroup->type = tokens[0].c_str();
        assert(tokens[2] == DEFLineEndingChar);
      }
      // group2id.insert(make_pair(myGroup.name,groups.size()));
      // groups.emplace_back(myGroup);
    }
  }
  assert(countRegions == numRegions);
  assert(tokens[0] == "END");
  assert(tokens[1] == "REGIONS");

#ifdef DEBUG
  cout << "End read def regions" << endl;
#endif
  return;
}

void circuit::read_def_groups(ifstream& is) {
#ifdef DEBUG
  cout << "start read def groups" << endl;
#endif
  vector< string > tokens(1);
  get_next_n_tokens(is, tokens, 2, DEFCommentChar);
  assert(tokens[1] == DEFLineEndingChar);

  unsigned numGroups = atoi(tokens[0].c_str());

  while(tokens[0] != "END") {
    get_next_n_tokens(is, tokens, 2, DEFCommentChar);
    if(tokens[0] == "-") {
      group* myGroup = locateOrCreateGroup(tokens[1].c_str());
      get_next_token(is, tokens[0], DEFCommentChar);
      while(tokens[0] != "+") {
        myGroup->tag = tokens[0].c_str();
        for(int i = 0; i < cells.size(); i++) {
          cell* theCell = &cells[i];
          if(strncmp(myGroup->tag.c_str(), theCell->name.c_str(),
                     myGroup->tag.size() - 1) == 0) {
            myGroup->siblings.emplace_back(theCell);
            theCell->group = myGroup->name;
            theCell->inGroup = true;
          }
        }
        get_next_token(is, tokens[0], DEFCommentChar);
      }
      get_next_n_tokens(is, tokens, 3, DEFCommentChar);
      assert(tokens[2] == DEFLineEndingChar);
      assert(tokens[0] == "REGION");
    }
    else if(tokens[0] == "END") {
      assert(tokens[1] == "GROUPS");
      break;
    }
    else {
      cerr << "read_def_groups : unsupported keyword !! " << endl;
      exit(2);
    }
  }
#ifdef DEBUG
  cout << "end read def groups" << endl;
#endif
  return;
}

void circuit::read_lef(const string& input) {
  //	cout << "  .lef file       : "<< input <<endl;
  ifstream dot_lef(input.c_str());
  if(!dot_lef.good()) {
    cerr << "read_cell_lef:: cannot open `" << input << "' for reading."
         << endl;
    exit(1);
  }

  vector< string > tokens(1);

  while(!dot_lef.eof()) {
    get_next_token(dot_lef, tokens[0], LEFCommentChar);
    if(tokens[0] == LEFLineEndingChar) continue;

    if(tokens[0] == "VERSION") {
      get_next_token(dot_lef, tokens[0], LEFCommentChar);
      LEFVersion = tokens[0];
#ifdef DEBUG
      cout << "lef version: " << LEFVersion << endl;
#endif
    }
    else if(tokens[0] == "NAMECASESENSITIVE") {
      get_next_token(dot_lef, tokens[0], LEFCommentChar);
      LEFNamesCaseSensitive = tokens[0];
#ifdef DEBUG
      cout << "names case sensitive: " << LEFNamesCaseSensitive << endl;
#endif
    }
    else if(tokens[0] == "BUSBITCHARS") {
      get_next_token(dot_lef, tokens[0], LEFCommentChar);
      unsigned index1 = tokens[0].find_first_of("\"");
      unsigned index2 = tokens[0].find_last_of("\"");
      assert(index1 != string::npos);
      assert(index2 != string::npos);
      assert(index2 > index1);
      LEFBusCharacters = tokens[0].substr(index1 + 1, index2 - index1 - 1);
#ifdef DEBUG
      cout << "bus bit characters: " << LEFBusCharacters << endl;
#endif
    }
    else if(tokens[0] == "DIVIDERCHAR") {
      get_next_token(dot_lef, tokens[0], LEFCommentChar);
      unsigned index1 = tokens[0].find_first_of("\"");
      unsigned index2 = tokens[0].find_last_of("\"");
      assert(index1 != string::npos);
      assert(index2 != string::npos);
      assert(index2 > index1);
      LEFDelimiter = tokens[0].substr(index1 + 1, index2 - index1 - 1);
#ifdef DEBUG
      cout << "divide character: " << LEFDelimiter << endl;
#endif
    }
    else if(tokens[0] == "UNITS") {
      get_next_n_tokens(dot_lef, tokens, 3, LEFCommentChar);
      assert(tokens.size() == 3);
      assert(tokens[0] == "DATABASE");
      assert(tokens[1] == "MICRONS");
      DEFdist2Microns = atoi(tokens[2].c_str());
#ifdef DEBUG
      cout << "unit distance to microns: " << DEFdist2Microns << endl;
#endif
      get_next_n_tokens(dot_lef, tokens, 3, LEFCommentChar);
      assert(tokens[0] == LEFLineEndingChar);
      assert(tokens[1] == "END");
      assert(tokens[2] == "UNITS");
    }
    else if(tokens[0] == "MANUFACTURINGGRID") {
      get_next_token(dot_lef, tokens[0], LEFLineEndingChar);
      LEFManufacturingGrid = atof(tokens[0].c_str());
#ifdef DEBUG
      cout << "manufacturing grid: " << LEFManufacturingGrid << endl;
#endif
    }
    else if(tokens[0] == "SITE")
      read_lef_site(dot_lef);
    else if(tokens[0] == "LAYER")
      read_lef_layer(dot_lef);
    else if(tokens[0] == "VIA")
      read_lef_via(dot_lef);
    else if(tokens[0] == "VIARULE")
      read_lef_viaRule(dot_lef);
    else if(tokens[0] == "MAXVIASTACK") {
      get_next_n_tokens(dot_lef, tokens, 5, LEFCommentChar);
      MAXVIASTACK = atoi(tokens[0].c_str());
      minLayer = &layers[layer2id[tokens[2].c_str()]];
      maxLayer = &layers[layer2id[tokens[3].c_str()]];
      assert(tokens[4] == LEFLineEndingChar);
    }
    else if(tokens[0] == "PROPERTYDEFINITIONS")
      read_lef_property(dot_lef);
    else if(tokens[0] == "MACRO")
      read_lef_macro(dot_lef);
    else if(tokens[0] == "END") {
      get_next_token(dot_lef, tokens[0], LEFCommentChar);
      assert(tokens[0] == "LIBRARY");
      break;
    }
  }
  dot_lef.close();
  rowHeight = sites[0].height * static_cast< double >(DEFdist2Microns);
  wsite = static_cast< int >(sites[0].width * DEFdist2Microns + 0.5);
  assert(rowHeight != 0);
  assert(wsite != 0);
#ifdef DEBUG
  cout << "Reading lef done ... " << endl;
#endif
  return;
}

void circuit::read_cell_lef(const string& input) {
  //	cout << "  .lef file       : "<< input <<endl;
  ifstream dot_lef(input.c_str());
  if(!dot_lef.good()) {
    cerr << "read_cell_lef:: cannot open `" << input << "' for reading."
         << endl;
    exit(1);
  }

  vector< string > tokens(1);

  while(!dot_lef.eof()) {
    get_next_token(dot_lef, tokens[0], LEFCommentChar);
    if(tokens[0] == LEFLineEndingChar) continue;

    if(tokens[0] == "VERSION") {
      get_next_token(dot_lef, tokens[0], LEFCommentChar);
      LEFVersion = tokens[0];
#ifdef DEBUG
      cout << "lef version: " << LEFVersion << endl;
#endif
    }
    else if(tokens[0] == "BUSBITCHARS") {
      get_next_token(dot_lef, tokens[0], LEFCommentChar);
      unsigned index1 = tokens[0].find_first_of("\"");
      unsigned index2 = tokens[0].find_last_of("\"");
      assert(index1 != string::npos);
      assert(index2 != string::npos);
      assert(index2 > index1);
      LEFBusCharacters = tokens[0].substr(index1 + 1, index2 - index1 - 1);
#ifdef DEBUG
      cout << "bus bit characters: " << LEFBusCharacters << endl;
#endif
    }
    else if(tokens[0] == "DIVIDERCHAR") {
      get_next_token(dot_lef, tokens[0], LEFCommentChar);
      unsigned index1 = tokens[0].find_first_of("\"");
      unsigned index2 = tokens[0].find_last_of("\"");
      assert(index1 != string::npos);
      assert(index2 != string::npos);
      assert(index2 > index1);
      LEFDelimiter = tokens[0].substr(index1 + 1, index2 - index1 - 1);
#ifdef DEBUG
      cout << "divide character: " << LEFDelimiter << endl;
#endif
    }
    else if(tokens[0] == "UNITS") {
      get_next_n_tokens(dot_lef, tokens, 3, LEFCommentChar);
      assert(tokens.size() == 3);
      assert(tokens[0] == "DATABASE");
      assert(tokens[1] == "MICRONS");
      DEFdist2Microns = atoi(tokens[2].c_str());
#ifdef DEBUG
      cout << "unit distance to microns: " << DEFdist2Microns << endl;
#endif
      get_next_n_tokens(dot_lef, tokens, 3, LEFCommentChar);
      assert(tokens[0] == LEFLineEndingChar);
      assert(tokens[1] == "END");
      assert(tokens[2] == "UNITS");
    }
    else if(tokens[0] == "MACRO") {
      read_lef_macro(dot_lef);
    }
    else if(tokens[0] == "END") {
      get_next_token(dot_lef, tokens[0], LEFCommentChar);
      assert(tokens[0] == "LIBRARY");
      break;
    }
  }
  dot_lef.close();
#ifdef DEBUG
  cout << "Reading cell lef done ... " << endl;
#endif
  return;
}

void circuit::read_tech_lef(const string& input) {
  //	cout << "  .lef file       : "<< input <<endl;
  ifstream dot_lef(input.c_str());
  if(!dot_lef.good()) {
    cerr << "read_tech_lef:: cannot open `" << input << "' for reading."
         << endl;
    exit(1);
  }

  vector< string > tokens(1);
  while(!dot_lef.eof()) {
    get_next_token(dot_lef, tokens[0], LEFCommentChar);

    if(tokens[0] == LEFLineEndingChar) continue;

    if(tokens[0] == "VERSION") {
      get_next_token(dot_lef, tokens[0], LEFCommentChar);
      LEFVersion = tokens[0];
#ifdef DEBUG
      cout << "lef version: " << LEFVersion << endl;
#endif
    }
    else if(tokens[0] == "NAMECASESENSITIVE") {
      get_next_token(dot_lef, tokens[0], LEFCommentChar);
      LEFNamesCaseSensitive = tokens[0];
#ifdef DEBUG
      cout << "names case sensitive: " << LEFNamesCaseSensitive << endl;
#endif
    }
    else if(tokens[0] == "BUSBITCHARS") {
      get_next_token(dot_lef, tokens[0], LEFCommentChar);
      unsigned index1 = tokens[0].find_first_of("\"");
      unsigned index2 = tokens[0].find_last_of("\"");
      assert(index1 != string::npos);
      assert(index2 != string::npos);
      assert(index2 > index1);
      LEFBusCharacters = tokens[0].substr(index1 + 1, index2 - index1 - 1);
#ifdef DEBUG
      cout << "bus bit characters: " << LEFBusCharacters << endl;
#endif
    }
    else if(tokens[0] == "DIVIDERCHAR") {
      get_next_token(dot_lef, tokens[0], LEFCommentChar);
      unsigned index1 = tokens[0].find_first_of("\"");
      unsigned index2 = tokens[0].find_last_of("\"");
      assert(index1 != string::npos);
      assert(index2 != string::npos);
      assert(index2 > index1);
      LEFDelimiter = tokens[0].substr(index1 + 1, index2 - index1 - 1);
#ifdef DEBUG
      cout << "divide character: " << LEFDelimiter << endl;
#endif
    }
    else if(tokens[0] == "UNITS") {
      get_next_n_tokens(dot_lef, tokens, 3, LEFCommentChar);
      assert(tokens.size() == 3);
      assert(tokens[0] == "DATABASE");
      assert(tokens[1] == "MICRONS");
      DEFdist2Microns = atoi(tokens[2].c_str());
#ifdef DEBUG
      cout << "unit distance to microns: " << DEFdist2Microns << endl;
#endif
      get_next_n_tokens(dot_lef, tokens, 3, LEFCommentChar);
      assert(tokens[0] == LEFLineEndingChar);
      assert(tokens[1] == "END");
      assert(tokens[2] == "UNITS");
    }
    else if(tokens[0] == "MANUFACTURINGGRID") {
      get_next_token(dot_lef, tokens[0], LEFLineEndingChar);
      LEFManufacturingGrid = atof(tokens[0].c_str());
#ifdef DEBUG
      cout << "manufacturing grid: " << LEFManufacturingGrid << endl;
#endif
    }
    else if(tokens[0] == "SITE")
      read_lef_site(dot_lef);
    else if(tokens[0] == "LAYER")
      read_lef_layer(dot_lef);
    else if(tokens[0] == "VIA")
      read_lef_via(dot_lef);
    else if(tokens[0] == "VIARULE")
      read_lef_viaRule(dot_lef);
    else if(tokens[0] == "MAXVIASTACK") {
      get_next_n_tokens(dot_lef, tokens, 5, LEFCommentChar);
      MAXVIASTACK = atoi(tokens[0].c_str());
      minLayer = &layers[layer2id[tokens[2].c_str()]];
      maxLayer = &layers[layer2id[tokens[3].c_str()]];
      assert(tokens[4] == LEFLineEndingChar);
    }
    else if(tokens[0] == "PROPERTYDEFINITIONS")
      read_lef_property(dot_lef);
    else if(tokens[0] == "MACRO")
      read_lef_macro(dot_lef);
    else if(tokens[0] == "END") {
      get_next_token(dot_lef, tokens[0], LEFCommentChar);
      assert(tokens[0] == "LIBRARY");
      break;
    }
  }
  dot_lef.close();
#ifdef DEBUG
  cout << "Reading tech lef done ... " << endl;
#endif
  // after read_lef_site
  return;
}

// assumes the SITE keyword has already been read in
void circuit::read_lef_site(ifstream& is) {
#ifdef DEBUG
  cerr << "read_lef_site:: begin\n" << endl;
#endif
  site* mySite = NULL;
  vector< string > tokens(1);

  get_next_token(is, tokens[0], LEFCommentChar);
  mySite = locateOrCreateSite(tokens[0]);

  get_next_token(is, tokens[0], LEFCommentChar);
  while(tokens[0] != "END") {
    if(tokens[0] == "SIZE") {
      assert(mySite != NULL);
      get_next_n_tokens(is, tokens, 4, LEFCommentChar);
      assert(tokens[1] == "BY");
      assert(tokens[3] == LEFLineEndingChar);
      mySite->width = atof(tokens[0].c_str());
      mySite->height = atof(tokens[2].c_str());
    }
    else if(tokens[0] == "CLASS") {
      get_next_n_tokens(is, tokens, 2, LEFCommentChar);
      assert(tokens[1] == LEFLineEndingChar);
      mySite->type = tokens[0];
    }
    else if(tokens[0] == "SYMMETRY") {
      // NOTE: this contest does not allow flipping/rotation
      // even though symmetries are specified for a site
      get_next_token(is, tokens[0], LEFCommentChar);
      while(tokens[0] != LEFLineEndingChar) {
        mySite->symmetries.emplace_back(tokens[0]);
        get_next_token(is, tokens[0], LEFCommentChar);
      }
      assert(tokens[0] == LEFLineEndingChar);
    }
    else {
#ifdef DEBUG
      cout << "read_lef_site::unsupported keyword " << tokens[0] << endl;
#endif
    }
    get_next_token(is, tokens[0], LEFCommentChar);
  }
  get_next_token(is, tokens[0], LEFCommentChar);
  return;
}

// assumes the PROPERTYDIFINITIONS keyword has already been read in
void circuit::read_lef_property(ifstream& is) {
  vector< string > tokens(1);

  get_next_token(is, tokens[0], LEFCommentChar);
  while(tokens[0] != "END") {
    get_next_token(is, tokens[0], LEFCommentChar);
  }
  // Edge Spacing Hard Mapping -- by SGD
  edge_spacing[make_pair(1, 1)] = 0.4 * DEFdist2Microns;
  edge_spacing[make_pair(1, 2)] = 0.4 * DEFdist2Microns;
  edge_spacing[make_pair(2, 2)] = 0 * DEFdist2Microns;

  get_next_token(is, tokens[0], LEFCommentChar);
  return;
}

// assumes the LAYER keyword has already been read in
void circuit::read_lef_layer(ifstream& is) {
  layer* myLayer;
  vector< string > tokens(1);

  get_next_token(is, tokens[0], LEFCommentChar);
  myLayer = locateOrCreateLayer(tokens[0]);

  get_next_token(is, tokens[0], LEFCommentChar);
  while(tokens[0] != "END") {
    if(tokens[0] == "TYPE") {
      get_next_n_tokens(is, tokens, 2, LEFCommentChar);
      assert(tokens[1] == LEFLineEndingChar);
      myLayer->type = tokens[0];
    }
    else if(tokens[0] == "DIRECTION") {
      get_next_n_tokens(is, tokens, 2, LEFCommentChar);
      assert(tokens[1] == LEFLineEndingChar);
      myLayer->direction = tokens[0];
    }
    else if(tokens[0] == "PITCH") {
      get_next_n_tokens(is, tokens, 2, LEFCommentChar);
      myLayer->xPitch = atof(tokens[0].c_str());
      if(tokens[1] == LEFLineEndingChar)
        myLayer->yPitch = myLayer->xPitch;
      else {
        myLayer->yPitch = atof(tokens[1].c_str());
        get_next_token(is, tokens[0], LEFCommentChar);
        assert(tokens[0] == LEFLineEndingChar);
      }
    }
    else if(tokens[0] == "PROPERTY") {
      get_next_token(is, tokens[0], LEFCommentChar);
      if(tokens[0] == "LEF57_MINSTEP") {
        get_next_token(is, tokens[0], LEFCommentChar);
        while(tokens[0] != ";") {
          myLayer->minStep = myLayer->minStep + tokens[0].c_str() + ' ';
          get_next_token(is, tokens[0], LEFCommentChar);
        }
#ifdef DEBUG2
        cout << "minStep : " << myLayer->minStep << endl;
#endif
      }
      else if(tokens[0] == "LEF57_SPACING") {
        get_next_token(is, tokens[0], LEFCommentChar);
        while(tokens[0] != ";") {
          myLayer->spacing = myLayer->spacing + tokens[0].c_str() + ' ';
          get_next_token(is, tokens[0], LEFCommentChar);
        }
#ifdef DEBUG2
        cout << "spacing : " << myLayer->spacing << endl;
#endif
      }
      else {
#ifdef DEBUG
        cerr << "read_lef_layer::PROPERTY unsupported property " << tokens[0]
             << endl;
#endif
      }
    }
    else if(tokens[0] == "OFFSET") {
      get_next_n_tokens(is, tokens, 2, LEFCommentChar);
      myLayer->xOffset = atof(tokens[0].c_str());
      if(tokens[1] == LEFLineEndingChar)
        myLayer->yOffset = myLayer->xOffset;
      else {
        myLayer->yOffset = atof(tokens[1].c_str());
        get_next_token(is, tokens[0], LEFCommentChar);
        assert(tokens[0] == LEFLineEndingChar);
      }
    }
    else if(tokens[0] == "WIDTH") {
      get_next_n_tokens(is, tokens, 2, LEFCommentChar);
      assert(tokens[1] == LEFLineEndingChar);
      myLayer->width = atof(tokens[0].c_str());
    }
    else if(tokens[0] == "MAXWIDTH") {
      get_next_n_tokens(is, tokens, 2, LEFCommentChar);
      assert(tokens[1] == LEFLineEndingChar);
      myLayer->maxWidth = atof(tokens[0].c_str());
    }
    else if(tokens[0] == "SPACINGTABLE") {  // SKIP ( not saved )
      get_next_token(is, tokens[0], LEFCommentChar);
      while(tokens[0] != ";") {
        get_next_token(is, tokens[0], LEFCommentChar);
      }
    }
    else if(tokens[0] == "AREA") {
      get_next_n_tokens(is, tokens, 2, LEFCommentChar);
      assert(tokens[1] == LEFLineEndingChar);
      myLayer->area = atof(tokens[0].c_str());
    }
    else if(tokens[0] == "MINENCLOSEDAREA") {
      get_next_n_tokens(is, tokens, 2, LEFCommentChar);
      assert(tokens[1] == LEFLineEndingChar);
      myLayer->minEnclosedArea = atof(tokens[0].c_str());
    }
    else if(tokens[0] == "MINIMUMCUT") {
      get_next_n_tokens(is, tokens, 4, LEFCommentChar);
      mincut temp;
      temp.via_num = atof(tokens[0].c_str());
      temp.width = atof(tokens[2].c_str());
      if(tokens[3] == "LENGTH") {
        get_next_n_tokens(is, tokens, 4, LEFCommentChar);
        temp.length = atof(tokens[0].c_str());
        if(tokens[1] == "WITHIN") temp.within = atof(tokens[2].c_str());
        assert(tokens[3] == LEFLineEndingChar);
      }
      else if(tokens[3] == "FROMABOVE") {
        get_next_token(is, tokens[0], LEFCommentChar);
        // assert(tokens[0] == LEFLineEndingChar);
      }
      else if(tokens[3] == "FROMBELOW") {
        get_next_token(is, tokens[0], LEFCommentChar);
        // assert(tokens[0] == LEFLineEndingChar);
      }
      myLayer->mincut_rule.emplace_back(temp);
    }
    else if(tokens[0] == "ACCURRENTDENSITY") {
      while(tokens[0] != "TABLEENTRIES")
        get_next_token(is, tokens[0], LEFCommentChar);

      while(tokens[0] != LEFLineEndingChar)
        get_next_token(is, tokens[0], LEFCommentChar);
    }
    else if(tokens[0] == "SPACING") {
      get_next_n_tokens(is, tokens, 2, LEFCommentChar);
      space temp;
      temp.min = atof(tokens[0].c_str());
      if(tokens[1] == "ADJACENTCUTS") {
        get_next_n_tokens(is, tokens, 4, LEFCommentChar);
        assert(tokens[3] == LEFLineEndingChar);
        temp.adj = atoi(tokens[0].c_str());
        temp.max = atof(tokens[2].c_str());
        temp.type = "ADJACENTCUTS";
      }
      else if(tokens[1] == "CENTERTOCENTER") {
        get_next_token(is, tokens[0], LEFCommentChar);
        temp.type = "CENTERTOCENTER";
        assert(tokens[0] == LEFLineEndingChar);
      }
      else if(tokens[1] == "SAMENET") {
        get_next_token(is, tokens[0], LEFCommentChar);
        temp.type = "SAMENET";
        assert(tokens[0] == LEFLineEndingChar);
      }
      myLayer->spacing_rule.emplace_back(temp);
    }
    else if(tokens[0] == "ENCLOSURE") {  // SKIP ( not saved )
      get_next_n_tokens(is, tokens, 4, LEFCommentChar);
      assert(tokens[3] == LEFLineEndingChar);
    }
    else {
      while(tokens[0] != LEFLineEndingChar)
        get_next_token(is, tokens[0], LEFCommentChar);
#ifdef DEBUG
      cerr << "read_lef_layer:: unsupported keyword " << tokens[0] << endl;
#endif
    }
    get_next_token(is, tokens[0], LEFCommentChar);
  }

  get_next_token(is, tokens[0], LEFCommentChar);
  assert(myLayer->name == tokens[0]);
  return;
}

void circuit::read_def_vias(ifstream& is) {
  via* myVia;
  vector< string > tokens(1);
  get_next_n_tokens(is, tokens, 2, LEFCommentChar);
  assert(tokens[1] == DEFLineEndingChar);

  bool pass = false;

  unsigned countVias = 0;
  unsigned numVias = atoi(tokens[0].c_str());

  while(countVias <= numVias) {
    get_next_token(is, tokens[0], DEFCommentChar);
    if(tokens[0] == "-") {
      pass = false;
      get_next_token(is, tokens[0], DEFCommentChar);
      OPENDP_HASH_MAP< string, unsigned >::iterator it =
          via2id.find(tokens[0].c_str());
      if(it == via2id.end()) {
        myVia = locateOrCreateVia(tokens[0]);
      }
      else
        pass = true;
      //            countVias++;
    }
    else if(tokens[0] == "+") {
      if(pass == false) {
        get_next_token(is, tokens[0], DEFCommentChar);
        assert(tokens[0] == "RECT");
        get_next_n_tokens(is, tokens, 9, DEFCommentChar);
        assert(tokens[1] == "(");
        assert(tokens[4] == ")");
        rect theRect;
        theRect.xLL = atof(tokens[2].c_str());
        theRect.yLL = atof(tokens[3].c_str());
        theRect.xUR = atof(tokens[6].c_str());
        theRect.yUR = atof(tokens[7].c_str());
        layer* myLayer = &layers[layer2id[tokens[0].c_str()]];
        myVia->obses.emplace_back(make_pair(myLayer, theRect));
      }
    }
    else if(!strcmp(tokens[0].c_str(), DEFLineEndingChar)) {
      myVia = NULL;
    }
    else if(tokens[0] == "END") {
      get_next_token(is, tokens[0], DEFCommentChar);
      assert(tokens[0] == "VIAS");
      break;
    }
  }
  return;
}

void circuit::read_lef_via(ifstream& is) {
  via* myVia;
  vector< string > tokens(1);

  get_next_token(is, tokens[0], LEFCommentChar);
  myVia = locateOrCreateVia(tokens[0]);
  layer* myLayer;
  get_next_token(is, tokens[0], LEFCommentChar);
  while(tokens[0] != "END") {
    if(tokens[0] == "LAYER") {
      get_next_n_tokens(is, tokens, 2, LEFCommentChar);
      myLayer = &layers[layer2id[tokens[0].c_str()]];
      get_next_token(is, tokens[0], LEFCommentChar);
    }
    else if(tokens[0] == "RECT") {
      get_next_n_tokens(is, tokens, 5, LEFCommentChar);
      rect theRect;
      theRect.xLL = atof(tokens[0].c_str());
      theRect.yLL = atof(tokens[1].c_str());
      theRect.xUR = atof(tokens[2].c_str());
      theRect.yUR = atof(tokens[3].c_str());
      myVia->obses.emplace_back(make_pair(myLayer, theRect));
      assert(tokens[4] == LEFLineEndingChar);
      get_next_token(is, tokens[0], LEFCommentChar);
    }
    else if(tokens[0] == "DEFAULT") {
      myVia->viaRule = tokens[0].c_str();
      get_next_token(is, tokens[0], LEFCommentChar);
    }
    else if(tokens[0] == "TOPOFSTACKONLY") {
      myVia->property = tokens[0].c_str();
      get_next_token(is, tokens[0], LEFCommentChar);
    }
    else {
      get_next_token(is, tokens[0], LEFCommentChar);
      // cerr << "read_lef_via:: unsupported keyword " << tokens[0] << endl;
      // exit(1);
    }
  }
  return;
}

void circuit::read_lef_viaRule(ifstream& is) {
  viaRule myViaRule;
  vector< string > tokens(1);
  get_next_n_tokens(is, tokens, 2, LEFCommentChar);
  myViaRule.name = tokens[0].c_str();
  get_next_token(is, tokens[0], LEFCommentChar);
  while(tokens[0] != "END") {
    if(tokens[0] == "LAYER") {
      get_next_n_tokens(is, tokens, 2, LEFCommentChar);
      layer* myLayer = &layers[layer2id[tokens[0]]];
      myViaRule.layers.emplace_back(myLayer);
      get_next_token(is, tokens[0], LEFCommentChar);
    }
    else if(tokens[0] == "ENCLOSURE") {
      get_next_n_tokens(is, tokens, 3, LEFCommentChar);
      double temp1 = atof(tokens[0].c_str());
      double temp2 = atof(tokens[1].c_str());
      myViaRule.enclosure.emplace_back(make_pair(temp1, temp2));
      assert(tokens[2] == LEFLineEndingChar);
      get_next_token(is, tokens[0], LEFCommentChar);
    }
    else if(tokens[0] == "WIDTH") {
      get_next_n_tokens(is, tokens, 4, LEFCommentChar);
      double temp1 = atof(tokens[0].c_str());
      double temp2 = atof(tokens[2].c_str());
      myViaRule.width.emplace_back(make_pair(temp1, temp2));
      assert(tokens[3] == LEFLineEndingChar);
      get_next_token(is, tokens[0], LEFCommentChar);
    }
    else if(tokens[0] == "RECT") {
      get_next_n_tokens(is, tokens, 5, LEFCommentChar);
      rect theRect;
      theRect.xLL = atof(tokens[0].c_str());
      theRect.yLL = atof(tokens[1].c_str());
      theRect.xUR = atof(tokens[2].c_str());
      theRect.yUR = atof(tokens[3].c_str());
      myViaRule.viaRect = theRect;
      assert(tokens[4] == LEFLineEndingChar);
      get_next_token(is, tokens[0], LEFCommentChar);
    }
    else if(tokens[0] == "SPACING") {
      get_next_n_tokens(is, tokens, 4, LEFCommentChar);
      double temp1 = atof(tokens[0].c_str());
      double temp2 = atof(tokens[2].c_str());
      myViaRule.spacing.emplace_back(make_pair(temp1, temp2));
      assert(tokens[3] == LEFLineEndingChar);
      get_next_token(is, tokens[0], LEFCommentChar);
    }
    else {
#ifdef DEBUG
      cerr << "read_lef_viaRule:: unsupported keyword " << tokens[0] << endl;
#endif
      get_next_token(is, tokens[0], LEFCommentChar);
      // exit(1);
    }
  }
  viaRules.emplace_back(myViaRule);
  return;
}

// assumes the keyword MACRO has already been read in
void circuit::read_lef_macro(ifstream& is) {
#ifdef DEBUG
  cout << " read_lef_macro start" << endl;
#endif

  macro* myMacro;
  vector< string > tokens(1);

  get_next_token(is, tokens[0], LEFCommentChar);
  myMacro = locateOrCreateMacro(tokens[0]);

  get_next_token(is, tokens[0], LEFCommentChar);
  while(tokens[0] != "END") {
    if(tokens[0] == "PROPERTY") {
      get_next_n_tokens(is, tokens, 12, LEFCommentChar);
      myMacro->edgetypeLeft = atoi(tokens[4].c_str());
      myMacro->edgetypeRight = atoi(tokens[8].c_str());
      assert(tokens[1] == "\"");
      assert(tokens[11] == LEFLineEndingChar);
    }
    else if(tokens[0] == "CLASS") {
      get_next_n_tokens(is, tokens, 2, LEFCommentChar);
      assert(tokens[1] == LEFLineEndingChar);
      myMacro->type = tokens[0];
    }
    else if(tokens[0] == "ORIGIN") {
      get_next_n_tokens(is, tokens, 3, LEFCommentChar);
      assert(tokens[2] == LEFLineEndingChar);
      myMacro->xOrig = atof(tokens[0].c_str());
      myMacro->yOrig = atof(tokens[1].c_str());
    }
    else if(tokens[0] == "SIZE") {
      get_next_n_tokens(is, tokens, 4, LEFCommentChar);
      assert(tokens[1] == "BY");
      assert(tokens[3] == LEFLineEndingChar);
      myMacro->width = atof(tokens[0].c_str());
      myMacro->height = atof(tokens[2].c_str());
      // remove because of DAC 16 bench ( read_def_size )
      // if( (int)floor(myMacro->height*DEFdist2Microns/rowHeight+0.5) >
      // max_cell_height )
      //    max_cell_height =
      //    (int)floor(myMacro->height*DEFdist2Microns/rowHeight+0.5);
    }
    else if(tokens[0] == "SITE")
      read_lef_macro_site(is, myMacro);
    else if(tokens[0] == "PIN")
      read_lef_macro_pin(is, myMacro);
    else if(tokens[0] == "SYMMETRY") {
      // NOTE: this contest does not allow flipping/rotation
      // even though symmetries are specified for a macro
      get_next_token(is, tokens[0], LEFCommentChar);
      while(tokens[0] != LEFLineEndingChar)
        get_next_token(is, tokens[0], LEFCommentChar);
      assert(tokens[0] == LEFLineEndingChar);
    }
    else if(tokens[0] == "FOREIGN") {
      get_next_n_tokens(is, tokens, 2, LEFCommentChar);
      // assert(tokens[1] == LEFLineEndingChar);
    }
    else if(tokens[0] == "OBS") {
      get_next_token(is, tokens[0], LEFCommentChar);
      // NOTE: this contest does not handle other than LAYER keyword for OBS
      while(tokens[0] != "END") {
        if(tokens[0] == "LAYER") {
          get_next_n_tokens(is, tokens, 2, LEFCommentChar);
          assert(tokens[1] == LEFLineEndingChar);
          // NOTE: this contest only checks legality on metal 1 layer
          if(tokens[0] == "metal1") {
            get_next_token(is, tokens[0], LEFCommentChar);
            while(tokens[0] != "END" && tokens[0] != "LAYER") {
              assert(tokens[0] == "RECT");
              get_next_n_tokens(is, tokens, 5, LEFCommentChar);
              rect theRect;
              theRect.xLL = atof(tokens[0].c_str());
              theRect.yLL = atof(tokens[1].c_str());
              theRect.xUR = atof(tokens[2].c_str());
              theRect.yUR = atof(tokens[3].c_str());
              myMacro->obses.emplace_back(theRect);
              get_next_token(is, tokens[0], LEFCommentChar);
            }
          }
          else
            get_next_token(is, tokens[0], LEFCommentChar);
        }
        else
          get_next_token(is, tokens[0], LEFCommentChar);
      }
    }
    else {
      // get_next_token(is, tokens[0], LEFCommentChar);
      // cerr << "read_lef_macro:: unsupported keyword " << tokens[0] << endl;
      // exit(1);
    }
    get_next_token(is, tokens[0], LEFCommentChar);
  }

  read_lef_macro_define_top_power(myMacro);

  get_next_token(is, tokens[0], LEFCommentChar);
  assert(myMacro->name == tokens[0]);
  return;
}

// - - - - - - - define multi row cell & define top power - - - - - - - - //
void circuit::read_lef_macro_define_top_power(macro* myMacro) {
  bool isVddFound = false, isVssFound = false;
  string vdd_str, vss_str;

  auto pinPtr = myMacro->pins.find("vdd");
  if(pinPtr != myMacro->pins.end()) {
    vdd_str = "vdd";
    isVddFound = true;
  }
  else if(pinPtr != myMacro->pins.find("VDD")) {
    vdd_str = "VDD";
    isVddFound = true;
  }

  pinPtr = myMacro->pins.find("vss");
  if(pinPtr != myMacro->pins.end()) {
    vss_str = "vss";
    isVssFound = true;
  }
  else if(pinPtr != myMacro->pins.find("VSS")) {
    vss_str = "VSS";
    isVssFound = true;
  }

  if(isVddFound || isVssFound) {
    double max_vdd = 0;
    double max_vss = 0;

    macro_pin* pin_vdd = NULL;
    if(isVddFound) {
      pin_vdd = &myMacro->pins.at(vdd_str);
      for(int i = 0; i < pin_vdd->port.size(); i++) {
        if(pin_vdd->port[i].yUR > max_vdd) {
          max_vdd = pin_vdd->port[i].yUR;
        }
      }
    }

    macro_pin* pin_vss = NULL;
    if(isVssFound) {
      pin_vss = &myMacro->pins.at(vss_str);
      for(int j = 0; j < pin_vss->port.size(); j++) {
        if(pin_vss->port[j].yUR > max_vss) {
          max_vss = pin_vss->port[j].yUR;
        }
      }
    }

    if(max_vdd > max_vss)
      myMacro->top_power = VDD;
    else
      myMacro->top_power = VSS;

    if(pin_vdd && pin_vss) {
      if(pin_vdd->port.size() + pin_vss->port.size() > 2) {
        myMacro->isMulti = true;
      }
      else if(pin_vdd->port.size() + pin_vss->port.size() < 2) {
        cerr << "read_lef_macro:: power num error, vdd + vss => "
             << (pin_vdd->port.size() + pin_vss->port.size()) << endl;
        exit(1);
      }
    }
  }
}

// assumes the keyword SITE has already been read in
void circuit::read_lef_macro_site(ifstream& is, macro* myMacro) {
#ifdef DEBUG
  cerr << "read_lef_macro_site:: begin\n" << endl;
#endif
  site* mySite = NULL;
  vector< string > tokens(1);
  get_next_token(is, tokens[0], LEFCommentChar);

  mySite = locateOrCreateSite(tokens[0]);
  myMacro->sites.emplace_back(site2id.find(mySite->name)->second);

  // does not support [sitePattern]
  unsigned numArgs = 0;
  do {
    get_next_token(is, tokens[0], LEFCommentChar);
    ++numArgs;
  } while(tokens[0] != LEFLineEndingChar);

  if(numArgs > 1) {
    cout << "read_lef_macro_site:: WARNING -- bypassing " << numArgs
         << " additional field in " << "MACRO " << myMacro->name << " SITE "
         << sites[site2id.find(mySite->name)->second].name << "." << endl;
    cout << "It is likely that the fields are from [sitePattern], which are "
            "currently not expected."
         << endl;
  }
#ifdef DEBUG
  cerr << "read_lef_macro_site:: end\n" << endl;
#endif
  return;
}

// assumes the keyword PIN has already been read in
void circuit::read_lef_macro_pin(ifstream& is, macro* myMacro) {
#ifdef DEBUG2
  cerr << "read_lef_macro_pin:: begin\n" << endl;
#endif
  macro_pin myPin;
  vector< string > tokens(1);
  get_next_token(is, tokens[0], LEFCommentChar);
  string pinName = tokens[0];
  get_next_token(is, tokens[0], LEFCommentChar);
  while(tokens[0] != "END") {
    if(tokens[0] == "DIRECTION") {
      get_next_n_tokens(is, tokens, 2, LEFCommentChar);
      // assert(tokens[1] == LEFLineEndingChar);
      myPin.direction = tokens[0];
    }
    else if(tokens[0] == "SHAPE") {
      get_next_n_tokens(is, tokens, 2, LEFCommentChar);
      assert(tokens[1] == LEFLineEndingChar);
      myPin.shape = tokens[0];
    }
    else if(tokens[0] == "PORT") {
      get_next_token(is, tokens[0], LEFCommentChar);
      while(tokens[0] != "END") {
        if(tokens[0] == "LAYER") {
          get_next_n_tokens(is, tokens, 2, LEFCommentChar);
          assert(tokens[1] == LEFLineEndingChar);
          layer* myLayer = locateOrCreateLayer(tokens[0]);
          myPin.layer.emplace_back(layer2id.find(myLayer->name)->second);
        }
        else if(tokens[0] == "RECT") {
          rect theRect;
          get_next_n_tokens(is, tokens, 2, LEFCommentChar);
          theRect.xLL = min(atof(tokens[0].c_str()), theRect.xLL);
          if(tokens[1] == "ITERATE") {
            get_next_n_tokens(is, tokens, 3, LEFCommentChar);
            theRect.yLL = min(atof(tokens[0].c_str()), theRect.yLL);
            theRect.xUR = max(atof(tokens[1].c_str()), theRect.xUR);
            theRect.yUR = max(atof(tokens[2].c_str()), theRect.yUR);
          }
          else {
            theRect.yLL = min(atof(tokens[1].c_str()), theRect.yLL);
            get_next_n_tokens(is, tokens, 2, LEFCommentChar);
            theRect.xUR = max(atof(tokens[0].c_str()), theRect.xUR);
            theRect.yUR = max(atof(tokens[1].c_str()), theRect.yUR);
          }
          myPin.port.emplace_back(theRect);
          get_next_token(is, tokens[0], LEFCommentChar);
          assert(tokens[0] == LEFLineEndingChar);
        }
        else {
#ifdef DEBUG
          cerr << "read_lef_macro_pin:: unsupported keyword " << tokens[0]
               << endl;
#endif
        }
        get_next_token(is, tokens[0], LEFCommentChar);
      }
    }
    else if(tokens[0] == "TAPERRULE") {
      get_next_n_tokens(is, tokens, 2, LEFCommentChar);
      assert(tokens[1] == LEFLineEndingChar);
    }
    else if(tokens[0] == "USE") {
      while(tokens[0] != LEFLineEndingChar)
        get_next_token(is, tokens[0], LEFCommentChar);
    }
    else {
#ifdef DEBUG
      cerr << "read_lef_macro_pin:: unsupported keyword " << tokens[0] << endl;
#endif
    }
    get_next_token(is, tokens[0], LEFCommentChar);
  }
  get_next_token(is, tokens[0], LEFCommentChar);
  assert(pinName == tokens[0]);
  myMacro->pins[pinName] = myPin;
  if(pinName == FFClkPortName) myMacro->isFlop = true;
  return;
}

//
// Writing DEF
//
// It opens in_def_name pointer and write DEF in output
//
void circuit::write_def(const string& output) {
  ifstream dot_in_def(in_def_name.c_str());

  if(!dot_in_def.good()) {
    cerr << "write_def:: cannot open'" << in_def_name << "' for wiring. "
         << endl;
    exit(1);
  }

  // save writing pointer into ckt object
  fileOut = fopen(output.c_str(), "w");
  if(!fileOut) {
    cerr << "write_def:: cannot open '" << output << "' for writing. " << endl;
    exit(1);
  }

  // upto meets COMPONENTS, just copy contents from dot_in_def pointer
  string line;
  while(!dot_in_def.eof()) {
    if(dot_in_def.eof()) break;
    getline(dot_in_def, line);

    line += "\n";
    fwrite(line.c_str(), line.length(), 1, fileOut);

    if(strncmp(line.c_str(), "COMPONENTS", 10) == 0) {
      // Write Components Sections
      WriteDefComponents(in_def_name);
      do {
        getline(dot_in_def, line);
      } while(strncmp(line.c_str(), "END COMPONENTS", 14) != 0);

      line += "\n";
      fwrite(line.c_str(), line.length(), 1, fileOut);
    }
  }
  cout << " DEF file write success !! " << endl;
  cout << " location : " << output << endl;
  cout << "-------------------------------------------------------------------"
       << endl;
  return;
}

void circuit::copy_init_to_final() {
  for(vector< cell >::iterator theCell = cells.begin(); theCell != cells.end();
      ++theCell) {
    theCell->x_coord = theCell->init_x_coord;
    theCell->y_coord = theCell->init_y_coord;
  }

  return;
}

// large cell stor根据面积降序排序
void circuit::init_large_cell_stor() {
  large_cell_stor.reserve(cells.size());
  for(auto& curCell : cells) {
    large_cell_stor.emplace_back(
        make_pair(curCell.width * curCell.height, &curCell));
  }

  sort(large_cell_stor.begin(), large_cell_stor.end(),
       [](const pair< float, cell* >& lhs, const pair< float, cell* >& rhs) {
         return (lhs.first > rhs.first);
       });
  return;
}
