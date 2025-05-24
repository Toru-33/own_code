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

#ifndef EVALUATE_H
#define EVALUATE_H

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <vector>
#include <map>
#include <cmath>
#include <climits>
#include <algorithm>
#include <limits>
#include <assert.h>
#include <queue>
#include <omp.h>
#include "mymeasure.h"
// #include "th/ns.h"

// hashmap settings
#ifdef USE_GOOGLE_HASH
#include <sparsehash/dense_hash_map>
#define INITSTR "SANGGIDO!@#!@#"
#define OPENDP_HASH_MAP google::dense_hash_map
#else
#include <unordered_map>
#define OPENDP_HASH_MAP std::unordered_map
#endif

#define INIT false
#define FINAL true

#define PI_PIN 1
#define PO_PIN 2
#define NONPIO_PIN 3

#define OPENDP_NAMESPACE_OPEN namespace opendp {
#define OPENDP_NAMESPACE_CLOSE }

OPENDP_NAMESPACE_OPEN

enum power { VDD, VSS };

template < class T >
using max_heap = std::priority_queue< T >;

struct point {
  int x;
  int y;
  point(int x = 0, int y = 0) : x(x), y(y) {}
};  // lxm:定义一个point类，用来表示坐标点(A*算法需要)

struct rect {
  double xLL, yLL;
  double xUR, yUR;
  rect()
      : xLL(std::numeric_limits< double >::max()),
        yLL(std::numeric_limits< double >::max()),
        xUR(std::numeric_limits< double >::min()),
        yUR(std::numeric_limits< double >::min()) {}
  rect(double x_ll, double x_ur, double y_ll, double y_ur) {
    xLL = x_ll;
    xUR = x_ur;
    yLL = y_ll;
    yUR = y_ur;
  }
  void dump() { printf("%f : %f - %f : %f\n", xLL, yLL, xUR, yUR); }
};

struct site {
  std::string name;
  double width;     /* in microns */
  double height;    /* in microns */
  std::string type; /* equivalent to class, I/O pad or CORE */
  std::vector< std::string > symmetries; /* {X | Y | R90} */

  site() : name(""), width(0.0), height(0.0), type("") {}
  site(const site& s)
      : name(s.name),
        width(s.width),
        height(s.height),
        type(s.type),
        symmetries(s.symmetries) {}
  void print();
};

struct mincut {
  int via_num;
  double width;
  double length;
  double within;
  std::string direction;  // FROMABOVE or FROMBELOW
  mincut() : via_num(0), width(0.0), length(0.0), within(0.0), direction("") {}
};

struct space {
  int adj;
  std::string type;
  double min, max;
  space() : adj(0), type(""), min(0), max(0) {}
};

struct layer {
  std::string name;
  std::string type;
  std::string direction;
  double xPitch;  /* in microns */
  double yPitch;  /* in microns */
  double xOffset; /* in microns */
  double yOffset; /* in microns */
  double width;   /* in microns */

  // added by SGD
  double maxWidth;
  std::string spacing;
  std::string minStep;
  double area;
  double minEnclosedArea;
  std::vector< mincut > mincut_rule;
  std::vector< space > spacing_rule;
  // -------------
  layer()
      : name(""),
        type(""),
        direction(""),
        xPitch(0.0),
        yPitch(0.0),
        xOffset(0.0),
        yOffset(0.0),
        width(0.0),
        maxWidth(0.0),
        spacing(""),
        minStep(""),
        area(0.0),
        minEnclosedArea(0.0) {}
  void print();
};

struct viaRule {
  std::string name;
  std::vector< layer* > layers;
  std::vector< std::pair< double, double > > enclosure;
  std::vector< std::pair< double, double > > width;
  std::vector< std::pair< double, double > > spacing;
  rect viaRect;
  viaRule() : name("") {}
};

struct via {
  std::string name;
  std::string viaRule;
  std::string property;
  std::vector< std::pair< layer*, rect > > obses;
  via() : name(""), viaRule(""), property("") {}
};

struct macro_pin {
  std::string direction;

  std::vector< rect > port;
  std::vector< unsigned > layer;

  std::string shape;
  macro_pin() : direction(""), shape(""), layer(0) {}
};

struct macro {
  // lxm:对应cell的gate类型
  std::string name;
  std::string type; /* equivalent to class, I/O pad or CORE */
  bool isFlop;      /* clocked element or not */
  bool isMulti;     /* single row = false , multi row = true */
  double xOrig;     /* in microns */
  double yOrig;     /* in microns */
  double width;     /* in microns */
  double height;    /* in microns */
  // lxm:除非有PROPERTY字段，不然边距值都是0
  int edgetypeLeft;   // 1 or 2
  int edgetypeRight;  // 1 or 2
  std::vector< unsigned > sites;

  OPENDP_HASH_MAP< std::string, macro_pin > pins;

  std::vector< rect >
      obses;        /* keyword OBS for non-rectangular shapes in micros */
  power top_power;  // VDD = 0  VSS = 1 enum

  macro()
      : name(""),
        type(""),
        isFlop(false),
        isMulti(false),
        xOrig(0.0),
        yOrig(0.0),
        width(0.0),
        height(0.0),
        edgetypeLeft(0),
        edgetypeRight(0) {
#ifdef USE_GOOGLE_HASH
    pins.set_empty_key(INITSTR);
#endif
  }
  void print();
};

struct pin {
  // from verilog
  std::string name; /* Name of pins : instance name + "_" + port_name */
  unsigned id;
  unsigned owner; /* The owners of PIs or POs are UINT_MAX */
  unsigned net;
  unsigned type;    /* 1=PI_PIN, 2=PO_PIN, 3=others */
  bool isFlopInput; /* is this pin an input  of a clocked element? */
  bool isFlopCkPort;

  // from .def
  double x_coord, y_coord; /* (in DBU) */
  double x_offset,
      y_offset; /* COG of VIA relative to the origin of a cell, (in DBU) */
  bool isFixed; /* is this node fixed? */

  pin()
      : name(""),
        id(UINT_MAX),
        owner(UINT_MAX),
        net(UINT_MAX),
        type(UINT_MAX),
        isFlopInput(false),
        isFlopCkPort(false),
        x_coord(0.0),
        y_coord(0.0),
        x_offset(0.0),
        y_offset(0.0),
        isFixed(false) {}
  void print();
};

struct cell {
  // lxm:cell的名称，比如instxx
  std::string name;
  unsigned id;
  unsigned type;                  /* index to some predefined macro */
  int x_coord, y_coord;           /* (in DBU) */
  int init_x_coord, init_y_coord; /* (in DBU) */
  int x_pos, y_pos;               /* (in DBU) */
  double width, height;           /* (in DBU) */
  bool isFixed;                   /* fixed cell or not */
  bool isPlaced;  // lxm:一开始读def后不与def的Placed相对应，默认是false
  bool inGroup;   // lxm:如果cell属于某个group，则为true
  bool hold;      // lxm:如果预放置成功就为true
  bool is_ff;     // lxm:是否是FlipFlop单元
  int init_x_ff_coord, init_y_ff_coord;  // lxm:用来聚类后记录ff的原始坐标
  int cluster_id;
  std::vector< int > pins;            // lxm:cell的pin列表
  std::vector< int > connected_nets;  // lxm:连接到的net索引集合
  double capacitance;                 // th:ff的预设电容
  double
      nets_hpwl;  // lxm:获取每个cell的相关net的线长，用来设置每个分区的最大线长牺牲值

  int cost;  // lxm:用于衡量放置register的重叠的non-register cell的成本

  // std::vector< cell* > overlap_cells;  // lxm:记录与当前register重叠的cell

  bool isOverlap;  // lxm:是否需要Relegalization

  unsigned mergeIdx;  // lxm:记录要和合并cell的idx
  int level;          // lxm:记录cell的层级
  int parent_x;       // lxm:记录cell的父节点的x坐标
  int parent_y;       // lxm:记录cell的父节点的y坐标

  unsigned region;
  OPENDP_HASH_MAP< std::string, unsigned >
      ports; /* <port name, index to the pin> */
  std::string cellorient;
  std::string group;

  double dense_factor;
  int dense_factor_count;
  unsigned binId;
  double disp;

  cell()
      : name(""),
        type(UINT_MAX),
        id(UINT_MAX),
        x_coord(0),
        y_coord(0),
        init_x_coord(0),
        init_y_coord(0),
        x_pos(INT_MAX),
        y_pos(INT_MAX),
        width(0.0),
        height(0.0),
        cost(0),  // lxm
        isFixed(false),
        isPlaced(false),
        is_ff(false),      // lxm
        isOverlap(false),  // lxm
        inGroup(false),
        hold(false),
        region(UINT_MAX),
        cellorient(""),
        group(""),
        dense_factor(0.0),
        dense_factor_count(0),
        binId(UINT_MAX),
        disp(0.0) {
#ifdef USE_GOOGLE_HASH
    ports.set_empty_key(INITSTR);
#endif
  }
  void print();
};

struct pixel {
  std::string name;
  // lxm:row能完全囊括的就是1，否则就是0
  double util;
  int x_pos;
  int y_pos;
  unsigned group;  // group id
  // lxm:这个小格中包含的cell
  cell* linked_cell;
  bool isValid;  // false for dummy place
  pixel()
      : name(""),
        util(0.0),
        x_pos(0.0),
        y_pos(0.0),
        group(UINT_MAX),
        linked_cell(NULL),
        isValid(true) {}
};

struct net {
  std::string name;
  unsigned source;               /* input pin index to the net */
  std::vector< unsigned > sinks; /* sink pins indices of the net */
  double xLL, yLL, xUR, yUR;     // lxm:边界框
  bool is_IONET;                 // lxm:是否是IONET
  net() : name(""), source(UINT_MAX) {}
  void print();
};

struct row {
  /* from DEF file */
  std::string name;
  unsigned site;
  int origX; /* (in DBU) */
  int origY; /* (in DBU) */
  int stepX; /* (in DBU) */
  int stepY; /* (in DBU) */
  int numSites;
  // lxm:做detail或routing的case时，row的方向是FS、N的交替循环
  std::string siteorient;
  power top_power;

  std::vector< cell* > cell_list;

  row()
      : name(""),
        site(UINT_MAX),
        origX(0),
        origY(0),
        stepX(0),
        stepY(0),
        numSites(0),
        siteorient("") {}
  void print();
};

struct group {
  std::string name;
  std::string type;
  std::string tag;
  std::vector< rect > regions;
  std::vector< cell* > siblings;
  std::vector< pixel* > pixels;
  // lxm:当前group所有region的最大box区域
  rect boundary;
  double util;
  // lxm:for cts-driven placement
  std::vector< cell* > siblings_ff;

  group() : name(""), type(""), tag(""), util(0.0) {}
  void dump(std::string temp_) {
    std::cout << temp_ << " name : " << name << " type : " << type
              << " tag : " << tag << " end line " << std::endl;
    for(int i = 0; i < regions.size(); i++) regions[i].dump();
  };
};

struct sub_region {
  rect boundary;
  int x_pos, y_pos;
  int width, height;
  std::vector< cell* > siblings;
  sub_region() : x_pos(0), y_pos(0), width(0), height(0) {
    siblings.reserve(81920);
  }
};

struct density_bin {
  double lx, hx;     /* low/high x coordinate */
  double ly, hy;     /* low/high y coordinate */
  double area;       /* bin area */
  double m_util;     /* bin's movable cell area */
  double f_util;     /* bin's fixed cell area */
  double free_space; /* bin's freespace area */
  double overflow;
  double density_limit;
  void print();
};

struct track {
  std::string axis;  // X or Y
  unsigned start;
  unsigned num_track;
  unsigned step;
  std::vector< layer* > layers;
  track() : axis(""), start(0), num_track(0), step(0) {}
};

struct CandidatePosition {         // lxm:方便交换相同宽高cell
  std::pair< int, int > position;  // 候选位置坐标
  bool isSwap;                     // 是否为交换操作
  cell* swapCell;                  // 交换的cell指针，若非交换则为nullptr

  CandidatePosition() : isSwap(false), swapCell(nullptr) {}
};

// lxm: AbacusCluster and SubRow struct definitions for parallel Abacus
struct AbacusCluster {
  std::vector< cell* > cells;  // 簇中的单元
  double position;             // 簇的最优位置
  double total_weight;         // 总权重
  double total_width;          // 总宽度
  double q_value;              // 计算最优位置的中间值

  AbacusCluster()
      : position(0.0), total_weight(0.0), total_width(0.0), q_value(0.0) {}

  // 深拷贝构造函数
  AbacusCluster(const AbacusCluster& other)
      : cells(
            other
                .cells),  // vector<cell*>
                          // 的拷贝构造函数会复制指针，但这是期望的，因为cell对象本身由circuit类拥有
        position(other.position),
        total_weight(other.total_weight),
        total_width(other.total_width),
        q_value(other.q_value) {}

  // 深拷贝赋值操作符
  AbacusCluster& operator=(const AbacusCluster& other) {
    if(this == &other) {
      return *this;
    }
    cells = other.cells;  // 同上，复制指针vector
    position = other.position;
    total_weight = other.total_weight;
    total_width = other.total_width;
    q_value = other.q_value;
    return *this;
  }

  // 可选: 添加移动构造函数和移动赋值操作符 (C++11 及以上)
  AbacusCluster(AbacusCluster&& other) noexcept
      : cells(std::move(other.cells)),
        position(other.position),
        total_weight(other.total_weight),
        total_width(other.total_width),
        q_value(other.q_value) {
    // 将源对象置于有效但空的状态
    other.position = 0.0;
    other.total_weight = 0.0;
    other.total_width = 0.0;
    other.q_value = 0.0;
  }

  AbacusCluster& operator=(AbacusCluster&& other) noexcept {
    if(this == &other) {
      return *this;
    }
    cells = std::move(other.cells);
    position = other.position;
    total_weight = other.total_weight;
    total_width = other.total_width;
    q_value = other.q_value;

    // 将源对象置于有效但空的状态
    other.position = 0.0;
    other.total_weight = 0.0;
    other.total_width = 0.0;
    other.q_value = 0.0;
    return *this;
  }
};

struct SubRow {
  int start_x;                            // 子行起始位置(考虑障碍物)
  int end_x;                              // 子行结束位置
  int y_pos;                              // 行的y坐标 (实际坐标值)
  int remaining_width;                    // 剩余宽度
  std::vector< AbacusCluster > clusters;  // 当前行的簇
  SubRow() : start_x(0), end_x(0), y_pos(0), remaining_width(0) {}
};

struct YInterval {
  double yLL, yUR;
  cell* owner;
  bool operator<(YInterval const& o) const {
    if(yLL != o.yLL) return yLL < o.yLL;
    return owner < o.owner;
  }
};

struct Event {
  double x;
  bool isEnter;  // true = 左边界，false = 右边界
  YInterval ival;
};

class circuit {
 public:
  bool GROUP_IGNORE;
  void init_large_cell_stor();
  //--------------lxm:for cts-driven placement-----------
  std::vector< rect > fix_rects;
  int multi_number = 0;

  // lxm:迭代次数，用来多次迭代优化时钟线长
  int iter = 0;
  int max_iter = 0;

  void FF_placement_Ingroup(std::vector< cell* >& group_cells, rect box);
  void parallel_FF_placement_Ingroup();
  void lg_std_cells(std::vector< cell* > std_cells);
  void OptimizeSignalWireLength(std::vector< cell* > std_cells);

  void DensityNearestNeighborSearch(std::vector< cell* > std_cells);
  std::pair< bool, pixel* > density_nearest_neighbor_search(cell* theCell,
                                                            int x_coord,
                                                            int y_coord);
  bool update_pixel(cell* theCell, int x_pos,
                    int y_pos);  // lxm:更新合法化位置
  // lxm:初始化ff_cells
  void init_ff_cell();

  // lxm:往ff创建的时钟树中心聚类
  void root_driven_cluster();
  // lxm:将在macro内的cell移到最近的边界处
  void move_cells_to_nearest_fix();
  // lxm:获取优先区域
  void getOptimalRegion(std::vector< cell* > ff_cells);
  double UpdatecalculateCellHPWL(cell* theCell, int x, int y);
  double calculateCellHPWL(cell* theCell, int x,
                           int y);  // lxm:给定cell获取其信号线线长
  double calculateCLKHPWL(cell* theCell, int x, int y);
  opendp::point findOptimalStdCellPosition(
      cell* theCell,
      double initial_hpwl);  // lxm:for std cell
  opendp::point findOptimalStdCellPosition_Ingroup(
      cell* theCell, double initial_hpwl, rect box);  // lxm:for std cell

  opendp::point findOptimalFFPosition_Ingroup(cell* theCell,
                                              double initial_hpwl, rect box);

  opendp::point findOptimalCellPosition_MOAStar(cell* theCell, int root_x,
                                                int root_y, rect box);

  opendp::point optimizeFFCellPosition(cell* theCell, int root_x, int root_y,
                                       double initial_hpwl);
  opendp::point optimizeFFCellPosition_Den(cell* theCell, int root_x,
                                           int root_y, double initial_hpwl);

  opendp::point findOptimalCellPosition(cell* theCell, int root_x, int root_y,
                                        double initial_hpwl, rect box);

  double EROPS(std::vector< cell* > partition_cells, rect box);
  double OptimizeWirelengthWithAStar(std::vector< cell* >& std_cells);
  double OptimizeWirelengthWithAStar_Ingroup(std::vector< cell* >& std_cells,
                                             rect box);
  double calculateOptimalDensityFactorChange(cell* theCell, int new_x,
                                             int new_y);
  void updateDensityAfterFFCellMove(cell* movedCell, int oldBinId,
                                    int newBinId);
  void moveFFCellAndUpdateDensity(cell* movedCell, int new_x, int new_y);

  void search_in_direction(int displacement, int x_pos, int y_pos,
                           const std::pair< int, int >& dir,
                           std::vector< pixel* >& avail_list,
                           std::vector< double >& score, cell* theCell,
                           int x_start, int x_end, int y_start, int y_end,
                           int site_num, bool flag, int max_size);

  std::pair< bool, pixel* > hexagonal_search(cell* theCell, int x_coord,
                                             int y_coord);
  std::pair< bool, pixel* > hexagonal_search_FF(cell* theCell, int x_coord,
                                                int y_coord, rect box);

  //----------cts end--------

  OPENDP_HASH_MAP< std::string, unsigned >
      macro2id; /* OPENDP_HASH_MAP between macro name and ID */
  OPENDP_HASH_MAP< std::string, unsigned >
      cell2id; /* OPENDP_HASH_MAP between cell  name and ID */
  OPENDP_HASH_MAP< std::string, unsigned >
      pin2id; /* OPENDP_HASH_MAP between pin   name and ID */
  OPENDP_HASH_MAP< std::string, unsigned >
      net2id; /* OPENDP_HASH_MAP between net   name and ID */
  OPENDP_HASH_MAP< std::string, unsigned >
      row2id; /* OPENDP_HASH_MAP between row   name and ID */
  OPENDP_HASH_MAP< std::string, unsigned >
      site2id; /* OPENDP_HASH_MAP between site  name and ID */
  OPENDP_HASH_MAP< std::string, unsigned >
      layer2id; /* OPENDP_HASH_MAP between layer name and ID */

  OPENDP_HASH_MAP< std::string, unsigned > via2id;
  std::map< std::pair< int, int >, double > edge_spacing; /* spacing
                                                   OPENDP_HASH_MAP between edges
                                                   1 to 1 , 1 to 2, 2 to 2 */
  OPENDP_HASH_MAP< std::string, unsigned >
      group2id; /* group between name -> index */

  double design_util;
  double sum_displacement;

  unsigned num_fixed_nodes;
  double total_mArea; /* total movable cell area */
  double total_fArea; /* total fixed cell area (excluding terminal NIs) */
  double designArea;  /* total placeable area (excluding row blockages) */
  double rowHeight;
  double lx, rx, by,
      ty; /* placement image's left/right/bottom/top end coordintes */
  rect die;
  rect core;  // COREAREA

  double minVddCoordiY;  // VDD stripe coordinates for parsing
  power initial_power;   // informations

  double max_utilization;
  double displacement;
  double max_disp_const;
  int wsite;
  int max_cell_height;  // lxm:用于判断是否有多行高cell，最高为几个rowHeight

  double avg_cell_area;  // lxm:平均cell面积，用于密度计算
  int max_height;        // lxm:最大的cell高度，便于新的密度计算函数使用
  int max_cell_width;    // lxm:加上的最大cell宽度，便于新的密度计算函数使用

  int bin_size;  // lxm:多少rowHeioght为一个bin的宽高

  unsigned num_cpu;

  std::string out_def_name;
  std::string in_def_name;

  // lxm:用来存储FlipFlop单元
  std::vector< cell* > ff_cells;
  std::vector< cell* > all_ff_cells;
  std::vector< cell* > fix_cells;

  std::vector< cell* > group_cells;  // lxm:记录所有group的cell

  /* benchmark generation */
  std::string benchmark; /* benchmark name */

  // 2D - pixel grid;
  // lxm:每个grid的纵坐标是所在的行数，横坐标是所在的列数，第一维是列
  pixel** grid;
  pixel** grid_init;  // lxm:初始状态的grid,如果不合法就delete掉

  bool init_lg_flag;  // lxm:用于判断初始def是否已经合法化
  bool wire_flag;     // lxm:用于开启纯线长优化模式
  bool high_density;  // lxm:用于对高密度单独处理
  int thread_num;     // lxm:用于设置线程数
  cell dummy_cell;
  std::vector< sub_region > sub_regions;
  std::vector< track > tracks;

  // used for LEF file
  std::string LEFVersion;
  std::string LEFNamesCaseSensitive;
  std::string LEFDelimiter;
  std::string LEFBusCharacters;
  double LEFManufacturingGrid;

  unsigned MAXVIASTACK;
  layer* minLayer;
  layer* maxLayer;

  // used for DEF file
  std::string DEFVersion;
  std::string DEFDelimiter;
  std::string DEFBusCharacters;
  std::string design_name;
  unsigned DEFdist2Microns;
  std::vector< std::pair< unsigned, unsigned > > dieArea;

  std::vector< site > sites;   /* site list */
  std::vector< layer > layers; /* layer list */
  std::vector< macro > macros; /* macro list */
  // lxm:最后写回def用的是cells
  std::vector< cell > cells;       /* cell list */
  std::vector< cell* > cells_init; /* lxm: cell指针 list */
  std::vector< net > nets;         /* net list */
  std::vector< pin > pins;         /* pin list */

  std::vector< row >
      prevrows;  // fragmented row list
                 // (由于各种设计约束（如围栏区域、阻塞区域、不同的单元高度等）导致行（row）被分割成多个不连续的部分)
  std::vector< row > rows; /* row list */

  std::vector< via > vias;
  std::vector< viaRule > viaRules;
  std::vector< group > groups; /* group list from .def */

  std::vector< std::pair< double, cell* > >
      large_cell_stor;  // lxm：first是面积

  /* locateOrCreate helper functions - parser_helper.cpp */
  macro* locateOrCreateMacro(const std::string& macroName);
  cell* locateOrCreateCell(const std::string& cellName);
  net* locateOrCreateNet(const std::string& netName);
  pin* locateOrCreatePin(const std::string& pinName);
  row* locateOrCreateRow(const std::string& rowName);
  site* locateOrCreateSite(const std::string& siteName);
  layer* locateOrCreateLayer(const std::string& layerName);
  via* locateOrCreateVia(const std::string& viaName);
  group* locateOrCreateGroup(const std::string& groupName);
  void print();

  /* IO helpers for LEF - parser.cpp */
  void read_lef_site(std::ifstream& is);
  void read_lef_property(std::ifstream& is);
  void read_lef_layer(std::ifstream& is);
  void read_lef_via(std::ifstream& is);
  void read_lef_viaRule(std::ifstream& is);
  void read_lef_macro(std::ifstream& is);
  void read_lef_macro_site(std::ifstream& is, macro* myMacro);
  void read_lef_macro_pin(std::ifstream& is, macro* myMacro);
  // priv func
  void read_lef_macro_define_top_power(macro* myMacro);

  /* IO helpers for DEF - parser.cpp */
  void read_init_def_components(std::ifstream& is);
  void read_final_def_components(std::ifstream& is);
  void read_def_vias(std::ifstream& is);
  void read_def_pins(std::ifstream& is);
  void read_def_special_nets(std::ifstream& is);
  void read_def_nets(std::ifstream& is);
  void read_def_regions(std::ifstream& is);
  void read_def_groups(std::ifstream& is);
  void write_def(const std::string& output);

  void WriteDefComponents(const std::string& inputDef);

  FILE* fileOut;

  circuit();

  /* read files for legalizer - parser.cpp */
  void print_usage();
  void read_files(int argc, char* argv[]);
  void read_constraints(const std::string& input);
  void read_lef(const std::string& input);
  void read_tech_lef(const std::string& input);
  void read_cell_lef(const std::string& input);
  void read_def(const std::string& input, bool init_or_final);
  void read_def_size(const std::string& input);
  void copy_init_to_final();
  void calc_design_area_stats();

  // Si2 parsing engine
  int ReadDef(const std::string& input);
  // int DefVersionCbk(defrCallbackType_e c, const char* versionName,
  // defiUserData ud); int DefDividerCbk(defrCallbackType_e c, const char* h,
  // defiUserData ud); int DefDesignCbk(defrCallbackType_e c, const char*
  // std::string, defiUserData ud); int DefUnitsCbk(defrCallbackType_e c, double
  // d, defiUserData ud); int DefDieAreaCbk(defrCallbackType_e c, defiBox* box,
  // defiUserData ud); int DefRowCbk(defrCallbackType_e c, defiRow* row,
  // defiUserData ud);

  int ReadLef(const std::vector< std::string >& lefStor);

  // utility.cpp - By SGD
  void power_mapping();
  void evaluation();
  double Disp();
  double HPWL(std::string mode);
  double calc_density_factor(double unit);
  void calc_density_factor_new();  // lxm

  void group_analyze();
  std::pair< int, int > nearest_coord_to_rect_boundary(cell* theCell,
                                                       rect* theRect,
                                                       std::string mode);
  int dist_for_rect(cell* theCell, rect* theRect, std::string mode);
  bool check_overlap(rect cell, rect box);
  bool check_overlap(cell* theCell, rect* theRect, std::string mode);
  bool check_inside(rect cell, rect box);
  bool check_inside(cell* theCell, rect* theRect, std::string mode);
  std::pair< bool, std::pair< int, int > > bin_search(int x_pos, cell* theCell,
                                                      int x, int y);

  std::pair< bool, pixel* > diamond_search(cell* theCell, int x, int y);
  std::pair< bool, pixel* > diamond_search_disp(cell* theCell, int x, int y);

  std::vector< cell* > overlap_cells(cell* theCell);
  std::vector< cell* > get_cells_from_boundary(rect* theRect);
  double dist_benefit(cell* theCell, int x_coord, int y_coord);
  double dist_benefit_ff(cell* theCell, int x_coord, int y_coord);
  bool swap_cell(cell* cellA, cell* cellB);
  bool refine_move(cell* theCell, std::string mode);
  bool refine_move(cell* theCell, int x_coord, int y_coord);
  std::pair< bool, cell* > nearest_cell(int x_coord, int y_coord);

  // place.cpp - By SGD
  void simple_placement(CMeasure& measure);
  void non_group_cell_pre_placement();
  void group_cell_pre_placement();
  void non_group_cell_placement(std::string mode);
  void group_cell_placement(std::string mode);
  void group_cell_placement(std::string mode, std::string mode2);
  void brick_placement_1(group* theGroup);
  void brick_placement_2(group* theGroup);
  int group_refine(group* theGroup);
  int group_annealing(group* theGroup);
  int non_group_annealing();
  int non_group_refine();

  //----------------- lxm:for cts-driven placement -------------------

  // lxm:纯线长优化模式
  void wirelength_placement(CMeasure& measure);

  // cluster
  std::pair< bool, std::pair< int, int > > bin_search_FF(cell* theCell, int x,
                                                         int y);
  void FF_pre_placement_non_group();
  void FF_pre_placement_group();
  void FF_placement_non_group(std::string mode);
  void FF_placement_group(std::string mode);
  std::pair< bool, pixel* > diamond_search_FF(cell* theCell, int x_coord,
                                              int y_coord, rect box);
  void getParent(cell* theCell);
  void draw_search_information(std::vector< cell* > cell_list, cell* theCell,
                               int x_coord, int y_coord, rect box);

  bool map_move_FF(cell* theCell, int x, int y);
  bool map_move_FF(cell* theCell, std::string mode);
  bool shift_move_FF(cell* theCell, int x, int y);
  bool shift_move_FF(cell* theCell, std::string mode);
  bool swap_cell_FF(cell* cellA, cell* cellB);
  bool refine_move_FF(cell* theCell, std::string mode);
  bool refine_move_FF(cell* theCell, int x_coord, int y_coord);
  int FF_group_refine(group* theGroup);
  int FF_group_annealing(group* theGroup);
  int FF_non_group_annealing();
  int FF_non_group_refine();

  std::pair< bool, std::pair< int, int > > bin_search_site(
      int x_pos, cell* theCell, int x, int y,
      int site_num);  // lxm:可以根据输入来调控二分边界

  void relegalization_overlap(std::vector< cell* > cell_list);
  std::pair< bool, std::pair< int, int > > relegal_bin_search_site(
      int x_pos, cell* theCell, int x, int y,
      int site_num);  // lxm:可以根据输入来调控二分边界(relegal专用)
  CandidatePosition new_relegal_bin_search_site(int x_pos, cell* theCell, int x,
                                                int y, int site_num);

  void diamond_swap(cell* theCell, int x, int y);

  bool relegal_paint_pixel(cell* theCell, int x_pos, int y_pos);

  void FF_group_op(std::vector< cell* > partition_cells, rect box);
  std::pair< bool, pixel* > diamond_search_FF_group(cell* theCell, int x_coord,
                                                    int y_coord, rect box);
  //-------------------cts-end-----------------------------

  // Parallel Abacus functions
  void initializeSubRows();
  void partitionCells();
  void addCellToCluster(AbacusCluster& cluster, cell* theCell);
  void mergeClusters(AbacusCluster& target_cluster,
                     AbacusCluster& source_cluster);
  void collapseCluster(AbacusCluster& cluster,
                       const SubRow& row);  // Added row context
  void parallelAbacus();
  double tryInsertCell(cell* theCell, SubRow& row,
                       AbacusCluster& temp_cluster);  // Pass cluster by ref
  void placeCell(cell* theCell, SubRow& best_row,
                 AbacusCluster& placed_cluster);  // Pass placed cluster
  void placeRowDP(SubRow& row, cell* newCellToInsert);
  void handleRemainingCells();
  void placeCellAndReoptimizeRow(SubRow& row,
                                 cell* newCellToInsert /* can be nullptr if
                                  just re-placing the existing row */);

  // assign.cpp - By SGD
  void fixed_cell_assign();
  void print_pixels();
  void group_cell_region_assign();
  void non_group_cell_region_assign();
  void y_align();
  void cell_y_align(cell* theCell);
  void group_pixel_assign();
  void group_pixel_assign_2();
  void erase_pixel(cell* theCell);
  bool paint_pixel(cell* theCell, int x_pos, int y_pos);

  // check_legal.cpp - By SGD
  bool check_legality();
  void local_density_check(double unit, double target_Ut);
  void row_check(std::ofstream& os);
  void site_check(std::ofstream& os);
  void edge_check(std::ofstream& os);
  void power_line_check(std::ofstream& os);
  void placed_check(std::ofstream& log);
  void overlap_check(std::ofstream& os);
  void geometric_overlap_check(std::ofstream& os);
  // lxm:init_check:判断输入是否已经合法，如果合法就采用Relegalization
  bool init_check();

  void abacusLegalize();

  // Add a vector of OpenMP locks for sub_rows
  std::vector< omp_lock_t > sub_row_locks;
  void initialize_sub_row_locks();
  void destroy_sub_row_locks();

 private:  // Added private for new members
  std::vector< SubRow >
      sub_rows_vector;  // Renamed to avoid conflict with rows vector
  std::vector< std::vector< cell* > > tile_cells;
};

// parser_helper.cpp
bool is_special_char(char c);
bool read_line_as_tokens(std::istream& is, std::vector< std::string >& tokens);
void get_next_token(std::ifstream& is, std::string& token,
                    const char* beginComment);
void get_next_n_tokens(std::ifstream& is, std::vector< std::string >& tokens,
                       const unsigned n, const char* beginComment);

inline int IntConvert(double fp) { return (int)(fp + 0.5f); }

OPENDP_NAMESPACE_CLOSE

#endif
