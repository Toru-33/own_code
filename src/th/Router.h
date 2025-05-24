// lxm:level version

// #pragma once
// #include "global.h"
// // STL libraries
// #include <iostream>
// #include <fstream>
// #include <cassert>
// #include <iomanip>
// #include <sstream>
// #include <memory>
// #include <string>
// #include <vector>
// #include <algorithm>
// #include <set>
// #include <unordered_set>
// #include <unordered_map>
// #include <map>
// #include <cmath>
// #include <functional>
// #include <queue>
// // #include "stdafx.h"
// using namespace std;

// const double eps = 1e-1;

// using namespace std;

// double round(double number, unsigned int bits);

// class PointPair {
//   // V_{i,k,n}
//  public:
//   double x1, x2, y1, y2;  // from:(x1,y1), to: (x2,y2)
//   PointPair() {}
//   PointPair(double _x1, double _x2, double _y1, double _y2)
//       : x1(_x1), y1(_y1), x2(_x2), y2(_y2) {}
//   bool operator==(PointPair const& var) const {
//     return (x1 == var.x1 && y1 == var.y1 && x2 == var.x2 && y2 == var.y2);
//   }
// };

// class GrSteiner;
// // 要
// class GridPoint {
//  public:
//   double x, y;

//   GridPoint() {}
//   GridPoint(double _x, double _y) : x(_x), y(_y) {}

//   bool operator==(const GridPoint& rhs) const {
//     return (x == rhs.x && y == rhs.y);
//   }
//   friend inline std::ostream& operator<<(std::ostream& os,
//                                          const GridPoint& gp) {
//     os << "(" << gp.x << "," << gp.y << ")";
//     return os;
//   }
// };

// class Sink : public GridPoint {
//  public:
//   int id;
//   string name;
//   double cap;
//   Sink() {
//     id = -1;
//     x = -1;
//     y = -1;
//     cap = -1;
//     name = "";
//   }
//   Sink(int _id, double _x, double _y, double _cap, string _name) {
//     id = _id;
//     x = _x;
//     y = _y;
//     cap = _cap;
//     name = _name;
//   }

//   string str_xy() {
//     stringstream s;
//     s << "[" << x << "," << y << "]";
//     return s.str();
//   }
//   friend inline std::ostream& operator<<(std::ostream& os, const Sink& sink)
//   {
//     // os << "Sink " << sink.id << ", x: " << fixed << setprecision(0) <<
//     sink.x
//     // << ", y: " << sink.y;
//     os << sink.id << " " << fixed << setprecision(5) << sink.x << " " <<
//     sink.y; return os;
//   }
// };

// class GrSteiner : public GridPoint {
//  public:
//   GrSteiner* lc;
//   GrSteiner* rc;
//   GrSteiner* par;
//   double last_len;
//   int id;

//   GrSteiner() {};
//   GrSteiner(GridPoint p) {
//     x = p.x;
//     y = p.y;
//     lc = NULL;
//     rc = NULL;
//     par = NULL;
//     last_len = 0;
//     id = -1;
//   }

//   void set_lc(GrSteiner* child) { lc = child; }
//   void set_rc(GrSteiner* child) { rc = child; }
//   void set_par(GrSteiner* p) { par = p; }
//   void set_last_len(double len) { last_len = len; }
// };
// // 。
// class Segment {
//  public:
//   GridPoint p1, p2;  // p1 has lower y
//   // GridPoint center;
//   // vector<Segment> ch[2];
//   // vector<Segment> par;
//   int id = 0;  // unique id for each segment
//   double delay;
//   double cap;
//   vector< double > dis_a;
//   vector< double > dis_b;
//   vector< GrSteiner > left_wire_nodes;
//   vector< GrSteiner > right_wire_nodes;
//   double last_len_a = 0;
//   double last_len_b = 0;
//   int bufNumInPath = 0;
//   bool isCalculate = false;

//   Segment() {}
//   Segment(GridPoint u, GridPoint v) : p1(u), p2(v) {
//     delay = 0;
//     if(p1.y > p2.y) {
//       swap(p1, p2);
//     }
//   }

//   double get_last_len() { return max(last_len_a, last_len_b); }

//   bool isLeaf() {
//     if(abs(p1.x - p2.x) < 1e-5 && abs(p1.y - p2.y) < 1e-5)
//       return true;
//     else
//       return false;
//   }

//   bool operator==(const Segment& rhs) const {
//     return (p1 == rhs.p1 && p2 == rhs.p2);
//   }

//   friend inline std::ostream& operator<<(std::ostream& os, const Segment&
//   seg) {
//     os << "Seg: (" << seg.p1 << "," << seg.p2 << ")";
//     return os;
//   }

//   double slope() {
//     if(isLeaf()) {
//       return 0;
//     }
//     if(abs(p2.x - p1.x) < 1e-5) {
//       return 0;
//     }
//     return round(1.0 * (p1.y - p2.y) / (p1.x - p2.x));
//   }

//   Segment intersect(Segment& rhs) {
//     // DEBUG
//     // if (rhs.p1.x == 1147748 && rhs.p1.y == 5695000) {
//     //     cout << "hhhhhhhhhhhh" << endl;
//     //     cout << "zzzzzzzzzzzzz" << endl;
//     // }
//     // DEBUG

//     double cur_slope = slope();
//     double rhs_slope = rhs.slope();

//     // check if 4 points same line
//     // if (abs(cur_slope - rhs_slope) < eps) {
//     //     if (abs((rhs.p1.y - p1.y) * (p2.x - p1.x) - (p2.y - p1.y) *
//     (rhs.p1.x
//     //     - p1.x)) < eps) {
//     if(rhs.isLeaf()) {  // if current segment is intersecting a single grid
//                         // point
//       Segment ret;
//       if(abs((rhs.p1.y - p1.y) * (p2.x - p1.x) -
//              (p2.y - p1.y) * (rhs.p1.x - p1.x)) < eps) {
//         // if ((rhs.p1.y - p1.y) * (p2.x - p1.x) == (p2.y - p1.y) * (rhs.p1.x
//         -
//         // p1.x)) {
//         if(p1.y - eps <= rhs.p1.y &&
//            rhs.p1.y <= p2.y + eps) {  // valid intersection
//           Segment ret = rhs;
//           ret.id = -2;  // return single point intersection

//           ret.p1.x = round(ret.p1.x);
//           ret.p1.y = round(ret.p1.y);
//           ret.p2.x = round(ret.p2.x);
//           ret.p2.y = round(ret.p2.y);

//           return ret;
//         }
//       }
//       ret.id = -1;
//       return ret;
//     }
//     if(abs(cur_slope - rhs_slope) < eps) {
//       if(abs((rhs.p1.y - p1.y) * (p2.x - p1.x) -
//              (p2.y - p1.y) * (rhs.p1.x - p1.x)) < eps) {
//         assert(rhs.p1.y <= rhs.p2.y && p1.y <= p2.y);
//         GridPoint upper, lower;
//         if(rhs.p2.y < p2.y) {
//           upper = rhs.p2;
//         }
//         else {
//           upper = p2;
//         }
//         if(rhs.p1.y > p1.y) {
//           lower = rhs.p1;
//         }
//         else {
//           lower = p1;
//         }
//         if(upper.y < lower.y) {
//           Segment ret;
//           ret.id = -1;
//           return ret;

//           // cout << "No overlap between two segs on the line" << endl;
//           // exit(1);
//         }

//         lower.x = round(lower.x);
//         lower.y = round(lower.y);
//         upper.x = round(upper.x);
//         upper.y = round(upper.y);

//         return Segment(lower, upper);
//       }
//       else {
//         Segment ret;
//         ret.id = -1;
//         return ret;
//       }
//     }
//     else {
//       // might be 1 point or 0
//       double A1 = p2.y - p1.y;
//       // double B1 = p2.x - p1.x;
//       double B1 = p1.x - p2.x;
//       double C1 = A1 * p1.x + B1 * p1.y;
//       double A2 = rhs.p2.y - rhs.p1.y;
//       // double B2 = rhs.p2.x - rhs.p1.x;
//       double B2 = rhs.p1.x - rhs.p2.x;
//       double C2 = A2 * rhs.p1.x + B2 * rhs.p1.y;
//       double det = A1 * B2 - A2 * B1;

//       if(det == 0) {
//         Segment ret;
//         if(p1.x - p2.x < eps && p1.y - p2.y < eps) {
//           if(p1.x - rhs.p1.x < eps && p1.y - rhs.p1.y < eps) {
//             ret.p1 = GridPoint(rhs.p1.x, rhs.p1.y);
//             ret.p2 = GridPoint(rhs.p1.x, rhs.p1.y);
//             ret.id = -2;
//           }
//           else if(p1.x - rhs.p2.x < eps && p1.y - rhs.p2.y < eps) {
//             ret.p1 = GridPoint(rhs.p2.x, rhs.p2.y);
//             ret.p2 = GridPoint(rhs.p2.x, rhs.p2.y);
//             ret.id = -2;
//           }
//           else
//             ret.id = -1;
//         }
//         return ret;
//       }

//       double x = (B2 * C1 - B1 * C2) / det;
//       double y = (A1 * C2 - A2 * C1) / det;
//       x = round(x);
//       y = round(y);

//       Segment ret;

//       if(p1.y - eps <= y && y <= p2.y + eps && rhs.p1.y - eps <= y &&
//          y <= rhs.p2.y + eps) {  // valid intersection
//         ret.p1 = GridPoint(x, y);
//         ret.p2 = GridPoint(x, y);
//         ret.id = -2;  // return single point intersection
//       }
//       else {
//         ret.id = -1;
//       }
//       return ret;
//     }
//     // if return with id=-1, means no intersection
//   }
// };
// class TRR {
//  public:
//   Segment core;
//   double radius;
//   TRR() {}
//   TRR(Segment seg, double radi) : core(seg), radius(radi) {}
//   friend inline std::ostream& operator<<(std::ostream& os, const TRR& trr) {
//     os << trr.core << "; radius:" << trr.radius;
//     return os;
//   }
//   Segment intersect(Segment& seg) {
//     vector< GridPoint > trr_boundary_grid;
//     vector< Segment > trr_Sides;
//     trr_boundary_grid.emplace_back(core.p1.x, round(core.p1.y - radius));
//     trr_boundary_grid.emplace_back(round(core.p2.x + radius), core.p2.y);
//     trr_boundary_grid.emplace_back(core.p2.x, round(core.p2.y + radius));
//     trr_boundary_grid.emplace_back(round(core.p1.x - radius),
//                                    core.p1.y);  // clock-wise
//     for(int i = 0; i < 3; i++) {
//       trr_Sides.emplace_back(trr_boundary_grid[i], trr_boundary_grid[i + 1]);
//     }
//     trr_Sides.emplace_back(trr_boundary_grid[3], trr_boundary_grid[0]);
//     // for (auto& seg1 : trr_Sides) {
//     //     cout << seg1 << endl;
//     // }

//     if(seg.p1.x == 1147748 && seg.p1.y == 5695000) {
//       cout << "hhhhhhhhhhhhhhh" << endl;
//     }
//     for(auto& side : trr_Sides) {
//       // DEBUG
//       // if (side == trr_Sides[3]) {
//       //     cout << "hhhhhhhhhhhhhhh" << endl;
//       // }
//       // DEBUG

//       Segment intersection = side.intersect(seg);
//       if(intersection.id != -1) {
//         return intersection;
//       }
//     }
//     Segment ret;
//     ret.id = -1;
//     return ret;
//   }
// };
// // 。
// class TreeNode : public GridPoint {
//  public:
//   int id;
//   int level = -1;
//   TreeNode* lc;
//   TreeNode* rc;
//   TreeNode* par;
//   FPOS grad;
//   FPOS history_grad;
//   TreeNode() {
//     zeroFPoint.SetZero();
//     id = -1;
//     x = -1;
//     y = -1;
//     lc = NULL;
//     rc = NULL;
//     par = NULL;
//     grad = zeroFPoint;
//     history_grad = zeroFPoint;
//   }
//   TreeNode(int _id, double _x, double _y) {
//     id = _id;
//     x = _x;
//     y = _y;
//     lc = NULL;
//     rc = NULL;
//     par = NULL;
//     grad = zeroFPoint;
//     history_grad = zeroFPoint;
//   }
//   void set_base(int _id, double _x, double _y) {
//     id = _id;
//     x = _x;
//     y = _y;
//   }
//   void set_lc(TreeNode* child) { lc = child; }
//   void set_rc(TreeNode* child) { rc = child; }
//   void set_par(TreeNode* p) { par = p; }
// };

// class TreeTopology {
//  public:
//   TreeNode root;
//   TreeNode tmp_root;  // used for refinement evaluation
//   int leafNumber;
//   int size;
//   // alglib::integer_2d_array& HC_result;
//   // unordered_map<int, TreeNode> id_treeNode;

//   TreeTopology() {};

//   TreeTopology(TreeNode root, int leafNumber, int size) {
//     this->root = root;
//     // this->tmp_root = NULL;
//     this->leafNumber = leafNumber;
//     this->size = size;
//   }
//   // void init(int leafNum, int sz);
//   // construct binary tree structure based on current Hierarchical clustering
//   // result void constructTree(bool modifyCurrentTree = false); int
//   // getSumOfDiameter(); randomly switch leaf nodes to reduce sum of diameter
//   // void refineStructure(int iter = 10000);
// };

// class Wire {
//  public:
//   GrSteiner startpoint;
//   GrSteiner endpoint;
//   int type;

//   Wire(GrSteiner startpoint, GrSteiner endpoint, int type) {
//     this->startpoint = startpoint;
//     this->endpoint = endpoint;
//     this->type = type;
//   }

//   friend inline std::ostream& operator<<(std::ostream& os, const Wire& gp) {
//     os << "(" << gp.startpoint << "(" << gp.startpoint.id << ")" << ","
//        << gp.endpoint << "(" << gp.endpoint.id << ")" << ")";
//     return os;
//   }
// };

// class Buffer {
//  public:
//   GrSteiner startpoint;
//   GrSteiner endpoint;
//   int type;
//   string direction;  // 用来画图

//   Buffer(GrSteiner startpoint, GrSteiner endpoint, int type,
//          string direction = "right") {
//     this->startpoint = startpoint;
//     this->endpoint = endpoint;
//     this->type = type;
//     this->direction = direction;
//   }
// };

// class Blockage {
//  public:
//   GridPoint lower_left;
//   GridPoint upper_left;
//   GridPoint upper_right;
//   GridPoint lower_right;

//   Blockage(GridPoint lower_left, GridPoint upper_left, GridPoint upper_right,
//            GridPoint lower_right) {
//     this->lower_left = lower_left;
//     this->upper_left = upper_left;
//     this->upper_right = upper_right;
//     this->lower_right = lower_right;
//   }
// };
// class Router {
//  public:
//   // int MAX_RUNTIME = 3600;  // test 1
//   // int NUM_TAPS = 4;        // test 1
//   // vector<TAP> taps;
//   int delayModel = 1;  // 。
//   int num_sinks = 0;
//   int wireTypeName = 0;        // 。
//   double unit_wire_res = 0;    // 。
//   double unit_wire_cap = 0;    // 。
//   vector< GridPoint > layout;  // 。
//   vector< int > my_source;
//   vector< double > vdd_sim;
//   double slew_limit;
//   double cap_limit;
//   vector< Blockage > blockages;
//   vector< vector< double > > wire_lib = {{0, 0.0117, 0.0234}};
//   vector< vector< double > > buf_lib;
//   vector< int > bufType = {0, 1, 2, 3, 4, 5, 6, 7, 8};

//   vector< Sink > sinks = {Sink()};

//   vector< Segment > vertexMS;         // set of segments  //。
//   vector< TRR > vertexTRR;            // 。
//   vector< double > vertexDistE;       // 。
//   vector< double > vertexDistE_2;     // 。
//   vector< TreeNode > treeNodes;       // 。
//   vector< TreeNode > treeNodesForDP;  // 。

//   TreeTopology topo;

//   vector< GridPoint > pl;
//   vector< GrSteiner > sol;
//   // vector<Wire> wires;
//   // vector<Buffer> buffers;
//   // vector<GrSteiner > res_nodes;
//   vector< pair< int, int > > res_lines;

//   double distance = 700000;
//   int small_type = 0;
//   int largeType = 2;
//   double firstBufNeedCap = 230;
//   int internal_num = 0;   // 。
//   int gr_node_size = -1;  // 。
//   double totalClkWL = 0;  // 。
//   int mode = 0;
//   // Structures to store the final routing result
//   // GridPoint clockSource;

//   // vector<vector<GridPoint>> sol;

//   // void init();
//   // void readInput();
//   // To generate topology(try L1 based metric and L2 based ones)
//   // void NS();          // nearest neighbor topology
//   // void HC();          // Agglomerative  Hierarchical Clustering
//   // void CL();          // clustering based algorithm
//   // void Refinement();  // clustering with refinement 4<= m <=6 in the
//   original
//   // paper void RGM();         // Recursive Geometric Matching

//   // Generate embedding
//   Router() {};
//   void DME();  // Deferred-Merge Embedding
//   TreeNode HTree(std::vector< Sink > ffs, int& idCounter);  // th
//   std::pair< double, double > RootDME();
//   void route();
//   std::pair< double, double > rootRoute();
//   void buildSolution();
//   // void reportTotalWL();
//   void writeSolution();
//   void assignLevels(TreeNode* root);
//   TreeNode bottom_up(vector< TreeNode > nodesForMerge);
//   void bottom_up2();
//   void bottom_up3();
//   FPOS merge2(TreeNode* left, TreeNode* right);
//   void updateCor(TreeNode* node);

//   vector< TreeNode > findNearestPair(vector< TreeNode > nodesForMerge);
//   TreeNode merge(vector< TreeNode > pair);
//   void remove_pair(vector< TreeNode >& nodes, vector< TreeNode > pair);
//   void calculate_buf_dist(TreeNode curNode);

//   double buf_dist(int curId, Segment ms, double e_dist, int flag);
//   double findFirstBufDistance(double cap);
//   void calculate_polarity(Segment& ms_v, Segment ms_a, Segment ms_b, double
//   m1,
//                           double m2);
//   void preOrderTraversal(TreeNode& root);
//   void preOrderTraversalSolution(TreeNode& curNode);
//   void wireSolutionWithoutSnaking(GrSteiner curSteiner, GrSteiner
//   childSteiner,
//                                   int lr);
//   void insert_buf(vector< GrSteiner > nodes, int lr);
//   vector< GrSteiner > find_buf(vector< GrSteiner > nodes, int d);
//   void wireSolutionWithSnaking(GrSteiner curSteiner, GrSteiner childSteiner,
//                                int lr);
//   void add_buf(GrSteiner startpoint, GrSteiner endpoint, int buf_type,
//                string direction = "right");
//   void sol_normalization();
//   int gen_id();
//   bool isAroundBlockage(Segment ms_v, Segment ms_a, Segment ms_b);
//   double merging_cost(TreeNode p1, TreeNode p2);
//   double merging_cost2(double x1, double y1, double x2, double y2);
//   // vector<Wire> maze_route(GridPoint l_1, GridPoint l_2, double e_a_dist,
//   // double e_b_dist, int id_1, int id_2);
//   void buildLineSol();
//   void buildTreeLine(TreeNode* curNode);
//   FPOS getGrad(double x1, double y1, double x2, double y2);
//   FPOS getGrad2(double x1, double y1, double x2, double y2);
// };

#pragma once
#include "global.h"
// STL libraries
#include <iostream>
#include <fstream>
#include <cassert>
#include <iomanip>
#include <sstream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <cmath>
#include <functional>
#include <queue>
// #include "stdafx.h"
using namespace std;

const double eps = 1e-1;

using namespace std;

double round(double number, unsigned int bits);

class PointPair {
  // V_{i,k,n}
 public:
  double x1, x2, y1, y2;  // from:(x1,y1), to: (x2,y2)
  PointPair() {}
  PointPair(double _x1, double _x2, double _y1, double _y2)
      : x1(_x1), y1(_y1), x2(_x2), y2(_y2) {}
  bool operator==(PointPair const& var) const {
    return (x1 == var.x1 && y1 == var.y1 && x2 == var.x2 && y2 == var.y2);
  }
};

class GrSteiner;
// 要
class GridPoint {
 public:
  double x, y;

  GridPoint() {}
  GridPoint(double _x, double _y) : x(_x), y(_y) {}

  bool operator==(const GridPoint& rhs) const {
    return (x == rhs.x && y == rhs.y);
  }
  friend inline std::ostream& operator<<(std::ostream& os,
                                         const GridPoint& gp) {
    os << "(" << gp.x << "," << gp.y << ")";
    return os;
  }
};

class Sink : public GridPoint {
 public:
  int id;
  string name;
  double cap;
  Sink() {
    id = -1;
    x = -1;
    y = -1;
    cap = -1;
    name = "";
  }
  Sink(int _id, double _x, double _y, double _cap, string _name) {
    id = _id;
    x = _x;
    y = _y;
    cap = _cap;
    name = _name;
  }

  string str_xy() {
    stringstream s;
    s << "[" << x << "," << y << "]";
    return s.str();
  }
  friend inline std::ostream& operator<<(std::ostream& os, const Sink& sink) {
    // os << "Sink " << sink.id << ", x: " << fixed << setprecision(0) << sink.x
    // << ", y: " << sink.y;
    os << sink.id << " " << fixed << setprecision(5) << sink.x << " " << sink.y;
    return os;
  }
};

class GrSteiner : public GridPoint {
 public:
  GrSteiner* lc;
  GrSteiner* rc;
  GrSteiner* par;
  double last_len;
  int id;

  GrSteiner() {};
  GrSteiner(GridPoint p) {
    x = p.x;
    y = p.y;
    lc = NULL;
    rc = NULL;
    par = NULL;
    last_len = 0;
    id = -1;
  }

  void set_lc(GrSteiner* child) { lc = child; }
  void set_rc(GrSteiner* child) { rc = child; }
  void set_par(GrSteiner* p) { par = p; }
  void set_last_len(double len) { last_len = len; }
};
// 。
class Segment {
 public:
  GridPoint p1, p2;  // p1 has lower y
  // GridPoint center;
  // vector<Segment> ch[2];
  // vector<Segment> par;
  int id = 0;  // unique id for each segment
  double delay;
  double cap;
  vector< double > dis_a;
  vector< double > dis_b;
  vector< GrSteiner > left_wire_nodes;
  vector< GrSteiner > right_wire_nodes;
  double last_len_a = 0;
  double last_len_b = 0;
  int bufNumInPath = 0;
  bool isCalculate = false;

  Segment() {}
  Segment(GridPoint u, GridPoint v) : p1(u), p2(v) {
    delay = 0;
    if(p1.y > p2.y) {
      swap(p1, p2);
    }
  }

  double get_last_len() { return max(last_len_a, last_len_b); }

  bool isLeaf() {
    if(abs(p1.x - p2.x) < 1e-5 && abs(p1.y - p2.y) < 1e-5)
      return true;
    else
      return false;
  }

  bool operator==(const Segment& rhs) const {
    return (p1 == rhs.p1 && p2 == rhs.p2);
  }

  friend inline std::ostream& operator<<(std::ostream& os, const Segment& seg) {
    os << "Seg: (" << seg.p1 << "," << seg.p2 << ")";
    return os;
  }

  double slope() {
    if(isLeaf()) {
      return 0;
    }
    if(abs(p2.x - p1.x) < 1e-5) {
      return 0;
    }
    return round(1.0 * (p1.y - p2.y) / (p1.x - p2.x));
  }

  Segment intersect(Segment& rhs) {
    // DEBUG
    // if (rhs.p1.x == 1147748 && rhs.p1.y == 5695000) {
    //     cout << "hhhhhhhhhhhh" << endl;
    //     cout << "zzzzzzzzzzzzz" << endl;
    // }
    // DEBUG

    double cur_slope = slope();
    double rhs_slope = rhs.slope();

    // check if 4 points same line
    // if (abs(cur_slope - rhs_slope) < eps) {
    //     if (abs((rhs.p1.y - p1.y) * (p2.x - p1.x) - (p2.y - p1.y) * (rhs.p1.x
    //     - p1.x)) < eps) {
    if(rhs.isLeaf()) {  // if current segment is intersecting a single grid
                        // point
      Segment ret;
      if(abs((rhs.p1.y - p1.y) * (p2.x - p1.x) -
             (p2.y - p1.y) * (rhs.p1.x - p1.x)) < eps) {
        // if ((rhs.p1.y - p1.y) * (p2.x - p1.x) == (p2.y - p1.y) * (rhs.p1.x -
        // p1.x)) {
        if(p1.y - eps <= rhs.p1.y &&
           rhs.p1.y <= p2.y + eps) {  // valid intersection
          Segment ret = rhs;
          ret.id = -2;  // return single point intersection

          ret.p1.x = round(ret.p1.x);
          ret.p1.y = round(ret.p1.y);
          ret.p2.x = round(ret.p2.x);
          ret.p2.y = round(ret.p2.y);

          return ret;
        }
      }
      ret.id = -1;
      return ret;
    }
    if(abs(cur_slope - rhs_slope) < eps) {
      if(abs((rhs.p1.y - p1.y) * (p2.x - p1.x) -
             (p2.y - p1.y) * (rhs.p1.x - p1.x)) < eps) {
        assert(rhs.p1.y <= rhs.p2.y && p1.y <= p2.y);
        GridPoint upper, lower;
        if(rhs.p2.y < p2.y) {
          upper = rhs.p2;
        }
        else {
          upper = p2;
        }
        if(rhs.p1.y > p1.y) {
          lower = rhs.p1;
        }
        else {
          lower = p1;
        }
        if(upper.y < lower.y) {
          Segment ret;
          ret.id = -1;
          return ret;

          // cout << "No overlap between two segs on the line" << endl;
          // exit(1);
        }

        lower.x = round(lower.x);
        lower.y = round(lower.y);
        upper.x = round(upper.x);
        upper.y = round(upper.y);

        return Segment(lower, upper);
      }
      else {
        Segment ret;
        ret.id = -1;
        return ret;
      }
    }
    else {
      // might be 1 point or 0
      double A1 = p2.y - p1.y;
      // double B1 = p2.x - p1.x;
      double B1 = p1.x - p2.x;
      double C1 = A1 * p1.x + B1 * p1.y;
      double A2 = rhs.p2.y - rhs.p1.y;
      // double B2 = rhs.p2.x - rhs.p1.x;
      double B2 = rhs.p1.x - rhs.p2.x;
      double C2 = A2 * rhs.p1.x + B2 * rhs.p1.y;
      double det = A1 * B2 - A2 * B1;

      if(det == 0) {
        Segment ret;
        if(p1.x - p2.x < eps && p1.y - p2.y < eps) {
          if(p1.x - rhs.p1.x < eps && p1.y - rhs.p1.y < eps) {
            ret.p1 = GridPoint(rhs.p1.x, rhs.p1.y);
            ret.p2 = GridPoint(rhs.p1.x, rhs.p1.y);
            ret.id = -2;
          }
          else if(p1.x - rhs.p2.x < eps && p1.y - rhs.p2.y < eps) {
            ret.p1 = GridPoint(rhs.p2.x, rhs.p2.y);
            ret.p2 = GridPoint(rhs.p2.x, rhs.p2.y);
            ret.id = -2;
          }
          else
            ret.id = -1;
        }
        return ret;
      }

      double x = (B2 * C1 - B1 * C2) / det;
      double y = (A1 * C2 - A2 * C1) / det;
      x = round(x);
      y = round(y);

      Segment ret;

      if(p1.y - eps <= y && y <= p2.y + eps && rhs.p1.y - eps <= y &&
         y <= rhs.p2.y + eps) {  // valid intersection
        ret.p1 = GridPoint(x, y);
        ret.p2 = GridPoint(x, y);
        ret.id = -2;  // return single point intersection
      }
      else {
        ret.id = -1;
      }
      return ret;
    }
    // if return with id=-1, means no intersection
  }
};
class TRR {
 public:
  Segment core;
  double radius;
  TRR() {}
  TRR(Segment seg, double radi) : core(seg), radius(radi) {}
  friend inline std::ostream& operator<<(std::ostream& os, const TRR& trr) {
    os << trr.core << "; radius:" << trr.radius;
    return os;
  }
  Segment intersect(Segment& seg) {
    vector< GridPoint > trr_boundary_grid;
    vector< Segment > trr_Sides;
    trr_boundary_grid.emplace_back(core.p1.x, round(core.p1.y - radius));
    trr_boundary_grid.emplace_back(round(core.p2.x + radius), core.p2.y);
    trr_boundary_grid.emplace_back(core.p2.x, round(core.p2.y + radius));
    trr_boundary_grid.emplace_back(round(core.p1.x - radius),
                                   core.p1.y);  // clock-wise
    for(int i = 0; i < 3; i++) {
      trr_Sides.emplace_back(trr_boundary_grid[i], trr_boundary_grid[i + 1]);
    }
    trr_Sides.emplace_back(trr_boundary_grid[3], trr_boundary_grid[0]);
    // for (auto& seg1 : trr_Sides) {
    //     cout << seg1 << endl;
    // }

    if(seg.p1.x == 1147748 && seg.p1.y == 5695000) {
      cout << "hhhhhhhhhhhhhhh" << endl;
    }
    for(auto& side : trr_Sides) {
      // DEBUG
      // if (side == trr_Sides[3]) {
      //     cout << "hhhhhhhhhhhhhhh" << endl;
      // }
      // DEBUG

      Segment intersection = side.intersect(seg);
      if(intersection.id != -1) {
        return intersection;
      }
    }
    Segment ret;
    ret.id = -1;
    return ret;
  }
};
// 。
class TreeNode : public GridPoint {
 public:
  int id;
  TreeNode* lc;
  TreeNode* rc;
  TreeNode* par;
  FPOS grad;
  FPOS history_grad;
  TreeNode() {
    zeroFPoint.SetZero();
    id = -1;
    x = -1;
    y = -1;
    lc = NULL;
    rc = NULL;
    par = NULL;
    grad = zeroFPoint;
    history_grad = zeroFPoint;
  }
  TreeNode(int _id, double _x, double _y) {
    id = _id;
    x = _x;
    y = _y;
    lc = NULL;
    rc = NULL;
    par = NULL;
    grad = zeroFPoint;
    history_grad = zeroFPoint;
  }
  void set_base(int _id, double _x, double _y) {
    id = _id;
    x = _x;
    y = _y;
  }
  void set_lc(TreeNode* child) { lc = child; }
  void set_rc(TreeNode* child) { rc = child; }
  void set_par(TreeNode* p) { par = p; }
};

class TreeTopology {
 public:
  TreeNode root;
  TreeNode tmp_root;  // used for refinement evaluation
  int leafNumber;
  int size;
  // alglib::integer_2d_array& HC_result;
  // unordered_map<int, TreeNode> id_treeNode;

  TreeTopology() {};

  TreeTopology(TreeNode root, int leafNumber, int size) {
    this->root = root;
    // this->tmp_root = NULL;
    this->leafNumber = leafNumber;
    this->size = size;
  }
  // void init(int leafNum, int sz);
  // construct binary tree structure based on current Hierarchical clustering
  // result void constructTree(bool modifyCurrentTree = false); int
  // getSumOfDiameter(); randomly switch leaf nodes to reduce sum of diameter
  // void refineStructure(int iter = 10000);
};

class Wire {
 public:
  GrSteiner startpoint;
  GrSteiner endpoint;
  int type;

  Wire(GrSteiner startpoint, GrSteiner endpoint, int type) {
    this->startpoint = startpoint;
    this->endpoint = endpoint;
    this->type = type;
  }

  friend inline std::ostream& operator<<(std::ostream& os, const Wire& gp) {
    os << "(" << gp.startpoint << "(" << gp.startpoint.id << ")" << ","
       << gp.endpoint << "(" << gp.endpoint.id << ")" << ")";
    return os;
  }
};

class Buffer {
 public:
  GrSteiner startpoint;
  GrSteiner endpoint;
  int type;
  string direction;  // 用来画图

  Buffer(GrSteiner startpoint, GrSteiner endpoint, int type,
         string direction = "right") {
    this->startpoint = startpoint;
    this->endpoint = endpoint;
    this->type = type;
    this->direction = direction;
  }
};

class Blockage {
 public:
  GridPoint lower_left;
  GridPoint upper_left;
  GridPoint upper_right;
  GridPoint lower_right;

  Blockage(GridPoint lower_left, GridPoint upper_left, GridPoint upper_right,
           GridPoint lower_right) {
    this->lower_left = lower_left;
    this->upper_left = upper_left;
    this->upper_right = upper_right;
    this->lower_right = lower_right;
  }
};
class Router {
 public:
  // int MAX_RUNTIME = 3600;  // test 1
  // int NUM_TAPS = 4;        // test 1
  // vector<TAP> taps;
  int delayModel = 1;  // 。
  int num_sinks = 0;
  int wireTypeName = 0;        // 。
  double unit_wire_res = 0;    // 。
  double unit_wire_cap = 0;    // 。
  vector< GridPoint > layout;  // 。
  vector< int > my_source;
  vector< double > vdd_sim;
  double slew_limit;
  double cap_limit;
  vector< Blockage > blockages;
  vector< vector< double > > wire_lib = {{0, 0.0117, 0.0234}};
  vector< vector< double > > buf_lib;
  vector< int > bufType = {0, 1, 2, 3, 4, 5, 6, 7, 8};

  vector< Sink > sinks = {Sink()};

  vector< Segment > vertexMS;         // set of segments  //。
  vector< TRR > vertexTRR;            // 。
  vector< double > vertexDistE;       // 。
  vector< double > vertexDistE_2;     // 。
  vector< TreeNode > treeNodes;       // 。
  vector< TreeNode > treeNodesForDP;  // 。

  TreeTopology topo;

  vector< GridPoint > pl;
  vector< GrSteiner > sol;
  // vector<Wire> wires;
  // vector<Buffer> buffers;
  // vector<GrSteiner > res_nodes;
  vector< pair< int, int > > res_lines;

  double distance = 700000;
  int small_type = 0;
  int largeType = 2;
  double firstBufNeedCap = 230;
  int internal_num = 0;   // 。
  int gr_node_size = -1;  // 。
  double totalClkWL = 0;  // 。
  int mode = 0;
  // Structures to store the final routing result
  // GridPoint clockSource;

  // vector<vector<GridPoint>> sol;

  // void init();
  // void readInput();
  // To generate topology(try L1 based metric and L2 based ones)
  // void NS();          // nearest neighbor topology
  // void HC();          // Agglomerative  Hierarchical Clustering
  // void CL();          // clustering based algorithm
  // void Refinement();  // clustering with refinement 4<= m <=6 in the original
  // paper void RGM();         // Recursive Geometric Matching

  // Generate embedding
  Router() {};
  void DME();  // Deferred-Merge Embedding
  std::pair< double, double > RootDME();
  void route();
  std::pair< double, double > rootRoute();
  void buildSolution();
  // void reportTotalWL();
  void writeSolution();

  TreeNode bottom_up(vector< TreeNode > nodesForMerge);
  void bottom_up2();
  void bottom_up3();
  FPOS merge2(TreeNode* left, TreeNode* right);
  void updateCor(TreeNode* node);

  vector< TreeNode > findNearestPair(vector< TreeNode > nodesForMerge);
  TreeNode merge(vector< TreeNode > pair);
  void remove_pair(vector< TreeNode >& nodes, vector< TreeNode > pair);
  void calculate_buf_dist(TreeNode curNode);

  double buf_dist(int curId, Segment ms, double e_dist, int flag);
  double findFirstBufDistance(double cap);
  void calculate_polarity(Segment& ms_v, Segment ms_a, Segment ms_b, double m1,
                          double m2);
  void preOrderTraversal(TreeNode& root);
  void preOrderTraversalSolution(TreeNode& curNode);
  void wireSolutionWithoutSnaking(GrSteiner curSteiner, GrSteiner childSteiner,
                                  int lr);
  void insert_buf(vector< GrSteiner > nodes, int lr);
  vector< GrSteiner > find_buf(vector< GrSteiner > nodes, int d);
  void wireSolutionWithSnaking(GrSteiner curSteiner, GrSteiner childSteiner,
                               int lr);
  void add_buf(GrSteiner startpoint, GrSteiner endpoint, int buf_type,
               string direction = "right");
  void sol_normalization();
  int gen_id();
  bool isAroundBlockage(Segment ms_v, Segment ms_a, Segment ms_b);
  double merging_cost(TreeNode p1, TreeNode p2);
  double merging_cost2(double x1, double y1, double x2, double y2);
  // vector<Wire> maze_route(GridPoint l_1, GridPoint l_2, double e_a_dist,
  // double e_b_dist, int id_1, int id_2);
  void buildLineSol();
  void buildTreeLine(TreeNode* curNode);
  FPOS getGrad(double x1, double y1, double x2, double y2);
  FPOS getGrad2(double x1, double y1, double x2, double y2);
};