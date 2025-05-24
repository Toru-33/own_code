// lxm:level version

// #include "Router.h"
// #include <climits>
// // #include "GenTree.h"
// #include "global.h"
// // #include "mkl.h"
// #include "CTSTree.h"
// #include <chrono>

// using std::cout;
// using std::endl;
// using std::setprecision;

// #define COMPLETE_LINKAGE 0
// #define SINGLE_LINKAGE 1
// #define L1 1
// #define L2 2
// #define eps 1e-6
// const string padding(30, '=');

// double round(double number, unsigned int bits) {
//   stringstream ss;
//   ss << fixed << setprecision(bits) << number;
//   ss >> number;
//   return number;
// }

// static int id = 2000;
// static int NEG_MAX_EXP = -300;

// int genID() { return id++; }

// int getID() { return id; }

// enum class Direction { EMPTY, VERTICAL, HRIZONTAL };

// template < class T >
// class Point {
//  public:
//   T x, y;
//   Point(T x, T y) : x(x), y(y) {}
//   Point() : x(-1), y(-1) {}
//   bool operator==(const Point& rhs) const { return (x == rhs.x && y ==
//   rhs.y); }
// };

// class GridState {
//   short value;
//   GridState(int value) : value(value) {}

//  public:
//   // empty 000, start 001, end 010, block 100
//   static const GridState EMPTY, START, END;

//   GridState(const GridState& s) { value = s.value; }
//   bool isMet() { return value == 3; }
//   bool canVisited() { return value != 1; }
//   void addState(const GridState s) { value |= s.value; }
//   void reset() { value &= 4; }
// };
// const GridState GridState::EMPTY = 0;
// const GridState GridState::START = 1;
// const GridState GridState::END = 2;

// using PointInt = Point< int >;

// class RouteResource {
//   vector< vector< PointInt > > parents;
//   vector< vector< Direction > > dirs;
//   vector< vector< int > > turns;
//   vector< vector< bool > > visited;
//   PointInt cur;
//   queue< PointInt > q;

//  public:
//   RouteResource() : cur(-1, -1) {}
//   PointInt getCur() { return cur; }
//   void construct(int width, int height) {
//     parents.resize(width);
//     for(auto& v : parents) {
//       v.resize(height);
//     }
//     dirs.resize(width);
//     for(auto& v : dirs) {
//       v.resize(height);
//     }
//     turns.resize(width);
//     for(auto& v : turns) {
//       v.resize(height);
//     }
//     visited = std::move(
//         vector< vector< bool > >(width, vector< bool >(height, false)));
//     cur = PointInt(-1, -1);

//     queue< PointInt > tmp;
//     swap(q, tmp);
//   }
//   void init(PointInt start, Direction dir) {
//     for(auto& v : visited) {
//       std::fill(v.begin(), v.end(), false);
//     }
//     cur = PointInt(-1, -1);
//     queue< PointInt > tmp;
//     swap(q, tmp);

//     q.push(start);
//     parents[start.x][start.y] = PointInt(-1, -1);
//     dirs[start.x][start.y] = dir;
//     turns[start.x][start.y] = 0;
//     visited[start.x][start.y] = true;
//   }
//   PointInt next() {
//     for(auto p = parents[cur.x][cur.y]; p.x >= 0 && p.y >= 0;
//         cur = p, p = parents[cur.x][cur.y]) {
//       if(dirs[p.x][p.y] != dirs[cur.x][cur.y]) {
//         cur = p;
//         return p;
//       }
//     }
//     return cur;
//   }
//   bool walk(vector< vector< GridState > >& states,
//             vector< vector< double > >& h_edges,
//             vector< vector< double > >& v_edges) {
//     int num = q.size();
//     assert(num != 0);
//     for(int i = 0; i < num; ++i) {
//       auto p = q.front();
//       q.pop();

//       auto& st = states[p.x][p.y];
//       st.addState(GridState::START);

//       if(!st.isMet()) {
//         if(p.x > 0 && states[p.x - 1][p.y].canVisited() &&
//            h_edges[p.x - 1][p.y] > 0.0) {
//           int turn = dirs[p.x][p.y] == Direction::HRIZONTAL
//                          ? turns[p.x][p.y]
//                          : turns[p.x][p.y] + 1;

//           if(!visited[p.x - 1][p.y] || turns[p.x - 1][p.y] > turn) {
//             if(!visited[p.x - 1][p.y]) {
//               visited[p.x - 1][p.y] = true;
//               q.emplace(p.x - 1, p.y);
//             }
//             dirs[p.x - 1][p.y] = Direction::HRIZONTAL;
//             turns[p.x - 1][p.y] = turn;
//             parents[p.x - 1][p.y] = PointInt(p.x, p.y);
//           }
//         }
//         if(p.x < states.size() - 1 && states[p.x + 1][p.y].canVisited() &&
//            h_edges[p.x][p.y] > 0.0) {
//           int turn = dirs[p.x][p.y] == Direction::HRIZONTAL
//                          ? turns[p.x][p.y]
//                          : turns[p.x][p.y] + 1;

//           if(!visited[p.x + 1][p.y] || turns[p.x + 1][p.y] > turn) {
//             if(!visited[p.x + 1][p.y]) {
//               visited[p.x + 1][p.y] = true;
//               q.emplace(p.x + 1, p.y);
//             }
//             dirs[p.x + 1][p.y] = Direction::HRIZONTAL;
//             turns[p.x + 1][p.y] = turn;
//             parents[p.x + 1][p.y] = PointInt(p.x, p.y);
//           }
//         }
//         if(p.y > 0 && states[p.x][p.y - 1].canVisited() &&
//            v_edges[p.x][p.y - 1] > 0.0) {
//           int turn = dirs[p.x][p.y] == Direction::VERTICAL
//                          ? turns[p.x][p.y]
//                          : turns[p.x][p.y] + 1;

//           if(!visited[p.x][p.y - 1] || turns[p.x][p.y - 1] > turn) {
//             if(!visited[p.x][p.y - 1]) {
//               visited[p.x][p.y - 1] = true;
//               q.emplace(p.x, p.y - 1);
//             }
//             dirs[p.x][p.y - 1] = Direction::VERTICAL;
//             turns[p.x][p.y - 1] = turn;
//             parents[p.x][p.y - 1] = PointInt(p.x, p.y);
//           }
//         }
//         if(p.y < states[0].size() - 1 && states[p.x][p.y + 1].canVisited() &&
//            v_edges[p.x][p.y] > 0.0) {
//           int turn = dirs[p.x][p.y] == Direction::VERTICAL
//                          ? turns[p.x][p.y]
//                          : turns[p.x][p.y] + 1;

//           if(!visited[p.x][p.y + 1] || turns[p.x][p.y + 1] > turn) {
//             if(!visited[p.x][p.y + 1]) {
//               visited[p.x][p.y + 1] = true;
//               q.emplace(p.x, p.y + 1);
//             }
//             dirs[p.x][p.y + 1] = Direction::VERTICAL;
//             turns[p.x][p.y + 1] = turn;
//             parents[p.x][p.y + 1] = PointInt(p.x, p.y);
//           }
//         }
//       }
//       else {
//         cur = p;
//         return true;
//       }
//     }
//     return false;
//   }
// };

// // 。
// inline double L1Dist(GridPoint p1, GridPoint p2) {
//   return abs(p1.x - p2.x) + abs(p1.y - p2.y);
// }
// inline double L2Dist(TreeNode p1, TreeNode p2) {
//   return abs(p1.x - p2.x) + abs(p1.y - p2.y);
// }
// inline double L3Dist(GrSteiner p1, GrSteiner p2) {
//   return abs(p1.x - p2.x) + abs(p1.y - p2.y);
// }

// // 。
// Segment TRRintersect(TRR& trr1, TRR& trr2) {
//   // get four edges
//   // cout << "Merging: " << trr1 << " and " << trr2 << endl;
//   vector< GridPoint > trr1_boundary_grid;
//   vector< GridPoint > trr2_boundary_grid;
//   vector< Segment > trr1_Sides;
//   vector< Segment > trr2_Sides;
//   if(trr1.core.slope() > 0) {
//     trr1_boundary_grid.emplace_back(trr1.core.p1.x,
//                                     trr1.core.p1.y - trr1.radius);
//     trr1_boundary_grid.emplace_back(trr1.core.p2.x + trr1.radius,
//                                     trr1.core.p2.y);
//     trr1_boundary_grid.emplace_back(trr1.core.p2.x,
//                                     trr1.core.p2.y + trr1.radius);
//     trr1_boundary_grid.emplace_back(trr1.core.p1.x - trr1.radius,
//                                     trr1.core.p1.y);  // clock-wise
//   }
//   else if(trr1.core.slope() < 0) {
//     trr1_boundary_grid.emplace_back(trr1.core.p1.x + trr1.radius,
//                                     trr1.core.p1.y);
//     trr1_boundary_grid.emplace_back(trr1.core.p2.x,
//                                     trr1.core.p2.y + trr1.radius);
//     trr1_boundary_grid.emplace_back(trr1.core.p2.x - trr1.radius,
//                                     trr1.core.p2.y);
//     trr1_boundary_grid.emplace_back(
//         trr1.core.p1.x,
//         trr1.core.p1.y - trr1.radius);  // clock-wise
//   }
//   else {  // leaf node
//     trr1_boundary_grid.emplace_back(trr1.core.p1.x,
//                                     trr1.core.p1.y - trr1.radius);
//     trr1_boundary_grid.emplace_back(trr1.core.p1.x + trr1.radius,
//                                     trr1.core.p1.y);
//     trr1_boundary_grid.emplace_back(trr1.core.p1.x,
//                                     trr1.core.p1.y + trr1.radius);
//     trr1_boundary_grid.emplace_back(trr1.core.p1.x - trr1.radius,
//                                     trr1.core.p1.y);  // clock-wise
//   }

//   if(trr2.core.slope() > 0) {
//     trr2_boundary_grid.emplace_back(trr2.core.p1.x,
//                                     trr2.core.p1.y - trr2.radius);
//     trr2_boundary_grid.emplace_back(trr2.core.p2.x + trr2.radius,
//                                     trr2.core.p2.y);
//     trr2_boundary_grid.emplace_back(trr2.core.p2.x,
//                                     trr2.core.p2.y + trr2.radius);
//     trr2_boundary_grid.emplace_back(trr2.core.p1.x - trr2.radius,
//                                     trr2.core.p1.y);  // clock-wise
//   }
//   else if(trr2.core.slope() < 0) {
//     trr2_boundary_grid.emplace_back(trr2.core.p1.x + trr2.radius,
//                                     trr2.core.p1.y);
//     trr2_boundary_grid.emplace_back(trr2.core.p2.x,
//                                     trr2.core.p2.y + trr2.radius);
//     trr2_boundary_grid.emplace_back(trr2.core.p2.x - trr2.radius,
//                                     trr2.core.p2.y);
//     trr2_boundary_grid.emplace_back(
//         trr2.core.p1.x,
//         trr2.core.p1.y - trr2.radius);  // clock-wise
//   }
//   else {  // leaf node
//     trr2_boundary_grid.emplace_back(trr2.core.p1.x,
//                                     trr2.core.p1.y - trr2.radius);
//     trr2_boundary_grid.emplace_back(trr2.core.p1.x + trr2.radius,
//                                     trr2.core.p1.y);
//     trr2_boundary_grid.emplace_back(trr2.core.p1.x,
//                                     trr2.core.p1.y + trr2.radius);
//     trr2_boundary_grid.emplace_back(trr2.core.p1.x - trr2.radius,
//                                     trr2.core.p1.y);  // clock-wise
//   }

//   for(int i = 0; i < 4; i++) {
//     trr1_boundary_grid[i].x = round(trr1_boundary_grid[i].x, 5);
//     trr1_boundary_grid[i].y = round(trr1_boundary_grid[i].y, 5);
//     trr2_boundary_grid[i].x = round(trr2_boundary_grid[i].x, 5);
//     trr2_boundary_grid[i].y = round(trr2_boundary_grid[i].y, 5);
//   }

//   for(int i = 0; i < 3; i++) {
//     trr1_Sides.emplace_back(trr1_boundary_grid[i], trr1_boundary_grid[i +
//     1]); trr2_Sides.emplace_back(trr2_boundary_grid[i], trr2_boundary_grid[i
//     + 1]);
//   }
//   trr1_Sides.emplace_back(trr1_boundary_grid[3], trr1_boundary_grid[0]);
//   trr2_Sides.emplace_back(trr2_boundary_grid[3], trr2_boundary_grid[0]);

//   vector< Segment > segList;
//   for(auto& seg1 : trr1_Sides) {
//     for(auto& seg2 : trr2_Sides) {
//       Segment seg = seg1.intersect(seg2);
//       if(seg.id != -1) {
//         segList.emplace_back(seg);
//       }
//     }
//   }

//   if(segList.size() == 0) {
//     // cout << "Cannot find intersection between two TRRs" << endl;
//     Segment ret;
//     ret.id = -1;
//     return ret;
//   }

//   int seglab = -1;
//   for(auto& seg : segList) {
//     seglab += 1;
//     if(seg.id != -2) {
//       return seg;
//     }
//   }

//   return segList[seglab];
// }

// // 。
// TreeNode Router::merge(vector< TreeNode > pair) {
//   // cout<<"totalClkWL1:"<<totalClkWL<<endl;
//   int node1_id = pair[0].id;
//   int node2_id = pair[1].id;
//   Segment ms_a, ms_b;

//   TreeNode tr_v;
//   if(node1_id <= num_sinks) {
//     ms_a = vertexMS[node1_id] = Segment(sinks[node1_id], sinks[node1_id]);
//     vertexMS[node1_id].cap = sinks[node1_id].cap;
//   }
//   else {
//     ms_a = vertexMS[node1_id];
//   }

//   if(node2_id <= num_sinks) {
//     ms_b = vertexMS[node2_id] = Segment(sinks[node2_id], sinks[node2_id]);
//     vertexMS[node2_id].cap = sinks[node2_id].cap;
//   }
//   else {
//     ms_b = vertexMS[node2_id];
//   }

//   double d, t_1;
//   d = t_1 = L1Dist(ms_a.p1, ms_b.p1);
//   GridPoint l_1 = ms_a.p1;
//   GridPoint l_2 = ms_b.p1;

//   double t_2 = L1Dist(ms_a.p1, ms_b.p2);
//   if(t_2 < d) {
//     l_1 = ms_a.p1;
//     l_2 = ms_b.p2;
//     d = t_2;
//   }

//   double t_3 = L1Dist(ms_a.p2, ms_b.p1);
//   if(t_3 < d) {
//     l_1 = ms_a.p2;
//     l_2 = ms_b.p1;
//     d = t_3;
//   }

//   double t_4 = L1Dist(ms_a.p2, ms_b.p2);
//   if(t_4 < d) {
//     l_1 = ms_a.p2;
//     l_2 = ms_b.p2;
//     d = t_4;
//   }

//   wireTypeName = wire_lib[0][0];
//   unit_wire_res = wire_lib[0][1];
//   unit_wire_cap = wire_lib[0][2];

//   double e_a_dist, e_b_dist;
//   TreeNode& lc = treeNodes[node1_id];
//   TreeNode& rc = treeNodes[node2_id];

//   TRR trr_a, trr_b;
//   Segment ms_v;

//   if(delayModel == 0) {
//     if(fabs(ms_b.delay - ms_a.delay) < d) {
//       double e_a_dist = (ms_b.delay - ms_a.delay + d) / 2;
//       double e_b_dist = (ms_a.delay - ms_b.delay + d) / 2;

//       lc = treeNodes[node1_id];
//       vertexDistE[lc.id] = e_a_dist;
//       vertexDistE_2[lc.id] = e_a_dist;

//       rc = treeNodes[node2_id];
//       vertexDistE[rc.id] = e_b_dist;
//       vertexDistE_2[rc.id] = e_b_dist;

//       vertexTRR[lc.id] = trr_a = TRR(ms_a, e_a_dist);
//       vertexTRR[rc.id] = trr_b = TRR(ms_b, e_b_dist);

//       ms_v = TRRintersect(trr_a, trr_b);

//       if(ms_v.id == -1) {
//         cout << "trr_a的四个点: [" << trr_a.core.p2.x << ", "
//              << (trr_a.core.p2.y + trr_a.radius) << "], ["
//              << (trr_a.core.p2.x + trr_a.radius) << ", " << trr_a.core.p2.y
//              << "], [" << trr_a.core.p1.x << ", "
//              << (trr_a.core.p1.y - trr_a.radius) << "], ["
//              << (trr_a.core.p1.x - trr_a.radius) << ", " << trr_a.core.p1.y
//              << "]" << endl;
//         cout << "trr_b的四个点: [" << trr_b.core.p2.x << ", "
//              << (trr_b.core.p2.y + trr_b.radius) << "], ["
//              << (trr_b.core.p2.x + trr_b.radius) << ", " << trr_b.core.p2.y
//              << "], [" << trr_b.core.p1.x << ", "
//              << (trr_b.core.p1.y - trr_b.radius) << "], ["
//              << (trr_b.core.p1.x - trr_b.radius) << ", " << trr_b.core.p1.y
//              << "]" << endl;

//         auto _p = GridPoint((l_1.x + l_2.x) / 2, (l_1.y + l_2.y) / 2);
//         ms_v = Segment(_p, _p);
//       }
//       ms_v.delay = e_a_dist + ms_a.delay;
//       internal_num += 1;
//       int curId = num_sinks + internal_num;
//       vertexMS[curId] = ms_v;

//       tr_v = TreeNode(curId, (ms_v.p1.x + ms_v.p2.x) / 2,
//                       (ms_v.p1.y + ms_v.p2.y) / 2);
//       tr_v.lc = &lc;
//       tr_v.rc = &rc;
//       lc.par = &tr_v;
//       rc.par = &tr_v;
//       treeNodes[curId] = tr_v;

//       totalClkWL += vertexDistE_2[lc.id];
//       totalClkWL += vertexDistE_2[rc.id];
//     }
//     else {
//       if(ms_a.delay <= ms_b.delay) {
//         e_a_dist = ms_b.delay - ms_a.delay;
//         e_b_dist = 0;

//         lc = treeNodes[node1_id];
//         vertexDistE[lc.id] = d;
//         vertexDistE_2[lc.id] = e_a_dist;

//         rc = treeNodes[node2_id];
//         vertexDistE[rc.id] = e_b_dist;
//         vertexDistE_2[rc.id] = e_b_dist;

//         vertexTRR[lc.id] = trr_a = TRR(ms_a, d);
//         vertexTRR[rc.id] = trr_b = TRR(ms_b, e_b_dist);

//         ms_v = Segment(l_2, l_2);

//         ms_v.delay = ms_b.delay;

//         internal_num += 1;
//         int curId = num_sinks + internal_num;
//         vertexMS[curId] = ms_v;

//         tr_v = TreeNode(curId, (ms_v.p1.x + ms_v.p2.x) / 2,
//                         (ms_v.p1.y + ms_v.p2.y) / 2);
//         tr_v.lc = &lc;
//         tr_v.rc = &rc;
//         lc.par = &tr_v;
//         rc.par = &tr_v;
//         treeNodes[curId] = tr_v;

//         totalClkWL += vertexDistE_2[lc.id];
//         totalClkWL += vertexDistE_2[rc.id];
//       }
//       else {
//         e_b_dist = ms_a.delay - ms_b.delay;
//         e_a_dist = 0;

//         lc = treeNodes[node1_id];
//         vertexDistE[lc.id] = e_a_dist;
//         vertexDistE_2[lc.id] = e_a_dist;

//         rc = treeNodes[node2_id];
//         vertexDistE[rc.id] = d;
//         vertexDistE_2[rc.id] = e_b_dist;

//         vertexTRR[lc.id] = TRR(ms_a, e_a_dist);
//         vertexTRR[rc.id] = TRR(ms_b, d);

//         ms_v = Segment(l_1, l_1);

//         ms_v.delay = ms_a.delay;

//         internal_num += 1;
//         int curId = num_sinks + internal_num;
//         vertexMS[curId] = ms_v;

//         tr_v = TreeNode(curId, (ms_v.p1.x + ms_v.p2.x) / 2,
//                         (ms_v.p1.y + ms_v.p2.y) / 2);
//         tr_v.lc = &lc;
//         tr_v.rc = &rc;
//         lc.par = &tr_v;
//         rc.par = &tr_v;
//         treeNodes[curId] = tr_v;

//         totalClkWL += vertexDistE_2[lc.id];
//         totalClkWL += vertexDistE_2[rc.id];
//       }
//     }
//   }

//   else if(delayModel == 1) {
//     double y = ms_b.delay - ms_a.delay +
//                unit_wire_res * d * (ms_b.cap + unit_wire_cap * d / 2);
//     double z = unit_wire_res * (ms_a.cap + ms_b.cap + unit_wire_cap * d);
//     double x = y / z;
//     x = round(x);

//     if(x < 0) {
//       double cap2 = ms_b.cap * unit_wire_res;
//       x = (sqrt(cap2 * cap2 +
//                 2 * unit_wire_res * unit_wire_cap * (ms_a.delay -
//                 ms_b.delay)) -
//            cap2) /
//           (unit_wire_res * unit_wire_cap);
//       e_a_dist = 0;
//       e_b_dist = round(x);

//       // lc = treeNodes[node1_id];
//       vertexDistE[lc.id] = e_a_dist;
//       vertexDistE_2[lc.id] = e_a_dist;

//       // rc = treeNodes[node2_id];
//       vertexDistE[rc.id] = d;
//       vertexDistE_2[rc.id] = e_b_dist;

//       vertexTRR[lc.id] = TRR(ms_a, e_a_dist);
//       vertexTRR[rc.id] = TRR(ms_b, d);

//       ms_v = Segment(l_1, l_1);

//       ms_v.delay = ms_a.delay;
//       ms_v.cap = ms_a.cap + ms_b.cap + x * unit_wire_cap;

//       internal_num += 1;
//       int curId = num_sinks + internal_num;
//       assert(curId <= num_sinks * 2 - 1);
//       vertexMS[curId] = ms_v;

//       tr_v = TreeNode(curId, (ms_v.p1.x + ms_v.p2.x) / 2,
//                       (ms_v.p1.y + ms_v.p2.y) / 2);
//       tr_v.lc = &lc;
//       tr_v.rc = &rc;
//       treeNodes[curId] = tr_v;
//       // treeNodes[node1_id].par = &treeNodes[curId];
//       // treeNodes[node2_id].par = &treeNodes[curId];
//       lc.par = &treeNodes[curId];
//       rc.par = &treeNodes[curId];
//       // treeNodes[lc.id].grad = getGrad2(lc.x, lc.y, tr_v.x, tr_v.y);
//       // treeNodes[rc.id].grad = getGrad2(rc.x, rc.y, tr_v.x, tr_v.y);
//       // lc.par = &tr_v;
//       // rc.par = &tr_v;
//       if(curId == (num_sinks * 2 - 1)) {
//         treeNodes[curId].par = NULL;
//       }

//       totalClkWL += vertexDistE_2[lc.id];
//       totalClkWL += vertexDistE_2[rc.id];
//       // cout<<"totalClkWL:"<<totalClkWL<<endl;
//       //  calculate_buf_dist(tr_v);
//     }
//     else if(x > d) {
//       double cap1 = ms_a.cap * unit_wire_cap;
//       x = (sqrt(cap1 * cap1 +
//                 2 * unit_wire_res * unit_wire_cap * (ms_b.delay -
//                 ms_a.delay)) -
//            cap1) /
//           (unit_wire_res * unit_wire_cap);
//       x = round(x);
//       e_a_dist = x;
//       e_b_dist = 0;

//       // lc = treeNodes[node1_id];
//       vertexDistE[lc.id] = d;
//       vertexDistE_2[lc.id] = e_a_dist;

//       // rc = treeNodes[node2_id];
//       vertexDistE[rc.id] = e_b_dist;
//       vertexDistE_2[rc.id] = e_b_dist;

//       vertexTRR[lc.id] = trr_a = TRR(ms_a, d);
//       vertexTRR[rc.id] = trr_b = TRR(ms_b, e_b_dist);

//       ms_v = Segment(l_2, l_2);

//       ms_v.delay = ms_b.delay;
//       ms_v.cap = ms_a.cap + ms_b.cap + x * unit_wire_cap;

//       internal_num += 1;
//       int curId = num_sinks + internal_num;
//       assert(curId <= num_sinks * 2 - 1);
//       vertexMS[curId] = ms_v;

//       tr_v = TreeNode(curId, (ms_v.p1.x + ms_v.p2.x) / 2,
//                       (ms_v.p1.y + ms_v.p2.y) / 2);
//       tr_v.lc = &lc;
//       tr_v.rc = &rc;
//       treeNodes[curId] = tr_v;
//       // treeNodes[node1_id].par = &treeNodes[curId];
//       // treeNodes[node2_id].par = &treeNodes[curId];
//       lc.par = &treeNodes[curId];
//       rc.par = &treeNodes[curId];
//       // treeNodes[lc.id].grad = getGrad2(lc.x, lc.y, tr_v.x, tr_v.y);
//       // treeNodes[rc.id].grad = getGrad2(rc.x, rc.y, tr_v.x, tr_v.y);
//       // lc.par = &tr_v;
//       // rc.par = &tr_v;/home/eda/lxm/cts_opendp_cluster/src
//       treeNodes[curId] = tr_v;
//       if(curId == (num_sinks * 2 - 1)) {
//         treeNodes[curId].par = NULL;
//       }

//       totalClkWL += vertexDistE_2[lc.id];
//       totalClkWL += vertexDistE_2[rc.id];

//       // cout<<"totalClkWL:"<<totalClkWL<<endl;
//       //  calculate_buf_dist(tr_v);
//     }
//     else {
//       e_a_dist = round(x);
//       e_b_dist = d - e_a_dist;

//       // lc = treeNodes[node1_id];
//       vertexDistE[lc.id] = e_a_dist;
//       vertexDistE_2[lc.id] = e_a_dist;

//       // rc = treeNodes[node2_id];
//       vertexDistE[rc.id] = e_b_dist;
//       vertexDistE_2[rc.id] = e_b_dist;

//       vertexTRR[lc.id] = trr_a = TRR(ms_a, e_a_dist);
//       vertexTRR[rc.id] = trr_b = TRR(ms_b, e_b_dist);

//       // ms_v = TRRintersect(trr_a, trr_b);

//       // if (ms_v.id == -1) {
//       // cout << "trr_a的四个点: [" << trr_a.core.p2.x << ", " <<
//       // (trr_a.core.p2.y + trr_a.radius) << "], ["
//       //     << (trr_a.core.p2.x + trr_a.radius) << ", " << trr_a.core.p2.y
//       <<
//       //     "], ["
//       //     << trr_a.core.p1.x << ", " << (trr_a.core.p1.y - trr_a.radius)
//       <<
//       //     "], ["
//       //     << (trr_a.core.p1.x - trr_a.radius) << ", " << trr_a.core.p1.y
//       <<
//       //     "]" << endl;
//       // cout << "trr_b的四个点: [" << trr_b.core.p2.x << ", " <<
//       // (trr_b.core.p2.y + trr_b.radius) << "], ["
//       //     << (trr_b.core.p2.x + trr_b.radius) << ", " << trr_b.core.p2.y
//       <<
//       //     "], ["
//       //     << trr_b.core.p1.x << ", " << (trr_b.core.p1.y - trr_b.radius)
//       <<
//       //     "], ["
//       //     << (trr_b.core.p1.x - trr_b.radius) << ", " << trr_b.core.p1.y
//       <<
//       //     "]" << endl;

//       // for (auto& sink : sinks) {
//       //     cout << sink << endl;
//       // }
//       // ms_v = TRRintersect(trr_a, trr_b);
//       auto _p = GridPoint((l_1.x + l_2.x) / 2, (l_1.y + l_2.y) / 2);
//       ms_v = Segment(_p, _p);
//       // cout << "Merge failure" << endl;
//       // exit(1);
//       // }

//       ms_v.delay =
//           unit_wire_res * e_a_dist * (ms_a.cap + unit_wire_cap * e_a_dist /
//           2) + ms_a.delay;
//       ms_v.delay = round(ms_v.delay, 5);
//       ms_v.cap = ms_a.cap + ms_b.cap + d * unit_wire_cap;

//       internal_num += 1;
//       int curId = num_sinks + internal_num;
//       assert(curId <= num_sinks * 2 - 1);
//       vertexMS[curId] = ms_v;

//       tr_v = TreeNode(curId, (ms_v.p1.x + ms_v.p2.x) / 2,
//                       (ms_v.p1.y + ms_v.p2.y) / 2);
//       tr_v.lc = &lc;
//       tr_v.rc = &rc;
//       treeNodes[curId] = tr_v;

//       lc.par = &treeNodes[curId];
//       rc.par = &treeNodes[curId];

//       treeNodes[curId] = tr_v;
//       if(curId == (num_sinks * 2 - 1)) {
//         treeNodes[curId].par = NULL;
//       }

//       totalClkWL += vertexDistE_2[lc.id];
//       totalClkWL += vertexDistE_2[rc.id];

//       // cout<<"totalClkWL:"<<totalClkWL<<endl;
//       //  calculate_buf_dist(tr_v);
//     }
//   }

//   return tr_v;
// }

// // 。
// TreeNode Router::bottom_up(vector< TreeNode > pts) {
//   static fzq::CTSTree gen;
//   std::vector< fzq::FPos >& sinks = gen.getSinks();
//   sinks.clear();
//   for(auto& v : pts) {
//     sinks.emplace_back(v.x, v.y, v.id);
//     // std::cout<<"v.x:"<<v.x<<"\tv.y:"<<v.y<<"\tv.id:"<<v.id<<std::endl;
//   }
//   //  auto s = std::chrono::high_resolution_clock::now();
//   gen.init();
//   /*std::cout << "init:"
//       <<
//   float(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now()
//   - s).count()) / 1000000
//       << std::endl;
//   auto s2 = std::chrono::high_resolution_clock::now();*/
//   // exit(0);
//   TreeNode node;
//   long long mt = 0;
//   while(1) {
//     auto v = gen.get();
//     if(v.first.id == -1) break;
//     auto s3 = std::chrono::high_resolution_clock::now();
//     vector< TreeNode > node_pair;
//     TreeNode n1(v.first.id, v.first.x, v.first.y);
//     TreeNode n2(v.second.id, v.second.x, v.second.y);
//     // if(n1.id<9000)
//     // std::cout<<"lc.id:"<<n1.id<<"\tlc.x"<<n1.x<<"\tlc.y"<<n1.y<<std::endl;
//     node_pair.emplace_back(n1);
//     node_pair.emplace_back(n2);
//     // for(auto i:node_pair){
//     //   std::cout << "node_pair"<<i<<"_id_:"<<i.id<<endl;
//     // }
//     // cout<<endl;
//     // cout<<"totalClkWL:"<<totalClkWL<<endl;//神级补丁 别删 删了就g
//     node = merge(node_pair);
//     // treeNodesForDP[]

//     //  mt +=
//     //
//     std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now()
//     //  - s3).count();
//     // std::cout << v.first.x << "," << v.first.y << " " << v.second.x << ","
//     <<
//     // v.second.y << std::endl;
//     gen.add(fzq::FPos(node.x, node.y, node.id));
//   }
//   //
//   std::cout<<"node"<<node.lc->id<<"\tnode.x"<<node.lc->x<<"\tnode.y"<<node.lc->y<<std::endl;
//   return node;
// }

// // Deferred-Merge Embedding //。
// void Router::DME() {
//   int sz = 2 * num_sinks - 1;
//   if(vertexMS.capacity() == 0) vertexMS.resize(sz + 1);
//   if(vertexTRR.capacity() == 0) vertexTRR.resize(sz + 1);
//   if(vertexDistE.capacity() == 0) vertexDistE.resize(sz + 1);
//   if(vertexDistE_2.capacity() == 0) vertexDistE_2.resize(sz + 1);
//   if(treeNodes.capacity() == 0) treeNodes.resize(sz + 1);

//   assert(vertexMS.capacity() == sz + 1);
//   assert(vertexTRR.capacity() == sz + 1);
//   assert(vertexDistE.capacity() == sz + 1);
//   assert(vertexDistE_2.capacity() == sz + 1);
//   assert(treeNodes.capacity() == sz + 1);

//   vector< TreeNode > nodesForMerge;
//   for(int i = 1; i <= num_sinks; i++) {
//     TreeNode tr = TreeNode(sinks[i].id, sinks[i].x, sinks[i].y);
//     // TreeNode tr = TreeNode();
//     treeNodes[i].set_base(sinks[i].id, sinks[i].x, sinks[i].y);
//     // std::cout <<"sinks["<<i<<"]"<<sinks[i].id<<" "<<sinks[i].x<<"
//     // "<<sinks[i].y<<std::endl;
//     nodesForMerge.emplace_back(sinks[i].id, sinks[i].x, sinks[i].y);
//     // std::cout <<"nodesForMerge["<<i<<"]\t"<<nodesForMerge[i].id<<"
//     // "<<nodesForMerge[i].x<<" "<<nodesForMerge[i].y<<std::endl;
//   }

//   // for (int i = 0; i < nodesForMerge.size(); i++) {
//   //     cout << "tr" << nodesForMerge[i].id << ", x: " << nodesForMerge[i].x
//   <<
//   //     " y: " << nodesForMerge[i].y << endl;
//   // }
//   // 1. Build Tree of Segments (bottom up)
//   auto root = bottom_up(nodesForMerge);
//   // topo = make_shared<TreeTopology>(root, num_sinks, sz);

//   // postOrderTraversal(topo.root);
//   // cout  << "Finish bottom-up process" << endl;
//   // assert(!isnan(totalClkWL));
//   // cout << endl << "clock-net wirelength: " << totalClkWL << endl << endl;

//   // 2. Find Exact Placement(top down)
//   // pl.resize(topo.size + 1);
//   // sol.resize(topo.size + 1);

//   // preOrderTraversal(topo.root);
//   // sol_normalization();
//   // cout  << "Finish top-down process"  << endl;

//   // cout << padding << "Finished DME" << padding << endl;
// }
// double manhattanDistance(std::pair< double, double > a,
//                          std::pair< double, double > b) {
//   return std::abs(a.first - b.first) + std::abs(a.second - b.second);
// }
// TreeNode Router::HTree(std::vector< Sink > ffs, int& idCounter) {
//   return TreeNode(-1, -1, -1);
// }

// void Router::assignLevels(TreeNode* root) {
//   if(!root) return;

//   std::queue< TreeNode* > q;
//   root->level = 0;
//   q.push(root);

//   while(!q.empty()) {
//     TreeNode* current = q.front();
//     q.pop();

//     int currentLevel = current->level;
//     if(current->lc) {
//       current->lc->level = currentLevel + 1;
//       q.push(current->lc);
//     }
//     if(current->rc) {
//       current->rc->level = currentLevel + 1;
//       q.push(current->rc);
//     }
//   }
// }

// // lxm:返回根节点
// std::pair< double, double > Router::RootDME() {
//   int sz = 2 * num_sinks - 1;
//   if(vertexMS.capacity() == 0) vertexMS.resize(sz + 1);
//   if(vertexTRR.capacity() == 0) vertexTRR.resize(sz + 1);
//   if(vertexDistE.capacity() == 0) vertexDistE.resize(sz + 1);
//   if(vertexDistE_2.capacity() == 0) vertexDistE_2.resize(sz + 1);
//   if(treeNodes.capacity() == 0) treeNodes.resize(sz + 1);

//   assert(vertexMS.capacity() == sz + 1);
//   assert(vertexTRR.capacity() == sz + 1);
//   assert(vertexDistE.capacity() == sz + 1);
//   assert(vertexDistE_2.capacity() == sz + 1);
//   assert(treeNodes.capacity() == sz + 1);

//   vector< TreeNode > nodesForMerge;
//   for(int i = 1; i <= num_sinks; i++) {
//     TreeNode tr = TreeNode(sinks[i].id, sinks[i].x, sinks[i].y);
//     // TreeNode tr = TreeNode();
//     treeNodes[i].set_base(sinks[i].id, sinks[i].x, sinks[i].y);
//     // std::cout <<"sinks["<<i<<"]"<<sinks[i].id<<" "<<sinks[i].x<<"
//     // "<<sinks[i].y<<std::endl;
//     nodesForMerge.emplace_back(sinks[i].id, sinks[i].x, sinks[i].y);
//     // std::cout <<"nodesForMerge["<<i<<"]\t"<<nodesForMerge[i].id<<"
//     // "<<nodesForMerge[i].x<<" "<<nodesForMerge[i].y<<std::endl;
//   }

//   // for (int i = 0; i < nodesForMerge.size(); i++) {
//   //     cout << "tr" << nodesForMerge[i].id << ", x: " << nodesForMerge[i].x
//   <<
//   //     " y: " << nodesForMerge[i].y << endl;
//   // }

//   // 1. Build Tree of Segments (bottom up)
//   auto root = bottom_up(nodesForMerge);
//   // assignLevels
//   assignLevels(&treeNodes[root.id]);
//   double x = root.x, y = root.y;
//   // topo = make_shared<TreeTopology>(root, num_sinks, sz);

//   // postOrderTraversal(topo.root);
//   // cout  << "Finish bottom-up process" << endl;
//   // assert(!isnan(totalClkWL));
//   // cout << endl << "clock-net wirelength: " << totalClkWL << endl << endl;

//   // 2. Find Exact Placement(top down)
//   // pl.resize(topo.size + 1);
//   // sol.resize(topo.size + 1);

//   // preOrderTraversal(topo.root);
//   // sol_normalization();
//   // cout  << "Finish top-down process"  << endl;

//   // cout << padding << "Finished DME" << padding << endl;

//   return std::make_pair(x, y);
// }

// // 。
// void Router::route() {
//   // HC();  // try hierarchical clustering
//   DME();
// }

// std::pair< double, double > Router::rootRoute() {
//   // HC();  // try hierarchical clustering
//   return RootDME();
// }

// bool db_equal(double a, double b) { return abs(a - b) < eps; }

// // 。
// inline prec fastExp(prec a) {
//   a = 1.0 + a / 1024.0;
//   a *= a;
//   a *= a;
//   a *= a;
//   a *= a;
//   a *= a;
//   a *= a;
//   a *= a;
//   a *= a;
//   a *= a;
//   a *= a;
//   return a;
// }
// // 。
// float gradLess(float c1, float c2, float cof) {
//   float tmp = (c1 - c2) * cof;
//   float e1 = fastExp(tmp);
//   float sum_denom = e1 + 1;

//   float grad1 = 0;
//   if(tmp > NEG_MAX_EXP) {
//     grad1 = ((e1 + cof * e1 * c1) * sum_denom - cof * e1 * (c1 * e1 + c2)) /
//             (sum_denom * sum_denom);
//   }

//   float grad2 = ((1 - cof * c1) * sum_denom + (c1 + c2 * e1) * cof) /
//                 (sum_denom * sum_denom);

//   return grad1 - grad2;
// }
// // 。
// float gradGreater(float c1, float c2, float cof) {
//   float tmp = (c2 - c1) * cof;
//   float e2 = fastExp(tmp);
//   float sum_denom = 1 + e2;

//   float grad1 = ((1 + c1 * cof) * sum_denom - cof * (c1 + c2 * e2)) /
//                 (sum_denom * sum_denom);

//   float grad2 = 0;
//   if(tmp > NEG_MAX_EXP) {
//     grad2 = ((e2 - cof * e2 * c1) * sum_denom + cof * e2 * (c1 * e2 + c2)) /
//             (sum_denom * sum_denom);
//   }

//   return grad1 - grad2;
// }
// // 。
// //  FPOS Router::getGrad2(double x1, double y1, double x2, double y2) {
// //      return FPOS(x1 < x2 ? gradLess(x1, x2, wlen_cof.x): gradGreater(x1,
// x2,
// //      wlen_cof.x),
// //          y1 < y2 ? gradLess(y1, y2, wlen_cof.y): gradGreater(y1, y2,
// //          wlen_cof.y));
// //  }
// // 这个wlen_cof是什么东西啊

#include "Router.h"
#include <climits>
// #include "GenTree.h"
#include "global.h"
// #include "mkl.h"
#include "CTSTree.h"
#include <chrono>

using std::cout;
using std::endl;
using std::setprecision;

#define COMPLETE_LINKAGE 0
#define SINGLE_LINKAGE 1
#define L1 1
#define L2 2
#define eps 1e-6
const string padding(30, '=');

double round(double number, unsigned int bits) {
  stringstream ss;
  ss << fixed << setprecision(bits) << number;
  ss >> number;
  return number;
}

static int id = 2000;
static int NEG_MAX_EXP = -300;

int genID() { return id++; }

int getID() { return id; }

enum class Direction { EMPTY, VERTICAL, HRIZONTAL };

template < class T >
class Point {
 public:
  T x, y;
  Point(T x, T y) : x(x), y(y) {}
  Point() : x(-1), y(-1) {}
  bool operator==(const Point& rhs) const { return (x == rhs.x && y == rhs.y); }
};

class GridState {
  short value;
  GridState(int value) : value(value) {}

 public:
  // empty 000, start 001, end 010, block 100
  static const GridState EMPTY, START, END;

  GridState(const GridState& s) { value = s.value; }
  bool isMet() { return value == 3; }
  bool canVisited() { return value != 1; }
  void addState(const GridState s) { value |= s.value; }
  void reset() { value &= 4; }
};
const GridState GridState::EMPTY = 0;
const GridState GridState::START = 1;
const GridState GridState::END = 2;

using PointInt = Point< int >;

class RouteResource {
  vector< vector< PointInt > > parents;
  vector< vector< Direction > > dirs;
  vector< vector< int > > turns;
  vector< vector< bool > > visited;
  PointInt cur;
  queue< PointInt > q;

 public:
  RouteResource() : cur(-1, -1) {}
  PointInt getCur() { return cur; }
  void construct(int width, int height) {
    parents.resize(width);
    for(auto& v : parents) {
      v.resize(height);
    }
    dirs.resize(width);
    for(auto& v : dirs) {
      v.resize(height);
    }
    turns.resize(width);
    for(auto& v : turns) {
      v.resize(height);
    }
    visited = std::move(
        vector< vector< bool > >(width, vector< bool >(height, false)));
    cur = PointInt(-1, -1);

    queue< PointInt > tmp;
    swap(q, tmp);
  }
  void init(PointInt start, Direction dir) {
    for(auto& v : visited) {
      std::fill(v.begin(), v.end(), false);
    }
    cur = PointInt(-1, -1);
    queue< PointInt > tmp;
    swap(q, tmp);

    q.push(start);
    parents[start.x][start.y] = PointInt(-1, -1);
    dirs[start.x][start.y] = dir;
    turns[start.x][start.y] = 0;
    visited[start.x][start.y] = true;
  }
  PointInt next() {
    for(auto p = parents[cur.x][cur.y]; p.x >= 0 && p.y >= 0;
        cur = p, p = parents[cur.x][cur.y]) {
      if(dirs[p.x][p.y] != dirs[cur.x][cur.y]) {
        cur = p;
        return p;
      }
    }
    return cur;
  }
  bool walk(vector< vector< GridState > >& states,
            vector< vector< double > >& h_edges,
            vector< vector< double > >& v_edges) {
    int num = q.size();
    assert(num != 0);
    for(int i = 0; i < num; ++i) {
      auto p = q.front();
      q.pop();

      auto& st = states[p.x][p.y];
      st.addState(GridState::START);

      if(!st.isMet()) {
        if(p.x > 0 && states[p.x - 1][p.y].canVisited() &&
           h_edges[p.x - 1][p.y] > 0.0) {
          int turn = dirs[p.x][p.y] == Direction::HRIZONTAL
                         ? turns[p.x][p.y]
                         : turns[p.x][p.y] + 1;

          if(!visited[p.x - 1][p.y] || turns[p.x - 1][p.y] > turn) {
            if(!visited[p.x - 1][p.y]) {
              visited[p.x - 1][p.y] = true;
              q.emplace(p.x - 1, p.y);
            }
            dirs[p.x - 1][p.y] = Direction::HRIZONTAL;
            turns[p.x - 1][p.y] = turn;
            parents[p.x - 1][p.y] = PointInt(p.x, p.y);
          }
        }
        if(p.x < states.size() - 1 && states[p.x + 1][p.y].canVisited() &&
           h_edges[p.x][p.y] > 0.0) {
          int turn = dirs[p.x][p.y] == Direction::HRIZONTAL
                         ? turns[p.x][p.y]
                         : turns[p.x][p.y] + 1;

          if(!visited[p.x + 1][p.y] || turns[p.x + 1][p.y] > turn) {
            if(!visited[p.x + 1][p.y]) {
              visited[p.x + 1][p.y] = true;
              q.emplace(p.x + 1, p.y);
            }
            dirs[p.x + 1][p.y] = Direction::HRIZONTAL;
            turns[p.x + 1][p.y] = turn;
            parents[p.x + 1][p.y] = PointInt(p.x, p.y);
          }
        }
        if(p.y > 0 && states[p.x][p.y - 1].canVisited() &&
           v_edges[p.x][p.y - 1] > 0.0) {
          int turn = dirs[p.x][p.y] == Direction::VERTICAL
                         ? turns[p.x][p.y]
                         : turns[p.x][p.y] + 1;

          if(!visited[p.x][p.y - 1] || turns[p.x][p.y - 1] > turn) {
            if(!visited[p.x][p.y - 1]) {
              visited[p.x][p.y - 1] = true;
              q.emplace(p.x, p.y - 1);
            }
            dirs[p.x][p.y - 1] = Direction::VERTICAL;
            turns[p.x][p.y - 1] = turn;
            parents[p.x][p.y - 1] = PointInt(p.x, p.y);
          }
        }
        if(p.y < states[0].size() - 1 && states[p.x][p.y + 1].canVisited() &&
           v_edges[p.x][p.y] > 0.0) {
          int turn = dirs[p.x][p.y] == Direction::VERTICAL
                         ? turns[p.x][p.y]
                         : turns[p.x][p.y] + 1;

          if(!visited[p.x][p.y + 1] || turns[p.x][p.y + 1] > turn) {
            if(!visited[p.x][p.y + 1]) {
              visited[p.x][p.y + 1] = true;
              q.emplace(p.x, p.y + 1);
            }
            dirs[p.x][p.y + 1] = Direction::VERTICAL;
            turns[p.x][p.y + 1] = turn;
            parents[p.x][p.y + 1] = PointInt(p.x, p.y);
          }
        }
      }
      else {
        cur = p;
        return true;
      }
    }
    return false;
  }
};

// 。
inline double L1Dist(GridPoint p1, GridPoint p2) {
  return abs(p1.x - p2.x) + abs(p1.y - p2.y);
}
inline double L2Dist(TreeNode p1, TreeNode p2) {
  return abs(p1.x - p2.x) + abs(p1.y - p2.y);
}
inline double L3Dist(GrSteiner p1, GrSteiner p2) {
  return abs(p1.x - p2.x) + abs(p1.y - p2.y);
}

// 。
Segment TRRintersect(TRR& trr1, TRR& trr2) {
  // get four edges
  // cout << "Merging: " << trr1 << " and " << trr2 << endl;
  vector< GridPoint > trr1_boundary_grid;
  vector< GridPoint > trr2_boundary_grid;
  vector< Segment > trr1_Sides;
  vector< Segment > trr2_Sides;
  if(trr1.core.slope() > 0) {
    trr1_boundary_grid.emplace_back(trr1.core.p1.x,
                                    trr1.core.p1.y - trr1.radius);
    trr1_boundary_grid.emplace_back(trr1.core.p2.x + trr1.radius,
                                    trr1.core.p2.y);
    trr1_boundary_grid.emplace_back(trr1.core.p2.x,
                                    trr1.core.p2.y + trr1.radius);
    trr1_boundary_grid.emplace_back(trr1.core.p1.x - trr1.radius,
                                    trr1.core.p1.y);  // clock-wise
  }
  else if(trr1.core.slope() < 0) {
    trr1_boundary_grid.emplace_back(trr1.core.p1.x + trr1.radius,
                                    trr1.core.p1.y);
    trr1_boundary_grid.emplace_back(trr1.core.p2.x,
                                    trr1.core.p2.y + trr1.radius);
    trr1_boundary_grid.emplace_back(trr1.core.p2.x - trr1.radius,
                                    trr1.core.p2.y);
    trr1_boundary_grid.emplace_back(
        trr1.core.p1.x,
        trr1.core.p1.y - trr1.radius);  // clock-wise
  }
  else {  // leaf node
    trr1_boundary_grid.emplace_back(trr1.core.p1.x,
                                    trr1.core.p1.y - trr1.radius);
    trr1_boundary_grid.emplace_back(trr1.core.p1.x + trr1.radius,
                                    trr1.core.p1.y);
    trr1_boundary_grid.emplace_back(trr1.core.p1.x,
                                    trr1.core.p1.y + trr1.radius);
    trr1_boundary_grid.emplace_back(trr1.core.p1.x - trr1.radius,
                                    trr1.core.p1.y);  // clock-wise
  }

  if(trr2.core.slope() > 0) {
    trr2_boundary_grid.emplace_back(trr2.core.p1.x,
                                    trr2.core.p1.y - trr2.radius);
    trr2_boundary_grid.emplace_back(trr2.core.p2.x + trr2.radius,
                                    trr2.core.p2.y);
    trr2_boundary_grid.emplace_back(trr2.core.p2.x,
                                    trr2.core.p2.y + trr2.radius);
    trr2_boundary_grid.emplace_back(trr2.core.p1.x - trr2.radius,
                                    trr2.core.p1.y);  // clock-wise
  }
  else if(trr2.core.slope() < 0) {
    trr2_boundary_grid.emplace_back(trr2.core.p1.x + trr2.radius,
                                    trr2.core.p1.y);
    trr2_boundary_grid.emplace_back(trr2.core.p2.x,
                                    trr2.core.p2.y + trr2.radius);
    trr2_boundary_grid.emplace_back(trr2.core.p2.x - trr2.radius,
                                    trr2.core.p2.y);
    trr2_boundary_grid.emplace_back(
        trr2.core.p1.x,
        trr2.core.p1.y - trr2.radius);  // clock-wise
  }
  else {  // leaf node
    trr2_boundary_grid.emplace_back(trr2.core.p1.x,
                                    trr2.core.p1.y - trr2.radius);
    trr2_boundary_grid.emplace_back(trr2.core.p1.x + trr2.radius,
                                    trr2.core.p1.y);
    trr2_boundary_grid.emplace_back(trr2.core.p1.x,
                                    trr2.core.p1.y + trr2.radius);
    trr2_boundary_grid.emplace_back(trr2.core.p1.x - trr2.radius,
                                    trr2.core.p1.y);  // clock-wise
  }

  for(int i = 0; i < 4; i++) {
    trr1_boundary_grid[i].x = round(trr1_boundary_grid[i].x, 5);
    trr1_boundary_grid[i].y = round(trr1_boundary_grid[i].y, 5);
    trr2_boundary_grid[i].x = round(trr2_boundary_grid[i].x, 5);
    trr2_boundary_grid[i].y = round(trr2_boundary_grid[i].y, 5);
  }

  for(int i = 0; i < 3; i++) {
    trr1_Sides.emplace_back(trr1_boundary_grid[i], trr1_boundary_grid[i + 1]);
    trr2_Sides.emplace_back(trr2_boundary_grid[i], trr2_boundary_grid[i + 1]);
  }
  trr1_Sides.emplace_back(trr1_boundary_grid[3], trr1_boundary_grid[0]);
  trr2_Sides.emplace_back(trr2_boundary_grid[3], trr2_boundary_grid[0]);

  vector< Segment > segList;
  for(auto& seg1 : trr1_Sides) {
    for(auto& seg2 : trr2_Sides) {
      Segment seg = seg1.intersect(seg2);
      if(seg.id != -1) {
        segList.emplace_back(seg);
      }
    }
  }

  if(segList.size() == 0) {
    // cout << "Cannot find intersection between two TRRs" << endl;
    Segment ret;
    ret.id = -1;
    return ret;
  }

  int seglab = -1;
  for(auto& seg : segList) {
    seglab += 1;
    if(seg.id != -2) {
      return seg;
    }
  }

  return segList[seglab];
}

// 。
TreeNode Router::merge(vector< TreeNode > pair) {
  // cout<<"totalClkWL1:"<<totalClkWL<<endl;
  int node1_id = pair[0].id;
  int node2_id = pair[1].id;
  Segment ms_a, ms_b;

  TreeNode tr_v;
  if(node1_id <= num_sinks) {
    ms_a = vertexMS[node1_id] = Segment(sinks[node1_id], sinks[node1_id]);
    vertexMS[node1_id].cap = sinks[node1_id].cap;
  }
  else {
    ms_a = vertexMS[node1_id];
  }

  if(node2_id <= num_sinks) {
    ms_b = vertexMS[node2_id] = Segment(sinks[node2_id], sinks[node2_id]);
    vertexMS[node2_id].cap = sinks[node2_id].cap;
  }
  else {
    ms_b = vertexMS[node2_id];
  }

  double d, t_1;
  d = t_1 = L1Dist(ms_a.p1, ms_b.p1);
  GridPoint l_1 = ms_a.p1;
  GridPoint l_2 = ms_b.p1;

  double t_2 = L1Dist(ms_a.p1, ms_b.p2);
  if(t_2 < d) {
    l_1 = ms_a.p1;
    l_2 = ms_b.p2;
    d = t_2;
  }

  double t_3 = L1Dist(ms_a.p2, ms_b.p1);
  if(t_3 < d) {
    l_1 = ms_a.p2;
    l_2 = ms_b.p1;
    d = t_3;
  }

  double t_4 = L1Dist(ms_a.p2, ms_b.p2);
  if(t_4 < d) {
    l_1 = ms_a.p2;
    l_2 = ms_b.p2;
    d = t_4;
  }

  wireTypeName = wire_lib[0][0];
  unit_wire_res = wire_lib[0][1];
  unit_wire_cap = wire_lib[0][2];

  double e_a_dist, e_b_dist;
  TreeNode& lc = treeNodes[node1_id];
  TreeNode& rc = treeNodes[node2_id];

  TRR trr_a, trr_b;
  Segment ms_v;

  if(delayModel == 0) {
    if(fabs(ms_b.delay - ms_a.delay) < d) {
      double e_a_dist = (ms_b.delay - ms_a.delay + d) / 2;
      double e_b_dist = (ms_a.delay - ms_b.delay + d) / 2;

      lc = treeNodes[node1_id];
      vertexDistE[lc.id] = e_a_dist;
      vertexDistE_2[lc.id] = e_a_dist;

      rc = treeNodes[node2_id];
      vertexDistE[rc.id] = e_b_dist;
      vertexDistE_2[rc.id] = e_b_dist;

      vertexTRR[lc.id] = trr_a = TRR(ms_a, e_a_dist);
      vertexTRR[rc.id] = trr_b = TRR(ms_b, e_b_dist);

      ms_v = TRRintersect(trr_a, trr_b);

      if(ms_v.id == -1) {
        cout << "trr_a的四个点: [" << trr_a.core.p2.x << ", "
             << (trr_a.core.p2.y + trr_a.radius) << "], ["
             << (trr_a.core.p2.x + trr_a.radius) << ", " << trr_a.core.p2.y
             << "], [" << trr_a.core.p1.x << ", "
             << (trr_a.core.p1.y - trr_a.radius) << "], ["
             << (trr_a.core.p1.x - trr_a.radius) << ", " << trr_a.core.p1.y
             << "]" << endl;
        cout << "trr_b的四个点: [" << trr_b.core.p2.x << ", "
             << (trr_b.core.p2.y + trr_b.radius) << "], ["
             << (trr_b.core.p2.x + trr_b.radius) << ", " << trr_b.core.p2.y
             << "], [" << trr_b.core.p1.x << ", "
             << (trr_b.core.p1.y - trr_b.radius) << "], ["
             << (trr_b.core.p1.x - trr_b.radius) << ", " << trr_b.core.p1.y
             << "]" << endl;

        auto _p = GridPoint((l_1.x + l_2.x) / 2, (l_1.y + l_2.y) / 2);
        ms_v = Segment(_p, _p);
      }
      ms_v.delay = e_a_dist + ms_a.delay;
      internal_num += 1;
      int curId = num_sinks + internal_num;
      vertexMS[curId] = ms_v;

      tr_v = TreeNode(curId, (ms_v.p1.x + ms_v.p2.x) / 2,
                      (ms_v.p1.y + ms_v.p2.y) / 2);
      tr_v.lc = &lc;
      tr_v.rc = &rc;
      lc.par = &tr_v;
      rc.par = &tr_v;
      treeNodes[curId] = tr_v;

      totalClkWL += vertexDistE_2[lc.id];
      totalClkWL += vertexDistE_2[rc.id];
    }
    else {
      if(ms_a.delay <= ms_b.delay) {
        e_a_dist = ms_b.delay - ms_a.delay;
        e_b_dist = 0;

        lc = treeNodes[node1_id];
        vertexDistE[lc.id] = d;
        vertexDistE_2[lc.id] = e_a_dist;

        rc = treeNodes[node2_id];
        vertexDistE[rc.id] = e_b_dist;
        vertexDistE_2[rc.id] = e_b_dist;

        vertexTRR[lc.id] = trr_a = TRR(ms_a, d);
        vertexTRR[rc.id] = trr_b = TRR(ms_b, e_b_dist);

        ms_v = Segment(l_2, l_2);

        ms_v.delay = ms_b.delay;

        internal_num += 1;
        int curId = num_sinks + internal_num;
        vertexMS[curId] = ms_v;

        tr_v = TreeNode(curId, (ms_v.p1.x + ms_v.p2.x) / 2,
                        (ms_v.p1.y + ms_v.p2.y) / 2);
        tr_v.lc = &lc;
        tr_v.rc = &rc;
        lc.par = &tr_v;
        rc.par = &tr_v;
        treeNodes[curId] = tr_v;

        totalClkWL += vertexDistE_2[lc.id];
        totalClkWL += vertexDistE_2[rc.id];
      }
      else {
        e_b_dist = ms_a.delay - ms_b.delay;
        e_a_dist = 0;

        lc = treeNodes[node1_id];
        vertexDistE[lc.id] = e_a_dist;
        vertexDistE_2[lc.id] = e_a_dist;

        rc = treeNodes[node2_id];
        vertexDistE[rc.id] = d;
        vertexDistE_2[rc.id] = e_b_dist;

        vertexTRR[lc.id] = TRR(ms_a, e_a_dist);
        vertexTRR[rc.id] = TRR(ms_b, d);

        ms_v = Segment(l_1, l_1);

        ms_v.delay = ms_a.delay;

        internal_num += 1;
        int curId = num_sinks + internal_num;
        vertexMS[curId] = ms_v;

        tr_v = TreeNode(curId, (ms_v.p1.x + ms_v.p2.x) / 2,
                        (ms_v.p1.y + ms_v.p2.y) / 2);
        tr_v.lc = &lc;
        tr_v.rc = &rc;
        lc.par = &tr_v;
        rc.par = &tr_v;
        treeNodes[curId] = tr_v;

        totalClkWL += vertexDistE_2[lc.id];
        totalClkWL += vertexDistE_2[rc.id];
      }
    }
  }

  else if(delayModel == 1) {
    double y = ms_b.delay - ms_a.delay +
               unit_wire_res * d * (ms_b.cap + unit_wire_cap * d / 2);
    double z = unit_wire_res * (ms_a.cap + ms_b.cap + unit_wire_cap * d);
    double x = y / z;
    x = round(x);

    if(x < 0) {
      double cap2 = ms_b.cap * unit_wire_res;
      x = (sqrt(cap2 * cap2 +
                2 * unit_wire_res * unit_wire_cap * (ms_a.delay - ms_b.delay)) -
           cap2) /
          (unit_wire_res * unit_wire_cap);
      e_a_dist = 0;
      e_b_dist = round(x);

      // lc = treeNodes[node1_id];
      vertexDistE[lc.id] = e_a_dist;
      vertexDistE_2[lc.id] = e_a_dist;

      // rc = treeNodes[node2_id];
      vertexDistE[rc.id] = d;
      vertexDistE_2[rc.id] = e_b_dist;

      vertexTRR[lc.id] = TRR(ms_a, e_a_dist);
      vertexTRR[rc.id] = TRR(ms_b, d);

      ms_v = Segment(l_1, l_1);

      ms_v.delay = ms_a.delay;
      ms_v.cap = ms_a.cap + ms_b.cap + x * unit_wire_cap;

      internal_num += 1;
      int curId = num_sinks + internal_num;
      assert(curId <= num_sinks * 2 - 1);
      vertexMS[curId] = ms_v;

      tr_v = TreeNode(curId, (ms_v.p1.x + ms_v.p2.x) / 2,
                      (ms_v.p1.y + ms_v.p2.y) / 2);
      tr_v.lc = &lc;
      tr_v.rc = &rc;
      treeNodes[curId] = tr_v;
      // treeNodes[node1_id].par = &treeNodes[curId];
      // treeNodes[node2_id].par = &treeNodes[curId];
      lc.par = &treeNodes[curId];
      rc.par = &treeNodes[curId];
      // treeNodes[lc.id].grad = getGrad2(lc.x, lc.y, tr_v.x, tr_v.y);
      // treeNodes[rc.id].grad = getGrad2(rc.x, rc.y, tr_v.x, tr_v.y);
      // lc.par = &tr_v;
      // rc.par = &tr_v;
      if(curId == (num_sinks * 2 - 1)) {
        treeNodes[curId].par = NULL;
      }

      totalClkWL += vertexDistE_2[lc.id];
      totalClkWL += vertexDistE_2[rc.id];
      // cout<<"totalClkWL:"<<totalClkWL<<endl;
      //  calculate_buf_dist(tr_v);
    }
    else if(x > d) {
      double cap1 = ms_a.cap * unit_wire_cap;
      x = (sqrt(cap1 * cap1 +
                2 * unit_wire_res * unit_wire_cap * (ms_b.delay - ms_a.delay)) -
           cap1) /
          (unit_wire_res * unit_wire_cap);
      x = round(x);
      e_a_dist = x;
      e_b_dist = 0;

      // lc = treeNodes[node1_id];
      vertexDistE[lc.id] = d;
      vertexDistE_2[lc.id] = e_a_dist;

      // rc = treeNodes[node2_id];
      vertexDistE[rc.id] = e_b_dist;
      vertexDistE_2[rc.id] = e_b_dist;

      vertexTRR[lc.id] = trr_a = TRR(ms_a, d);
      vertexTRR[rc.id] = trr_b = TRR(ms_b, e_b_dist);

      ms_v = Segment(l_2, l_2);

      ms_v.delay = ms_b.delay;
      ms_v.cap = ms_a.cap + ms_b.cap + x * unit_wire_cap;

      internal_num += 1;
      int curId = num_sinks + internal_num;
      assert(curId <= num_sinks * 2 - 1);
      vertexMS[curId] = ms_v;

      tr_v = TreeNode(curId, (ms_v.p1.x + ms_v.p2.x) / 2,
                      (ms_v.p1.y + ms_v.p2.y) / 2);
      tr_v.lc = &lc;
      tr_v.rc = &rc;
      treeNodes[curId] = tr_v;
      // treeNodes[node1_id].par = &treeNodes[curId];
      // treeNodes[node2_id].par = &treeNodes[curId];
      lc.par = &treeNodes[curId];
      rc.par = &treeNodes[curId];
      // treeNodes[lc.id].grad = getGrad2(lc.x, lc.y, tr_v.x, tr_v.y);
      // treeNodes[rc.id].grad = getGrad2(rc.x, rc.y, tr_v.x, tr_v.y);
      // lc.par = &tr_v;
      // rc.par = &tr_v;/home/eda/lxm/cts_opendp_cluster/src
      treeNodes[curId] = tr_v;
      if(curId == (num_sinks * 2 - 1)) {
        treeNodes[curId].par = NULL;
      }

      totalClkWL += vertexDistE_2[lc.id];
      totalClkWL += vertexDistE_2[rc.id];

      // cout<<"totalClkWL:"<<totalClkWL<<endl;
      //  calculate_buf_dist(tr_v);
    }
    else {
      e_a_dist = round(x);
      e_b_dist = d - e_a_dist;

      // lc = treeNodes[node1_id];
      vertexDistE[lc.id] = e_a_dist;
      vertexDistE_2[lc.id] = e_a_dist;

      // rc = treeNodes[node2_id];
      vertexDistE[rc.id] = e_b_dist;
      vertexDistE_2[rc.id] = e_b_dist;

      vertexTRR[lc.id] = trr_a = TRR(ms_a, e_a_dist);
      vertexTRR[rc.id] = trr_b = TRR(ms_b, e_b_dist);

      // ms_v = TRRintersect(trr_a, trr_b);

      // if (ms_v.id == -1) {
      // cout << "trr_a的四个点: [" << trr_a.core.p2.x << ", " <<
      // (trr_a.core.p2.y + trr_a.radius) << "], ["
      //     << (trr_a.core.p2.x + trr_a.radius) << ", " << trr_a.core.p2.y <<
      //     "], ["
      //     << trr_a.core.p1.x << ", " << (trr_a.core.p1.y - trr_a.radius) <<
      //     "], ["
      //     << (trr_a.core.p1.x - trr_a.radius) << ", " << trr_a.core.p1.y <<
      //     "]" << endl;
      // cout << "trr_b的四个点: [" << trr_b.core.p2.x << ", " <<
      // (trr_b.core.p2.y + trr_b.radius) << "], ["
      //     << (trr_b.core.p2.x + trr_b.radius) << ", " << trr_b.core.p2.y <<
      //     "], ["
      //     << trr_b.core.p1.x << ", " << (trr_b.core.p1.y - trr_b.radius) <<
      //     "], ["
      //     << (trr_b.core.p1.x - trr_b.radius) << ", " << trr_b.core.p1.y <<
      //     "]" << endl;

      // for (auto& sink : sinks) {
      //     cout << sink << endl;
      // }
      // ms_v = TRRintersect(trr_a, trr_b);
      auto _p = GridPoint((l_1.x + l_2.x) / 2, (l_1.y + l_2.y) / 2);
      ms_v = Segment(_p, _p);
      // cout << "Merge failure" << endl;
      // exit(1);
      // }

      ms_v.delay =
          unit_wire_res * e_a_dist * (ms_a.cap + unit_wire_cap * e_a_dist / 2) +
          ms_a.delay;
      ms_v.delay = round(ms_v.delay, 5);
      ms_v.cap = ms_a.cap + ms_b.cap + d * unit_wire_cap;

      internal_num += 1;
      int curId = num_sinks + internal_num;
      assert(curId <= num_sinks * 2 - 1);
      vertexMS[curId] = ms_v;

      tr_v = TreeNode(curId, (ms_v.p1.x + ms_v.p2.x) / 2,
                      (ms_v.p1.y + ms_v.p2.y) / 2);
      tr_v.lc = &lc;
      tr_v.rc = &rc;
      treeNodes[curId] = tr_v;

      lc.par = &treeNodes[curId];
      rc.par = &treeNodes[curId];

      treeNodes[curId] = tr_v;
      if(curId == (num_sinks * 2 - 1)) {
        treeNodes[curId].par = NULL;
      }

      totalClkWL += vertexDistE_2[lc.id];
      totalClkWL += vertexDistE_2[rc.id];

      // cout<<"totalClkWL:"<<totalClkWL<<endl;
      //  calculate_buf_dist(tr_v);
    }
  }

  return tr_v;
}

// 。
TreeNode Router::bottom_up(vector< TreeNode > pts) {
  static fzq::CTSTree gen;
  std::vector< fzq::FPos >& sinks = gen.getSinks();
  sinks.clear();
  for(auto& v : pts) {
    sinks.emplace_back(v.x, v.y, v.id);
    // std::cout<<"v.x:"<<v.x<<"\tv.y:"<<v.y<<"\tv.id:"<<v.id<<std::endl;
  }
  //  auto s = std::chrono::high_resolution_clock::now();
  gen.init();
  /*std::cout << "init:"
      <<
  float(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now()
  - s).count()) / 1000000
      << std::endl;
  auto s2 = std::chrono::high_resolution_clock::now();*/
  // exit(0);
  TreeNode node;
  long long mt = 0;
  while(1) {
    auto v = gen.get();
    if(v.first.id == -1) break;
    auto s3 = std::chrono::high_resolution_clock::now();
    vector< TreeNode > node_pair;
    TreeNode n1(v.first.id, v.first.x, v.first.y);
    TreeNode n2(v.second.id, v.second.x, v.second.y);
    // if(n1.id<9000)
    // std::cout<<"lc.id:"<<n1.id<<"\tlc.x"<<n1.x<<"\tlc.y"<<n1.y<<std::endl;
    node_pair.emplace_back(n1);
    node_pair.emplace_back(n2);
    // for(auto i:node_pair){
    //   std::cout << "node_pair"<<i<<"_id_:"<<i.id<<endl;
    // }
    // cout<<endl;
    // cout<<"totalClkWL:"<<totalClkWL<<endl;//神级补丁 别删 删了就g
    node = merge(node_pair);
    // treeNodesForDP[]

    //  mt +=
    //  std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now()
    //  - s3).count();
    // std::cout << v.first.x << "," << v.first.y << " " << v.second.x << "," <<
    // v.second.y << std::endl;
    gen.add(fzq::FPos(node.x, node.y, node.id));
  }
  // std::cout<<"node"<<node.lc->id<<"\tnode.x"<<node.lc->x<<"\tnode.y"<<node.lc->y<<std::endl;
  return node;
}

// Deferred-Merge Embedding //。
void Router::DME() {
  int sz = 2 * num_sinks - 1;
  if(vertexMS.capacity() == 0) vertexMS.resize(sz + 1);
  if(vertexTRR.capacity() == 0) vertexTRR.resize(sz + 1);
  if(vertexDistE.capacity() == 0) vertexDistE.resize(sz + 1);
  if(vertexDistE_2.capacity() == 0) vertexDistE_2.resize(sz + 1);
  if(treeNodes.capacity() == 0) treeNodes.resize(sz + 1);

  assert(vertexMS.capacity() == sz + 1);
  assert(vertexTRR.capacity() == sz + 1);
  assert(vertexDistE.capacity() == sz + 1);
  assert(vertexDistE_2.capacity() == sz + 1);
  assert(treeNodes.capacity() == sz + 1);

  vector< TreeNode > nodesForMerge;
  for(int i = 1; i <= num_sinks; i++) {
    TreeNode tr = TreeNode(sinks[i].id, sinks[i].x, sinks[i].y);
    // TreeNode tr = TreeNode();
    treeNodes[i].set_base(sinks[i].id, sinks[i].x, sinks[i].y);
    // std::cout <<"sinks["<<i<<"]"<<sinks[i].id<<" "<<sinks[i].x<<"
    // "<<sinks[i].y<<std::endl;
    nodesForMerge.emplace_back(sinks[i].id, sinks[i].x, sinks[i].y);
    // std::cout <<"nodesForMerge["<<i<<"]\t"<<nodesForMerge[i].id<<"
    // "<<nodesForMerge[i].x<<" "<<nodesForMerge[i].y<<std::endl;
  }

  // for (int i = 0; i < nodesForMerge.size(); i++) {
  //     cout << "tr" << nodesForMerge[i].id << ", x: " << nodesForMerge[i].x <<
  //     " y: " << nodesForMerge[i].y << endl;
  // }
  // 1. Build Tree of Segments (bottom up)
  auto root = bottom_up(nodesForMerge);
  // topo = make_shared<TreeTopology>(root, num_sinks, sz);

  // postOrderTraversal(topo.root);
  // cout  << "Finish bottom-up process" << endl;
  // assert(!isnan(totalClkWL));
  // cout << endl << "clock-net wirelength: " << totalClkWL << endl << endl;

  // 2. Find Exact Placement(top down)
  // pl.resize(topo.size + 1);
  // sol.resize(topo.size + 1);

  // preOrderTraversal(topo.root);
  // sol_normalization();
  // cout  << "Finish top-down process"  << endl;

  // cout << padding << "Finished DME" << padding << endl;
}

// lxm:返回根节点
std::pair< double, double > Router::RootDME() {
  int sz = 2 * num_sinks - 1;
  if(vertexMS.capacity() == 0) vertexMS.resize(sz + 1);
  if(vertexTRR.capacity() == 0) vertexTRR.resize(sz + 1);
  if(vertexDistE.capacity() == 0) vertexDistE.resize(sz + 1);
  if(vertexDistE_2.capacity() == 0) vertexDistE_2.resize(sz + 1);
  if(treeNodes.capacity() == 0) treeNodes.resize(sz + 1);

  assert(vertexMS.capacity() == sz + 1);
  assert(vertexTRR.capacity() == sz + 1);
  assert(vertexDistE.capacity() == sz + 1);
  assert(vertexDistE_2.capacity() == sz + 1);
  assert(treeNodes.capacity() == sz + 1);

  vector< TreeNode > nodesForMerge;
  for(int i = 1; i <= num_sinks; i++) {
    TreeNode tr = TreeNode(sinks[i].id, sinks[i].x, sinks[i].y);
    // TreeNode tr = TreeNode();
    treeNodes[i].set_base(sinks[i].id, sinks[i].x, sinks[i].y);
    // std::cout <<"sinks["<<i<<"]"<<sinks[i].id<<" "<<sinks[i].x<<"
    // "<<sinks[i].y<<std::endl;
    nodesForMerge.emplace_back(sinks[i].id, sinks[i].x, sinks[i].y);
    // std::cout <<"nodesForMerge["<<i<<"]\t"<<nodesForMerge[i].id<<"
    // "<<nodesForMerge[i].x<<" "<<nodesForMerge[i].y<<std::endl;
  }

  // for (int i = 0; i < nodesForMerge.size(); i++) {
  //     cout << "tr" << nodesForMerge[i].id << ", x: " << nodesForMerge[i].x <<
  //     " y: " << nodesForMerge[i].y << endl;
  // }

  // 1. Build Tree of Segments (bottom up)
  auto root = bottom_up(nodesForMerge);
  double x = root.x, y = root.y;
  // topo = make_shared<TreeTopology>(root, num_sinks, sz);

  // postOrderTraversal(topo.root);
  // cout  << "Finish bottom-up process" << endl;
  // assert(!isnan(totalClkWL));
  //cout << endl << "clock-net wirelength: " << totalClkWL << endl << endl;

  // 2. Find Exact Placement(top down)
  // pl.resize(topo.size + 1);
  // sol.resize(topo.size + 1);

  // preOrderTraversal(topo.root);
  // sol_normalization();
  // cout  << "Finish top-down process"  << endl;

  // cout << padding << "Finished DME" << padding << endl;

  return std::make_pair(x, y);
}

// 。
void Router::route() {
  // HC();  // try hierarchical clustering
  DME();
}

std::pair< double, double > Router::rootRoute() {
  // HC();  // try hierarchical clustering
  return RootDME();
}

bool db_equal(double a, double b) { return abs(a - b) < eps; }

// 。
inline prec fastExp(prec a) {
  a = 1.0 + a / 1024.0;
  a *= a;
  a *= a;
  a *= a;
  a *= a;
  a *= a;
  a *= a;
  a *= a;
  a *= a;
  a *= a;
  a *= a;
  return a;
}
// 。
float gradLess(float c1, float c2, float cof) {
  float tmp = (c1 - c2) * cof;
  float e1 = fastExp(tmp);
  float sum_denom = e1 + 1;

  float grad1 = 0;
  if(tmp > NEG_MAX_EXP) {
    grad1 = ((e1 + cof * e1 * c1) * sum_denom - cof * e1 * (c1 * e1 + c2)) /
            (sum_denom * sum_denom);
  }

  float grad2 = ((1 - cof * c1) * sum_denom + (c1 + c2 * e1) * cof) /
                (sum_denom * sum_denom);

  return grad1 - grad2;
}
// 。
float gradGreater(float c1, float c2, float cof) {
  float tmp = (c2 - c1) * cof;
  float e2 = fastExp(tmp);
  float sum_denom = 1 + e2;

  float grad1 = ((1 + c1 * cof) * sum_denom - cof * (c1 + c2 * e2)) /
                (sum_denom * sum_denom);

  float grad2 = 0;
  if(tmp > NEG_MAX_EXP) {
    grad2 = ((e2 - cof * e2 * c1) * sum_denom + cof * e2 * (c1 * e2 + c2)) /
            (sum_denom * sum_denom);
  }

  return grad1 - grad2;
}
// 。
//  FPOS Router::getGrad2(double x1, double y1, double x2, double y2) {
//      return FPOS(x1 < x2 ? gradLess(x1, x2, wlen_cof.x): gradGreater(x1, x2,
//      wlen_cof.x),
//          y1 < y2 ? gradLess(y1, y2, wlen_cof.y): gradGreater(y1, y2,
//          wlen_cof.y));
//  }
// 这个wlen_cof是什么东西啊