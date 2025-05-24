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
#include <unordered_set>
#define _DEBUG
#define _DEBUG2

using opendp::cell;
using opendp::circuit;
using opendp::pixel;
using opendp::rect;
using opendp::row;

using std::cerr;
using std::cout;
using std::endl;
using std::max;
using std::min;
using std::pair;

// Fixed cell handle on parser ( no need to use this function during placement)
// //
void circuit::fixed_cell_assign() {
  for(int i = 0; i < cells.size(); i++) {
    cell* theCell = &cells[i];
    if(theCell->isFixed == true) {
      macro* theMacro = &macros[theCell->type];
      fix_cells.emplace_back(theCell);  // lxm:存储fix_cell
      rect new_rect(theCell->x_coord, theCell->x_coord + theCell->width,
                    theCell->y_coord, theCell->height + theCell->y_coord);
      fix_rects.emplace_back(new_rect);  // lxm:存储fix_cell的rect
      int y_start = (int)floor(theCell->y_coord / rowHeight);
      int y_end = (int)ceil((theCell->y_coord + theCell->height) / rowHeight);
      int x_start = (int)floor(theCell->x_coord / wsite);
      int x_end = (int)ceil((theCell->x_coord + theCell->width) / wsite);

      int y_start_rf = (int)floor(by / rowHeight);
      int y_end_rf = (int)ceil(ty / rowHeight);
      int x_start_rf = (int)floor(lx / wsite);
      int x_end_rf = (int)ceil(rx / wsite);

      y_start = max(y_start, y_start_rf);
      y_end = min(y_end, y_end_rf);
      x_start = max(x_start, x_start_rf);
      x_end = min(x_end, x_end_rf);

#ifdef DEBUG
      cout << " cell_name : " << theCell->name << endl;
      cout << " y_start : " << y_start << endl;
      cout << " y_end   : " << y_end << endl;
      cout << " x_start : " << x_start << endl;
      cout << " x_end   : " << x_end << endl;
#endif
      for(int j = y_start; j < y_end; j++) {
        for(int k = x_start; k < x_end; k++) {
          grid[j][k].linked_cell = theCell;
          grid[j][k].util = 1.0;
        }
      }
    }
  }
  return;
}

void circuit::print_pixels() {
  cout << " print grid " << endl;
  int num = 0;
  std::unordered_set< cell* >
      unique_linked_cells;  // 用于存储唯一的linked_cell指针
  for(int i = 0; i < rows.size(); i++) {
    for(int j = 0; j < rows[i].numSites; j++) {
      // cout << grid[i][j].util << " ";
      // if(grid[i][j].util > 0.00001) {
      //   cout << grid[i][j].util << " ";
      //   if(grid[i][j].group == UINT_MAX) {
      //     cout << " no_group ";
      //   }
      //   else {
      //     cout << groups[grid[i][j].group].name << " ";
      //   }
      // }
      // 假设 grid[i][j].linked_cell 是指向 linked_cell 的指针
      if(grid[i][j].linked_cell == &dummy_cell) {
        cout << "dummy_cell ";
      }
      if(grid[i][j].linked_cell != NULL) {
        unique_linked_cells.insert(grid[i][j].linked_cell);  // 添加到集合中
        // cout << "x: " << grid[i][j].linked_cell->width / wsite
        //      << "y: " << grid[i][j].linked_cell->height / rowHeight <<
        // endl;
        num++;
      }
      // if(j == 10062 && i == 10) {
      //   cout << "x: " << grid[i][j].linked_cell->width / wsite
      //        << "y: " << grid[i][j].linked_cell->height / rowHeight << endl;
      // }
    }

    // cout << endl;
  }
  // std::cout << "Number of unique linked_cells: " <<
  // unique_linked_cells.size()
  //           << std::endl;  // 打印不重复的个数
  cout << "grid num: " << num << endl;
  // cout << " end print grid" << endl;
  return;
}

// lxm:获取每个区域的利用率和每个区域内cell的所属区域
// lxm:将这里的init_coord都改成了coord
void circuit::group_cell_region_assign() {
  for(int i = 0; i < groups.size(); i++) {
    group* theGroup = &groups[i];

    double area = 0;
    // #pragma omp parallel for reduction(+ : area)
    for(int j = 0; j < rows.size(); j++) {
      for(int k = 0; k < rows[j].numSites; k++) {
        if(grid[j][k].group != UINT_MAX) {
          if(grid[j][k].isValid == true) {
            if(groups[grid[j][k].group].name == theGroup->name)
              area += wsite * rowHeight;
          }
        }
      }
    }
    // lxm:如果cell在区域内部，就属于该区域，否则找到x+y最近的区域
    double cell_area = 0;
    for(int j = 0; j < theGroup->siblings.size(); j++) {
      cell* theCell = theGroup->siblings[j];
      cell_area += theCell->width * theCell->height;
      int dist = INT_MAX;
      unsigned region_backup = UINT_MAX;
      for(int k = 0; k < theGroup->regions.size(); k++) {
        rect* theRect = &theGroup->regions[k];
        if(check_inside(theCell, theRect, "coord") == true) theCell->region = k;
        int temp_dist = dist_for_rect(theCell, theRect, "coord");
        if(temp_dist < dist) {
          dist = temp_dist;
          region_backup = k;
        }
      }
      if(theCell->region == UINT_MAX) {
        theCell->region = region_backup;
      }
      assert(theCell->region != UINT_MAX);
    }
    theGroup->util = cell_area / area;
  }
  return;
}

// lxm:根据区域外cell的个数划分子区域(每5000个一个区域)==>每个区域的y都是0-die_height，只要x在区域内或在最左(右)边就放在对应区域
void circuit::non_group_cell_region_assign() {
  unsigned non_group_cell_count = 0;
  unsigned cell_num_check = 0;
  unsigned fixed_cell_count = 0;
  // #pragma omp parallel for reduction(+ : non_group_cell_count, cell_num_check, \
//                                        fixed_cell_count)
  for(int i = 0; i < cells.size(); i++) {
    cell* theCell = &cells[i];
    if(theCell->isFixed == true) {
      fixed_cell_count++;
      continue;
    }
    if(theCell->inGroup == false) non_group_cell_count++;
  }

  unsigned group_num = non_group_cell_count / 5000;

  if(group_num == 0) group_num = 1;

  int x_step = (int)rx / group_num;
  sub_regions.reserve(group_num);

#ifdef DEBUG
  cout << "fixed_cell_count : " << fixed_cell_count << endl;
  cout << "non_group_cell_count : " << non_group_cell_count << endl;
  cout << "group_num : " << group_num << endl;
  cout << "x_step : " << x_step << endl;
#endif

  for(int j = 0; j < group_num; j++) {
    sub_region theSub;

    theSub.boundary.xLL = j * x_step;
    theSub.boundary.xUR = min(static_cast< double >((j + 1) * x_step), rx);
    theSub.boundary.yLL = 0;
    theSub.boundary.yUR = ty;

    for(int k = 0; k < cells.size(); k++) {
      cell* theCell = &cells[k];
      if(theCell->isFixed || theCell->inGroup) continue;
      if(theCell->init_x_coord >= j * x_step &&
         theCell->init_x_coord < (j + 1) * x_step) {
#ifdef DEBUG2
        theCell->print();
        cout << "j: " << j << endl;
        cout << "k: " << k << endl;
        cout << "xLL: " << theSub.boundary.xLL << endl;
        cout << " sibilings size : " << theSub.siblings.size() << endl;
#endif
        theSub.siblings.emplace_back(theCell);
        cell_num_check++;
      }
      else if(j == 0 && theCell->init_x_coord < 0.0) {
#ifdef DEBUG2
        theCell->print();
        cout << "j: " << j << endl;
        cout << "k: " << k << endl;
        cout << "xLL: " << theSub.boundary.xLL << endl;
        cout << " sibilings size : " << theSub.siblings.size() << endl;
#endif
        theSub.siblings.emplace_back(theCell);
        cell_num_check++;
      }
      else if(j == group_num - 1 && theCell->init_x_coord >= rx) {
#ifdef DEBUG2
        theCell->print();
        cout << "j: " << j << endl;
        cout << "k: " << k << endl;
        cout << "xLL: " << theSub.boundary.xLL << endl;
        cout << " sibilings size : " << theSub.siblings.size() << endl;
#endif
        theSub.siblings.emplace_back(theCell);
        cell_num_check++;
      }
    }
    sub_regions.emplace_back(theSub);
  }
#ifdef DEBUG
  cout << "non_group_cell_count : " << non_group_cell_count << endl;
  cout << "cell_num_check : " << cell_num_check << endl;
  cout << "fixed_cell_count : " << fixed_cell_count << endl;
  cout << "sub_region_num : " << sub_regions.size() << endl;
  cout << "- - - - - - - - - - - - - - - - -" << endl;
#endif
  assert(non_group_cell_count == cell_num_check);
  return;
}

void circuit::y_align() {
  for(int i = 0; i < cells.size(); i++) {
    cell* theCell = &cells[i];
    if(theCell->isFixed || theCell->isPlaced || theCell->hold) continue;

    cell_y_align(theCell);
  }
  return;
}

void circuit::cell_y_align(cell* theCell) {
  int cell_y_size = (int)ceil(theCell->height / rowHeight);
  macro* theMacro = &macros[theCell->type];
  pair< bool, pixel* > myPixel =
      diamond_search(theCell, theCell->init_x_coord, theCell->init_y_coord);
  theCell->y_pos = myPixel.second->y_pos;
  // top power align --> cell orient ( flip )
  if(max_cell_height > 1) {
    if(cell_y_size % 2 == 1 &&
       rows[myPixel.second->y_pos].top_power != theMacro->top_power)
      theCell->cellorient = "FS";
  }
  else {
    theCell->cellorient = rows[myPixel.second->y_pos].siteorient;
  }

  return;
}

// lxm:如果一个grid的部分在group的区域中，认为是无效的，全部在也认为有效===>此代码被注释了
void circuit::group_pixel_assign_2() {
  for(int i = 0; i < rows.size(); i++) {
    row* theRow = &rows[i];
    for(int j = 0; j < theRow->numSites; j++) {
      rect theGrid;
      theGrid.xLL = j * wsite;
      theGrid.xUR = (j + 1) * wsite;
      theGrid.yLL = i * rowHeight;
      theGrid.yUR = (i + 1) * rowHeight;
      for(int k = 0; k < groups.size(); k++) {
        group* theGroup = &groups[k];
        for(int l = 0; l < theGroup->regions.size(); l++) {
          rect theRect = theGroup->regions[l];
          // cout << "rect : " << theRect.xLL << " " << theRect.yLL << " -- " <<
          // theRect.xUR << " " << theRect.yUR << endl;
          // cout << "grid[" << i << "][" << j << "]";
          if(check_inside(theGrid, theGroup->regions[l]) == false &&
             check_overlap(theGrid, theGroup->regions[l]) == true) {
            grid[i][j].util = 0.0;
            grid[i][j].linked_cell = &dummy_cell;
            grid[i][j].isValid = false;
            // cout << "invalid grid[" << i << "][" << j << "] marked" << endl;
          }
        }
      }
    }
  }
  return;
}

// lxm:group的区域内如果有site被截断了，认为是无效的grid
void circuit::group_pixel_assign() {
  for(int i = 0; i < rows.size(); i++) {
    row* theRow = &rows[i];
    for(int j = 0; j < theRow->numSites; j++) {
      grid[i][j].util = 0.0;
    }
  }

  for(int i = 0; i < groups.size(); i++) {
    group* theGroup = &groups[i];
    for(int j = 0; j < theGroup->regions.size(); j++) {
      rect* theRect = &theGroup->regions[j];
      int row_start = (int)ceil(theRect->yLL / rowHeight);
      int row_end = (int)floor(theRect->yUR / rowHeight);

      // assert((int)floor(theRect->yUR) % (int)rowHeight == 0 );

      for(int k = row_start; k < row_end; k++) {
        row* theRow = &rows[k];
        int col_start = (int)floor(theRect->xLL / (double)theRow->stepX);
        int col_end = (int)ceil(theRect->xUR / (double)theRow->stepX);

        for(int l = col_start; l < col_end; l++) {
          grid[k][l].util += 1.0;
        }
        // lxm:如果group的区域没有完全与row对齐，需要减掉部分利用率
        if((int)(theRect->xLL + 0.5) % theRow->stepX != 0) {
          grid[k][col_start].util -=
              ((int)(theRect->xLL + 0.5) % theRow->stepX) /
              (double)theRow->stepX;
          // cout << "grid[" << k << "][" << col_start
          //      << "] util : " << grid[k][col_start].util << endl;
        }
        if((int)(theRect->xUR + 0.5) % theRow->stepX != 0) {
          // lxm：200改成了theRow->stepX
          grid[k][col_end - 1].util -=
              (theRow->stepX - (int)(theRect->xUR + 0.5) % theRow->stepX) /
              (double)theRow->stepX;
          // cout << "grid[" << k << "][" << col_end - 1
          //      << "] util : " << grid[k][col_end - 1].util << endl;
        }
      }
    }
    for(int j = 0; j < theGroup->regions.size(); j++) {
      rect* theRect = &theGroup->regions[j];
      // lxm:第i个row的高度是i-1~i所以这里是ceil和floor
      int row_start = (int)ceil(theRect->yLL / rowHeight);
      int row_end = (int)floor(theRect->yUR / rowHeight);
      for(int k = row_start; k < row_end; k++) {
        row* theRow = &rows[k];
        int col_start = (int)floor(theRect->xLL / (double)theRow->stepX);
        int col_end = (int)ceil(theRect->xUR / (double)theRow->stepX);
        // assig groupid to each pixel ( grid )
        for(int l = col_start; l < col_end; l++) {
          // lxm:当作完整在区域内，linked_cell = NUL
          if(abs(grid[k][l].util - 1.0) < 1e-6) {
            grid[k][l].group = group2id[theGroup->name];
            grid[k][l].linked_cell = NULL;
            grid[k][l].isValid = true;
            grid[k][l].util = 1.0;
          }
          // lxm:如果grid在group的区域中比较多就认为是无效的
          else if(grid[k][l].util > 0 && grid[k][l].util < 1) {
#ifdef DEBUG2
            cout << "grid[" << k << "][" << l << "]" << endl;
            cout << "util : " << grid[k][l].util << endl;
#endif
            grid[k][l].linked_cell = &dummy_cell;
            grid[k][l].util = 0.0;
            grid[k][l].isValid = false;
          }
        }
      }
    }
  }
  return;
}

// lxm:把之前放了的cell清除，包括grid和cell的坐标信息还原
void circuit::erase_pixel(cell* theCell) {
  if(theCell->isFixed == true || theCell->isPlaced == false) return;

  macro* theMacro = &macros[theCell->type];
  int x_step = (int)ceil(theCell->width / wsite);
  int y_step = (int)ceil(theCell->height / rowHeight);

  theCell->isPlaced = false;
  theCell->hold = false;

  assert(theCell->x_pos == (int)floor(theCell->x_coord / wsite + 0.5));
  assert(theCell->y_pos == (int)floor(theCell->y_coord / rowHeight + 0.5));
  for(int i = theCell->y_pos; i < theCell->y_pos + y_step; i++) {
    for(int j = theCell->x_pos; j < theCell->x_pos + x_step; j++) {
      grid[i][j].linked_cell = NULL;
      grid[i][j].util = 0;
    }
  }
  // theCell->x_coord = 0;
  // theCell->y_coord = 0;
  theCell->x_pos = 0;
  theCell->y_pos = 0;
  return;
}

// lxm:更新当前cell的坐标信息(init_x_coord,init_y_coord不改)、朝向并改为placed，更新cell占据的grid
bool circuit::paint_pixel(cell* theCell, int x_pos, int y_pos) {
  assert(theCell->isPlaced == false);
  macro* theMacro = &macros[theCell->type];
  int x_step = (int)ceil(theCell->width / wsite);
  int y_step = (int)ceil(theCell->height / rowHeight);

  theCell->x_pos = x_pos;
  theCell->y_pos = y_pos;
  theCell->x_coord = x_pos * wsite;
  theCell->y_coord = y_pos * rowHeight;
  theCell->isPlaced = true;
#ifdef DEBUG
  cout << "paint cell : " << theCell->name << endl;
  cout << "group : " << theCell->group << endl;
  cout << "init_x_coord - init_y_coord : " << theCell->init_x_coord << " - "
       << theCell->init_y_coord << endl;
  cout << "x_coord - y_coord : " << theCell->x_coord << " - "
       << theCell->y_coord << endl;
  cout << "x_step - y_step : " << x_step << " - " << y_step << endl;
  cout << "x_pos - y_pos : " << x_pos << " - " << y_pos << endl;
#endif
  for(int i = y_pos; i < y_pos + y_step; i++) {
    for(int j = x_pos; j < x_pos + x_step; j++) {
      if(grid[i][j].linked_cell != NULL) {
        cerr << " Can't paint grid [" << i << "][" << j << "] !!!" << endl;
        cerr << " group name : " << groups[grid[i][j].group].name << endl;
        cerr << " Cell name : " << grid[i][j].linked_cell->name
             << " already occupied grid" << endl;
        exit(2);
        return false;
      }
      else {
        grid[i][j].linked_cell = theCell;
        grid[i][j].util = 1.0;
        if(init_lg_flag && theCell->is_ff) {
          if(grid_init[i][j].linked_cell != NULL) {
            if(!grid_init[i][j].linked_cell->is_ff &&
               !grid_init[i][j]
                    .linked_cell->isFixed) {  // lxm:util不为0表明已经放了cell
              // lxm:找与register重叠的cell并标记
              grid_init[i][j].linked_cell->isOverlap =
                  true;  // Assuming cell has an isOverlap attribute
              // cout << "overlap cell : " << grid_init[i][j].linked_cell->name
              //      << endl;
            }
          }
          grid_init[i][j].util = 1.0;
          grid_init[i][j].linked_cell = theCell;  // lxm:在此处标记放置信息
        }
      }
    }
  }

  if(!init_lg_flag) {
    if(max_cell_height > 1) {
      if(y_step % 2 == 1) {
        if(rows[y_pos].top_power != theMacro->top_power)
          theCell->cellorient = "FS";
        else
          theCell->cellorient = "N";
      }
    }
    else {
      theCell->cellorient = rows[y_pos].siteorient;
    }
  }

  return true;
}

// lxm:只对grid操作
bool circuit::relegal_paint_pixel(cell* theCell, int x_pos, int y_pos) {
  assert(theCell->isPlaced == false);
  macro* theMacro = &macros[theCell->type];
  int x_step = (int)ceil(theCell->width / wsite);
  int y_step = (int)ceil(theCell->height / rowHeight);

  theCell->x_pos = x_pos;
  theCell->y_pos = y_pos;
  theCell->x_coord = x_pos * wsite;
  theCell->y_coord = y_pos * rowHeight;
  theCell->isPlaced = true;
  for(int i = y_pos; i < y_pos + y_step; i++) {
    for(int j = x_pos; j < x_pos + x_step; j++) {
      if(grid[i][j].linked_cell != NULL) {
        cerr << "theCell->name : " << theCell->name << endl;
        cerr << "relegal Can't paint grid [" << i << "][" << j << "] !!!"
             << endl;
        if(theCell->inGroup)
          cerr << " group name : " << groups[grid[i][j].group].name << endl;
        cerr << " Cell name : " << grid[i][j].linked_cell->name
             << " already occupied grid" << endl;
        exit(2);
        return false;
      }
      else {
        grid[i][j].linked_cell = theCell;
        grid[i][j].util = 1.0;
      }
    }
  }
  if(!init_lg_flag) {
    if(max_cell_height > 1) {
      if(y_step % 2 == 1) {
        if(rows[y_pos].top_power != theMacro->top_power)
          theCell->cellorient = "FS";
        else
          theCell->cellorient = "N";
      }
    }
    else {
      theCell->cellorient = rows[y_pos].siteorient;
    }
  }

  return true;
}

// lxm:更新当前cell的坐标信息(init_x_coord,init_y_coord不改)、朝向并改为placed，更新cell占据的grid
bool circuit::update_pixel(cell* theCell, int x_pos, int y_pos) {
  // assert(theCell->isPlaced == true);
  macro* theMacro = &macros[theCell->type];
  int x_step = (int)ceil(theCell->width / wsite);
  int y_step = (int)ceil(theCell->height / rowHeight);

  for(int i = theCell->y_pos; i < theCell->y_pos + y_step; i++) {
    for(int j = theCell->x_pos; j < theCell->x_pos + x_step; j++) {
      grid[i][j].linked_cell = NULL;
      grid[i][j].util = 0;
    }
  }

  theCell->x_pos = x_pos;
  theCell->y_pos = y_pos;
  theCell->x_coord = x_pos * wsite;
  theCell->y_coord = y_pos * rowHeight;
  for(int i = y_pos; i < y_pos + y_step; i++) {
    for(int j = x_pos; j < x_pos + x_step; j++) {
      if(grid[i][j].linked_cell != NULL) {
        cerr << " Can't paint grid [" << i << "][" << j << "] !!!" << endl;
        cerr << " group name : " << groups[grid[i][j].group].name << endl;
        cerr << " Cell name : " << grid[i][j].linked_cell->name
             << " already occupied grid" << endl;
        exit(2);
        return false;
      }
      else {
        grid[i][j].linked_cell = theCell;
        grid[i][j].util = 1.0;
      }
    }
  }

  if(!init_lg_flag) {
    if(max_cell_height > 1) {
      if(y_step % 2 == 1) {
        if(rows[y_pos].top_power != theMacro->top_power)
          theCell->cellorient = "FS";
        else
          theCell->cellorient = "N";
      }
    }
    else {
      theCell->cellorient = rows[y_pos].siteorient;
    }
  }

  return true;
}