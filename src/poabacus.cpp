#include "circuit.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>  // For debugging
#include <omp.h>     // 使用OpenMP代替mutex
#include <map>
#include <atomic>
#include <set>  // 添加set头文件

using namespace opendp;
using std::abs;
using std::max;
using std::min;
using std::numeric_limits;
using std::pair;
using std::sort;
using std::vector;

void circuit::initializeSubRows() {
  sub_rows_vector.clear();
  for(size_t i = 0; i < rows.size(); ++i) {
    const auto& current_row = rows[i];
    vector< pair< int, int > >
        gaps;  // 存储由固定单元形成的障碍区域（wsite对齐后）

    // 定义当前行的物理X边界 (wsite对齐)，相对于core area
    int row_physical_X_start = current_row.origX - core.xLL;

    int row_physical_X_end =
        row_physical_X_start + current_row.numSites * wsite;

    // 当前行的y坐标，相对于core area
    int current_row_y = current_row.origY - core.yLL;

    for(const auto& fix_cell : fix_cells) {
      // 计算固定单元的Y坐标范围（相对于core area）
      double fix_cell_y_start = fix_cell->y_coord - core.yLL;
      double fix_cell_y_end = fix_cell_y_start + fix_cell->height;

      // 当前行的Y坐标范围（相对于core area）
      double current_row_y_start = current_row_y;
      double current_row_y_end = current_row_y + rowHeight;

      // 检查固定单元是否与当前行在Y方向上重叠
      if(current_row_y_start < fix_cell_y_end &&
         current_row_y_end > fix_cell_y_start) {
        // 计算固定单元的x坐标（相对于core area）
        double actual_fix_x =
            static_cast< double >(fix_cell->x_coord - core.xLL);
        double actual_fix_width = static_cast< double >(fix_cell->width);

        // 计算固定单元的wsite对齐后的起始和结束x坐标
        int gap_start = static_cast< int >(floor(actual_fix_x / wsite) * wsite);
        int gap_end = static_cast< int >(
            ceil((actual_fix_x + actual_fix_width) / wsite) * wsite);

        // 如果固定单元宽度大于0且对齐后起始和结束相同，则至少扩展到下一个wsite边界
        if(fix_cell->width > 0 && gap_end == gap_start) {
          gap_end = gap_start + wsite;
        }

        // 只考虑那些与当前行物理范围有重叠的固定单元
        if(gap_start < row_physical_X_end && gap_end > row_physical_X_start) {
          gap_start = max(gap_start, row_physical_X_start);
          gap_end = min(gap_end, row_physical_X_end);
          if(gap_end > gap_start) {
            gaps.push_back({gap_start, gap_end});
          }
        }
      }
    }

    sort(gaps.begin(), gaps.end());

    vector< pair< int, int > > merged_gaps;
    for(const auto& gap : gaps) {
      if(merged_gaps.empty() || merged_gaps.back().second < gap.first) {
        merged_gaps.push_back(gap);
      }
      else {
        merged_gaps.back().second = max(merged_gaps.back().second, gap.second);
      }
    }

    int last_end = row_physical_X_start;
    for(const auto& gap : merged_gaps) {
      if(gap.first > last_end) {
        SubRow sub_row;
        sub_row.start_x = last_end;
        sub_row.end_x = gap.first;
        sub_row.y_pos = current_row_y;  // 使用相对于core area的y坐标
        sub_row.remaining_width = sub_row.end_x - sub_row.start_x;
        if(sub_row.remaining_width > 0) {
          sub_rows_vector.push_back(sub_row);
        }
      }
      last_end = max(last_end, gap.second);
    }

    // 处理最后一个gap到行尾的剩余空间
    if(last_end < row_physical_X_end) {
      SubRow sub_row;
      sub_row.start_x = last_end;
      sub_row.end_x = row_physical_X_end;
      sub_row.y_pos = current_row_y;  // 使用相对于core area的y坐标
      sub_row.remaining_width = sub_row.end_x - sub_row.start_x;
      if(sub_row.remaining_width > 0) {
        sub_rows_vector.push_back(sub_row);
      }
    }
  }
}

void circuit::partitionCells() {
  if(thread_num <= 0) thread_num = 1;
  int N = static_cast< int >(sqrt(thread_num));
  if(N == 0) N = 1;
  int M = thread_num / N;
  if(M == 0) M = 1;

  double die_width = die.xUR - die.xLL;
  double die_height = die.yUR - die.yLL;
  if(die_width <= 0) die_width = 1.0;
  if(die_height <= 0) die_height = 1.0;

  double partition_width = die_width / M;
  double partition_height = die_height / N;
  if(partition_width <= 0) partition_width = die_width;
  if(partition_height <= 0) partition_height = die_height;

  tile_cells.assign(thread_num, vector< cell* >());

  for(size_t i = 0; i < cells.size(); ++i) {
    cell* current_cell = &cells[i];
    if(current_cell->isFixed) continue;

    int h_idx = static_cast< int >((current_cell->y_coord - die.yLL) /
                                   partition_height);
    int v_idx =
        static_cast< int >((current_cell->x_coord - die.xLL) / partition_width);

    h_idx = max(0, min(h_idx, N - 1));
    v_idx = max(0, min(v_idx, M - 1));

    int tile_idx = h_idx * M + v_idx;
    if(tile_idx >= 0 && tile_idx < thread_num) {
      tile_cells[tile_idx].push_back(current_cell);
    }
  }
}

void circuit::addCellToCluster(AbacusCluster& cluster, cell* theCell) {
  cluster.cells.push_back(theCell);
  // 权重可以使用单元面积或1，目前使用面积
  double cell_weight = theCell->pins.size();
  if(cell_weight <= 0) cell_weight = 1.0;  // 避免零或负权重

  cluster.total_weight += cell_weight;
  cluster.q_value +=
      cell_weight * (theCell->init_x_coord - cluster.total_width);
  cluster.total_width += theCell->width;
}

void circuit::mergeClusters(AbacusCluster& target_cluster,
                            AbacusCluster& source_cluster) {
  if(source_cluster.total_weight <= 0) return;  // 避免除以零或合并空簇

  // 更新合并后簇的属性
  target_cluster.q_value +=
      source_cluster.q_value -
      source_cluster.total_weight * target_cluster.total_width;
  target_cluster.total_weight += source_cluster.total_weight;
  target_cluster.total_width += source_cluster.total_width;
  target_cluster.cells.insert(target_cluster.cells.end(),
                              source_cluster.cells.begin(),
                              source_cluster.cells.end());

  // 清空源簇
  source_cluster.cells.clear();
  source_cluster.total_width = 0;
  source_cluster.total_weight = 0;
  source_cluster.q_value = 0;
}

void circuit::collapseCluster(AbacusCluster& cluster, const SubRow& row) {
  // 检查簇是否为空
  if(cluster.total_weight <= 0) {
    cluster.position = row.start_x;
    return;
  }

  // 计算最优位置 (按质量加权的初始位置)
  double optimal_pos = cluster.q_value / cluster.total_weight;

  // 确保位置在行边界内且对齐到site

  optimal_pos = round(optimal_pos / wsite) * wsite;

  // 设置簇位置，保证不超出行边界
  cluster.position = max((double)row.start_x, optimal_pos);
  cluster.position =
      min(cluster.position, (double)row.end_x - cluster.total_width);

  // 应用最终的site对齐（如果在边界处需要调整）
  if(cluster.position != optimal_pos) {
    cluster.position = round(cluster.position / wsite) * wsite;
    // 确保不会因取整导致超出行边界
    cluster.position = max((double)row.start_x, cluster.position);
    cluster.position =
        min(cluster.position, (double)row.end_x - cluster.total_width);
  }
}

void circuit::initialize_sub_row_locks() {
  if(!sub_rows_vector.empty()) {
    sub_row_locks.resize(sub_rows_vector.size());
    for(size_t i = 0; i < sub_rows_vector.size(); ++i) {
      omp_init_lock(&sub_row_locks[i]);
    }
  }
}

void circuit::destroy_sub_row_locks() {
  if(!sub_row_locks.empty()) {
    for(size_t i = 0; i < sub_row_locks.size(); ++i) {
      omp_destroy_lock(&sub_row_locks[i]);
    }
    sub_row_locks.clear();
  }
}

void circuit::parallelAbacus() {
  // 初始化子行和分区
  initializeSubRows();
  initialize_sub_row_locks();  // Initialize locks here
  std::cout << "Abacus: Initialized " << sub_rows_vector.size() << " sub-rows."
            << std::endl;
  partitionCells();
  std::cout << "Abacus: Partitioned cells into " << tile_cells.size()
            << " tiles." << std::endl;

  int total_cells_to_place = 0;
  for(const auto& tile : tile_cells) {
    total_cells_to_place += tile.size();
  }
  std::cout << "Abacus: Total cells to place: " << total_cells_to_place
            << std::endl;

  std::vector< std::vector< cell* > > per_thread_unplaced_cells(thread_num);
  std::atomic< int > cells_placed_counter(0);

  if(thread_num == 1) {
    for(size_t i = 0; i < tile_cells.size(); ++i) {
      vector< cell* >& current_tile_cells = tile_cells[i];
      sort(
          current_tile_cells.begin(), current_tile_cells.end(),
          [](const cell* a, const cell* b) { return a->x_coord < b->x_coord; });
      for(cell* theCell : current_tile_cells) {
        if(theCell->isFixed || theCell->isPlaced) {
          cells_placed_counter++;
          continue;
        }
        double best_cost = numeric_limits< double >::max();
        SubRow* best_sub_row = nullptr;
        AbacusCluster best_trial_cluster;
        for(size_t r_idx = 0; r_idx < sub_rows_vector.size(); ++r_idx) {
          SubRow& current_sub_row = sub_rows_vector[r_idx];
          double y_disp =
              abs(current_sub_row.y_pos - theCell->init_y_coord) / wsite;
          if(y_disp > displacement) {
            continue;
          }
          if(theCell->width > current_sub_row.remaining_width) {
            continue;
          }
          AbacusCluster trial_cluster;
          double cost = tryInsertCell(theCell, current_sub_row, trial_cluster);
          if(cost < best_cost) {
            best_cost = cost;
            best_sub_row = &current_sub_row;
            best_trial_cluster = trial_cluster;
          }
        }
        if(best_sub_row) {
          placeCellAndReoptimizeRow(*best_sub_row, theCell);
          int current_count = ++cells_placed_counter;
          if(current_count % 5000 == 0) {
            std::cout << "Abacus: Placed " << current_count << " cells ("
                      << (current_count * 100.0 / total_cells_to_place) << "%)"
                      << std::endl;
          }
        }
        else {
          per_thread_unplaced_cells[0].push_back(theCell);
        }
      }
    }
  }
  else {
#pragma omp parallel num_threads(thread_num)
    {
      int tid = omp_get_thread_num();
#pragma omp for schedule(dynamic, 1)
      for(int i = 0; i < (int)tile_cells.size(); ++i) {
        vector< cell* >& current_tile_cells = tile_cells[i];
        sort(current_tile_cells.begin(), current_tile_cells.end(),
             [](const cell* a, const cell* b) {
               return a->x_coord < b->x_coord;
             });
        int local_counter = 0;
        for(cell* theCell : current_tile_cells) {
          if(theCell->isFixed || theCell->isPlaced) continue;

          double min_total_cost = numeric_limits< double >::max();
          size_t best_r_idx =
              static_cast< size_t >(-1);  // Store index of the best row
          AbacusCluster best_trial_cluster_for_placement;

          for(size_t r_idx = 0; r_idx < sub_rows_vector.size(); ++r_idx) {
            omp_set_lock(&sub_row_locks[r_idx]);  // Acquire lock BEFORE
                                                  // accessing sub_row

            SubRow& current_sub_row =
                sub_rows_vector[r_idx];  // Safe access under lock
            double y_disp =
                abs(current_sub_row.y_pos - theCell->init_y_coord) / wsite;

            if(y_disp > displacement) {
              omp_unset_lock(&sub_row_locks[r_idx]);  // Release lock
              continue;
            }

            // Critical read of remaining_width under lock
            if(theCell->width > current_sub_row.remaining_width) {
              omp_unset_lock(&sub_row_locks[r_idx]);  // Release lock
              continue;
            }

            AbacusCluster trial_cluster;
            // tryInsertCell now safely accesses current_sub_row (including its
            // .clusters)
            double current_total_cost =
                tryInsertCell(theCell, current_sub_row, trial_cluster);

            if(current_total_cost < min_total_cost) {
              min_total_cost = current_total_cost;
              best_r_idx = r_idx;  // Store the index
              best_trial_cluster_for_placement =
                  trial_cluster;  // This is a copy
            }
            omp_unset_lock(
                &sub_row_locks[r_idx]);  // Release lock for this sub_row
          }

          if(best_r_idx !=
             static_cast< size_t >(-1)) {  // A suitable row was found
            bool placed_successfully = false;
            omp_set_lock(&sub_row_locks[best_r_idx]);  // Acquire lock for the
                                                       // chosen best_sub_row

            // Re-check remaining_width, as row state might have changed by
            // another cell's placement
            if(theCell->width <= sub_rows_vector[best_r_idx].remaining_width) {
              placeCellAndReoptimizeRow(sub_rows_vector[best_r_idx], theCell);
              placed_successfully = true;
              int current_count = ++cells_placed_counter;
              if(current_count % 5000 == 0) {
#pragma omp critical(print_progress_general)
                {
                  std::cout << "Abacus: Placed " << current_count << " cells ("
                            << (current_count * 100.0 / total_cells_to_place)
                            << "%)" << std::endl;
                }
              }
            }
            omp_unset_lock(&sub_row_locks[best_r_idx]);

            if(!placed_successfully) {
              per_thread_unplaced_cells[tid].push_back(theCell);
            }
          }
          else {  // No suitable row found
            per_thread_unplaced_cells[tid].push_back(theCell);
          }
          local_counter++;
          if(local_counter % 1000 == 0) {
#pragma omp critical(print_progress_thread)
            {
              std::cout << "Thread " << tid << " processed " << local_counter
                        << " cells out of " << current_tile_cells.size()
                        << std::endl;
            }
          }
        }
      }
    }
  }

  // Consolidate unplaced cells from all threads
  std::vector< cell* > cells_to_place_sequentially;
  for(const auto& thread_list : per_thread_unplaced_cells) {
    cells_to_place_sequentially.insert(cells_to_place_sequentially.end(),
                                       thread_list.begin(), thread_list.end());
  }

  std::cout << "Abacus: Parallel phase completed. "
            << cells_placed_counter.load() << " cells placed." << std::endl;
  std::cout << "Abacus: " << cells_to_place_sequentially.size()
            << " cells need sequential placement." << std::endl;

  if(!cells_to_place_sequentially.empty()) {
    int seq_placed = 0;
    for(cell* theCell : cells_to_place_sequentially) {
      if(theCell->isPlaced) continue;
      double best_cost = numeric_limits< double >::max();
      SubRow* best_sub_row = nullptr;
      AbacusCluster best_trial_cluster;
      for(size_t r_idx = 0; r_idx < sub_rows_vector.size(); ++r_idx) {
        // 获取锁以安全访问子行
        omp_set_lock(&sub_row_locks[r_idx]);

        SubRow& current_sub_row = sub_rows_vector[r_idx];
        double y_disp =
            abs(current_sub_row.y_pos - theCell->init_y_coord) / wsite;
        bool should_skip = false;

        if(y_disp > displacement) {
          should_skip = true;
        }
        else if(theCell->width > current_sub_row.remaining_width) {
          should_skip = true;
        }

        if(should_skip) {
          omp_unset_lock(&sub_row_locks[r_idx]);
          continue;
        }

        AbacusCluster trial_cluster;
        double cost =
            tryInsertCell(theCell, current_sub_row, trial_cluster) * 0.9;

        if(cost < best_cost) {
          best_cost = cost;
          best_sub_row = &current_sub_row;
          best_trial_cluster = trial_cluster;
        }

        omp_unset_lock(&sub_row_locks[r_idx]);
      }

      if(best_sub_row) {
        size_t row_index = best_sub_row - &sub_rows_vector[0];
        if(row_index < sub_row_locks.size()) {
          omp_set_lock(&sub_row_locks[row_index]);

          // 重新检查条件，因为其状态可能已经改变
          if(theCell->width <= best_sub_row->remaining_width) {
            placeCellAndReoptimizeRow(*best_sub_row, theCell);
            seq_placed++;
            if(seq_placed % 1000 == 0) {
              std::cout << "Abacus: Sequentially placed " << seq_placed
                        << " cells out of "
                        << cells_to_place_sequentially.size() << std::endl;
            }
          }

          omp_unset_lock(&sub_row_locks[row_index]);
        }
      }
    }
    std::cout << "Abacus: Sequential phase completed. " << seq_placed
              << " additional cells placed." << std::endl;
  }

// 确保所有处理都完成后再释放资源
#pragma omp barrier

  // 安全处理剩余单元
  handleRemainingCells();

  // 安全销毁锁
  try {
    destroy_sub_row_locks();  // Destroy locks at the end
  }
  catch(const std::exception& e) {
    std::cerr << "Warning: Exception during lock destruction: " << e.what()
              << std::endl;
    // 继续执行，避免中断程序
  }
}

double circuit::tryInsertCell(cell* theCell, SubRow& row,
                              AbacusCluster& temp_cluster) {
  temp_cluster = AbacusCluster();
  addCellToCluster(temp_cluster, theCell);

  double initial_x_disp = abs(theCell->init_x_coord - theCell->x_coord) /
                          wsite;  // This is effectively 0 for x to start
  double y_disp_for_check = abs(row.y_pos - theCell->init_y_coord) / wsite;
  if(initial_x_disp + y_disp_for_check > displacement) {
    return numeric_limits< double >::max();
  }

  // Position based on current x_coord for trial, then clamp and align
  temp_cluster.position = max((double)row.start_x, (double)theCell->x_coord);
  temp_cluster.position =
      min(temp_cluster.position, (double)row.end_x - temp_cluster.total_width);
  temp_cluster.position = round(temp_cluster.position / wsite) * wsite;
  temp_cluster.position = max((double)row.start_x, temp_cluster.position);

  if(temp_cluster.position + temp_cluster.total_width > row.end_x) {
    return numeric_limits< double >::max();  // Cannot fit
  }

  // Simulate merging for cost estimation (simplified from full DP)
  bool merged = false;
  AbacusCluster cluster_for_cost_calc =
      temp_cluster;  // Operate on a copy for cost calculation

  for(int i = (int)row.clusters.size() - 1; i >= 0; --i) {
    const AbacusCluster& existing_cluster = row.clusters[i];
    if(cluster_for_cost_calc.position <
           existing_cluster.position + existing_cluster.total_width &&
       cluster_for_cost_calc.position + cluster_for_cost_calc.total_width >
           existing_cluster.position) {
      AbacusCluster temp_merged = existing_cluster;  // Copy existing
      mergeClusters(
          temp_merged,
          cluster_for_cost_calc);  // Merge original temp_cluster (single cell)
      collapseCluster(temp_merged, row);  // Collapse the merged version
      cluster_for_cost_calc =
          temp_merged;  // This is now the cluster whose cost we evaluate
      merged = true;
      break;
    }
  }
  if(!merged) {
    collapseCluster(cluster_for_cost_calc,
                    row);  // Collapse the single-cell cluster if no merge
  }

  // Cost based on the position of the (potentially merged for estimation)
  // cluster
  double cost_x =
      abs(cluster_for_cost_calc.position - theCell->init_x_coord) / wsite;
  double cost_y = abs(row.y_pos - theCell->init_y_coord) / wsite;

  // Final check on estimated displacement for this specific configuration
  // The original displacement check is still important for hard limits.
  if(cost_x + cost_y >
     displacement * 1.05) {  // Allow a small tolerance if needed, or keep as >
                             // displacement
    return numeric_limits< double >::max();
  }

  const double DISPLACEMENT_PENALTY_FACTOR = 1.5;
  return DISPLACEMENT_PENALTY_FACTOR * (cost_x + cost_y);
}

// Helper function to get all unique cells for a row, including a potential new
// cell
namespace {  // Use an anonymous namespace to make it internal
std::vector< cell* > getAllCellsForRow(SubRow& row, cell* newCell) {
  std::vector< cell* > cellsInRow;
  std::map< cell*, bool > cellSet;  // To handle unique cells

  // Add newCell first if it exists and is not null
  if(newCell) {
    // Ensure newCell is not already in the row via another cluster
    bool already_present = false;
    for(const auto& cluster : row.clusters) {
      for(const auto* c_in_cluster : cluster.cells) {
        if(c_in_cluster == newCell) {
          already_present = true;
          break;
        }
      }
      if(already_present) break;
    }
    if(!already_present) {
      cellsInRow.push_back(newCell);
      cellSet[newCell] = true;
    }
  }

  // Add cells from existing clusters, ensuring uniqueness
  for(AbacusCluster& cluster : row.clusters) {
    for(cell* c : cluster.cells) {
      // Ensure the cell is not null and not already added
      if(c && cellSet.find(c) == cellSet.end()) {
        cellsInRow.push_back(c);
        cellSet[c] = true;
      }
    }
  }
  return cellsInRow;
}
}  // end anonymous namespace

void circuit::placeCellAndReoptimizeRow(
    SubRow& row,
    cell* newCellToInsert /* can be nullptr if just re-placing the existing row */) {
  std::vector< cell* > allCells = getAllCellsForRow(row, newCellToInsert);

  if(allCells.empty()) {
    row.remaining_width =
        row.end_x -
        row.start_x;       // Ensure remaining_width is correct for empty row
    row.clusters.clear();  // Ensure clusters are cleared if no cells
    return;
  }

  // Sort cells by their initial x-coordinate (classic Abacus DP requirement)
  sort(allCells.begin(), allCells.end(), [](const cell* a, const cell* b) {
    if(!a || !b) return false;  // Should not happen with proper cell management
    return a->init_x_coord < b->init_x_coord;
  });

  row.clusters.clear();  // Reset the row's clusters before DP

  for(cell* currentCell : allCells) {
    if(!currentCell) continue;  // Skip null cells

    AbacusCluster currentCellCluster;
    addCellToCluster(currentCellCluster, currentCell);
    currentCellCluster.position =
        currentCell
            ->x_coord;  // Use current x_coord as initial desired position

    while(!row.clusters.empty()) {
      AbacusCluster& lastPlacedCluster = row.clusters.back();
      AbacusCluster trialCluster = currentCellCluster;
      collapseCluster(trialCluster,
                      row);  // Get its optimal position if placed now

      if(trialCluster.position <
         lastPlacedCluster.position + lastPlacedCluster.total_width) {
        AbacusCluster mergedClusterCandidate = lastPlacedCluster;
        row.clusters.pop_back();
        mergeClusters(mergedClusterCandidate, currentCellCluster);
        currentCellCluster = mergedClusterCandidate;
      }
      else {
        break;
      }
    }
    collapseCluster(currentCellCluster, row);
    row.clusters.push_back(currentCellCluster);
  }

  // Final pass: Adjust all cluster positions to ensure no overlaps and assign
  // cell coordinates
  double current_pos_marker = row.start_x;

  for(AbacusCluster& cluster : row.clusters) {
    // 确保簇的起始位置大于等于当前标记（前一个簇的结束位置）
    cluster.position = max(cluster.position, current_pos_marker);

    // 对齐到站点网格
    cluster.position = floor(cluster.position / wsite) * wsite;

    // 确保位置在行边界内
    cluster.position =
        max(static_cast< double >(row.start_x), cluster.position);

    // 确保整个簇在行边界内
    if(cluster.position + cluster.total_width > row.end_x) {
      cluster.position = row.end_x - cluster.total_width;
      // 再次对齐并防止超出左边界
      cluster.position = floor(cluster.position / wsite) * wsite;
      cluster.position =
          max(static_cast< double >(row.start_x), cluster.position);
    }

    // 放置簇中的所有单元格
    double x_in_cluster = cluster.position;
    for(cell* c : cluster.cells) {
      if(!c) continue;

      // 直接分配精确的位置，不重复计算
      c->x_coord = x_in_cluster;
      c->y_coord = row.y_pos;

      // 确保Y坐标对齐
      c->y_coord = floor(c->y_coord / rowHeight) * rowHeight;
      c->y_coord = std::max(static_cast< double >(c->y_coord),
                            static_cast< double >(core.yLL));

      // 进行Y边界检查
      if(c->y_coord + c->height > core.yUR) {
        c->y_coord = core.yUR - c->height;
        c->y_coord = floor(c->y_coord / rowHeight) * rowHeight;
        c->y_coord = std::max(static_cast< double >(c->y_coord),
                              static_cast< double >(core.yLL));
      }

      // 更新位置信息
      c->x_pos = static_cast< int >(floor(c->x_coord / wsite));
      c->y_pos = static_cast< int >(floor(c->y_coord / rowHeight));
      c->isPlaced = true;

      // 下一个单元直接放在当前单元后面
      x_in_cluster += c->width;
    }

    // 更新当前位置标记为簇向上对齐到wsite后的结束位置
    double actual_cluster_end = cluster.position + cluster.total_width;
    current_pos_marker =
        std::min(static_cast< double >(row.end_x),
                 ceil(actual_cluster_end / static_cast< double >(wsite)) *
                     static_cast< double >(wsite));
  }

  // 更新行的剩余宽度
  double occupied_width = 0.0;
  for(const auto& cluster : row.clusters) {
    occupied_width += cluster.total_width;
  }
  row.remaining_width = max(0.0, row.end_x - row.start_x - occupied_width);
}

void circuit::handleRemainingCells() {
  // 处理无法在并行阶段放置的单元
  vector< cell* > unplaced_cells;
  int total_unplaced = 0;

  for(const auto& current_cell : cells) {
    if(!current_cell.isFixed && !current_cell.isPlaced) {
      unplaced_cells.push_back(const_cast< cell* >(&current_cell));
      total_unplaced++;
    }
  }

  if(!unplaced_cells.empty()) {
    std::cout << "Abacus: Final cleanup - " << total_unplaced
              << " cells remain unplaced." << std::endl;

    // 计算已放置的单元总数
    int total_cells = cells.size();
    double placed_percentage =
        100.0 * (total_cells - total_unplaced) / total_cells;
    std::cout << "Abacus: Overall placement rate: " << placed_percentage << "%"
              << std::endl;

    int final_placed = 0;
    int max_attempts = 3;  // 最多尝试3轮放置

    // 为子行重新优化以更新剩余空间信息
    for(auto& row : sub_rows_vector) {
      placeCellAndReoptimizeRow(row, nullptr);
    }

    // 按宽度降序排序未放置的单元，优先处理大单元
    std::sort(unplaced_cells.begin(), unplaced_cells.end(),
              [](const cell* a, const cell* b) { return a->width > b->width; });

    for(int attempt = 0; attempt < max_attempts && !unplaced_cells.empty();
        attempt++) {
      std::cout << "Abacus: Final placement attempt " << (attempt + 1)
                << " for " << unplaced_cells.size() << " cells." << std::endl;

      vector< cell* > still_unplaced;
      still_unplaced.reserve(unplaced_cells.size());
      int attempt_placed = 0;

      // 在最后尝试阶段进一步增大位移限制
      double local_displacement = displacement * (0.9 + attempt * 0.1);

      for(cell* theCell : unplaced_cells) {
        if(theCell->isPlaced) continue;  // 跳过已放置的单元

        double best_cost = numeric_limits< double >::max();
        SubRow* best_sub_row = nullptr;
        AbacusCluster best_trial_cluster;

        // 对所有子行搜索
        for(size_t r_idx = 0; r_idx < sub_rows_vector.size(); ++r_idx) {
          SubRow& current_sub_row = sub_rows_vector[r_idx];

          // 计算y方向位移
          double y_disp =
              abs(current_sub_row.y_pos - theCell->init_y_coord) / wsite;
          // 使用逐渐放松的位移约束
          if(y_disp > local_displacement) {
            continue;
          }

          // 检查单元是否能放入该子行
          if(theCell->width > current_sub_row.remaining_width) {
            continue;
          }

          AbacusCluster trial_cluster;
          // 在最后一轮尝试中，降低位移代价的权重，使单元更容易放置
          double cost =
              (attempt == max_attempts - 1)
                  ? tryInsertCell(theCell, current_sub_row, trial_cluster) * 0.3
                  : tryInsertCell(theCell, current_sub_row, trial_cluster) *
                        (0.8 - attempt * 0.2);

          if(cost < best_cost) {
            best_cost = cost;
            best_sub_row = &current_sub_row;
            best_trial_cluster = trial_cluster;
          }
        }

        if(best_sub_row) {
          placeCellAndReoptimizeRow(*best_sub_row, theCell);
          attempt_placed++;
          final_placed++;

          // 每放置100个单元输出一次进度
          if(attempt_placed % 100 == 0) {
            std::cout << "Abacus: Final phase placed " << attempt_placed
                      << " cells in attempt " << (attempt + 1) << std::endl;
          }
        }
        else {
          // 无法放置，加入下一轮尝试的列表
          still_unplaced.push_back(theCell);
        }
      }

      // 更新未放置列表为本轮仍未放置的单元
      unplaced_cells = still_unplaced;

      std::cout << "Abacus: Attempt " << (attempt + 1) << " complete. Placed "
                << attempt_placed << " cells, " << still_unplaced.size()
                << " cells remain unplaced." << std::endl;

      // 如果没有新的单元被放置，提前结束
      if(attempt_placed == 0) {
        std::cout << "Abacus: No progress made in this attempt. Stopping final "
                     "placement."
                  << std::endl;
        break;
      }

      // 每轮后重新优化所有行以最大化可用空间
      for(auto& row : sub_rows_vector) {
        placeCellAndReoptimizeRow(row, nullptr);
      }
    }
  }

  // 最终检查和处理
  int final_unplaced = 0;
  std::vector< cell* > truly_unplaced_cells;

  for(const auto& current_cell : cells) {
    if(!current_cell.isFixed && !current_cell.isPlaced) {
      truly_unplaced_cells.push_back(const_cast< cell* >(&current_cell));
      final_unplaced++;
    }
  }

  // 最后检查是否有重叠
  bool has_overlap = false;
  std::vector< std::pair< cell*, cell* > > overlapping_pairs;

  // 使用扫描线算法检测重叠
  std::vector< std::pair< cell*, bool > > events;  // (单元, 是否为开始事件)

  for(auto& c : cells) {
    if(c.isFixed || !c.isPlaced) continue;
    events.push_back({&c, true});   // 开始事件
    events.push_back({&c, false});  // 结束事件
  }

  // 按 x 坐标排序事件
  std::sort(events.begin(), events.end(), [](const auto& a, const auto& b) {
    double x_a =
        a.second ? a.first->x_coord : a.first->x_coord + a.first->width;
    double x_b =
        b.second ? b.first->x_coord : b.first->x_coord + b.first->width;
    if(x_a != x_b) return x_a < x_b;
    return !a.second;  // 结束事件优先
  });

  // 使用扫描线检测重叠
  std::map< int, std::vector< cell* > > active_cells;  // y坐标 -> 活跃单元列表

  for(const auto& event : events) {
    cell* c = event.first;
    bool is_start = event.second;

    if(is_start) {
      // 开始事件：添加到活跃集合
      int y_pos = static_cast< int >(floor(c->y_coord / rowHeight));
      int height_in_rows = static_cast< int >(ceil(c->height / rowHeight));

      for(int row = y_pos; row < y_pos + height_in_rows; ++row) {
        auto& cells_in_row = active_cells[row];

        // 检查与活跃单元的重叠
        for(cell* other : cells_in_row) {
          // 检查 x 轴重叠
          if(c->x_coord < other->x_coord + other->width &&
             c->x_coord + c->width > other->x_coord) {
            // 检查 y 轴重叠
            if(c->y_coord < other->y_coord + other->height &&
               c->y_coord + c->height > other->y_coord) {
              has_overlap = true;
              overlapping_pairs.push_back({c, other});
            }
          }
        }

        cells_in_row.push_back(c);
      }
    }
    else {
      // 结束事件：从活跃集合中移除
      int y_pos = static_cast< int >(floor(c->y_coord / rowHeight));
      int height_in_rows = static_cast< int >(ceil(c->height / rowHeight));

      for(int row = y_pos; row < y_pos + height_in_rows; ++row) {
        auto& cells_in_row = active_cells[row];
        cells_in_row.erase(
            std::remove(cells_in_row.begin(), cells_in_row.end(), c),
            cells_in_row.end());
      }
    }
  }

  // 处理重叠
  if(has_overlap) {
    std::cout << "Abacus: WARNING - Detected " << overlapping_pairs.size()
              << " overlapping cell pairs. Fixing overlaps..." << std::endl;

    // 将重叠单元移至可用空间
    std::set< cell* > cells_to_fix;
    for(const auto& [cell1, cell2] : overlapping_pairs) {
      // 选择面积较小的单元进行移动
      if(cell1->width * cell1->height <= cell2->width * cell2->height) {
        cells_to_fix.insert(cell1);
      }
      else {
        cells_to_fix.insert(cell2);
      }
    }

    // 对需要修复的单元重新尝试放置
    int fixed_count = 0;

    // 创建按y坐标排序的行索引与实际y坐标的映射
    std::vector< std::pair< size_t, int > > rows_by_y;
    for(size_t i = 0; i < sub_rows_vector.size(); ++i) {
      rows_by_y.push_back({i, sub_rows_vector[i].y_pos});
    }

    for(cell* theCell : cells_to_fix) {
      theCell->isPlaced = false;  // 标记为未放置

      bool placed = false;

      // 按照与单元初始y坐标的距离排序行
      int init_y = theCell->init_y_coord;
      std::sort(rows_by_y.begin(), rows_by_y.end(),
                [init_y](const auto& a, const auto& b) {
                  return abs(a.second - init_y) < abs(b.second - init_y);
                });

      // 从最近的行开始尝试
      for(const auto& [row_idx, y_pos] : rows_by_y) {
        if(y_pos < core.yLL || y_pos + rowHeight > core.yUR) {
          continue;  // 跳过核心区域外的行
        }

        // 检查位移约束
        double y_disp = abs(y_pos - theCell->init_y_coord) / wsite;
        if(y_disp > displacement) {  // 允许略微超出位移限制
          continue;
        }

        // 获取锁以安全访问子行
        omp_set_lock(&sub_row_locks[row_idx]);

        // 检查是否有足够空间
        if(theCell->width <= sub_rows_vector[row_idx].remaining_width) {
          placeCellAndReoptimizeRow(sub_rows_vector[row_idx], theCell);
          if(theCell->isPlaced) {
            fixed_count++;
            placed = true;
            omp_unset_lock(&sub_row_locks[row_idx]);
            break;
          }
        }

        omp_unset_lock(&sub_row_locks[row_idx]);
      }

      // 如果仍未放置，尝试所有可能的子行（不考虑位移约束）
      if(!placed) {
        // 按照剩余宽度从大到小排序子行
        std::sort(rows_by_y.begin(), rows_by_y.end(),
                  [this](const auto& a, const auto& b) {
                    return sub_rows_vector[a.first].remaining_width >
                           sub_rows_vector[b.first].remaining_width;
                  });

        for(const auto& [row_idx, y_pos] : rows_by_y) {
          omp_set_lock(&sub_row_locks[row_idx]);

          if(theCell->width <= sub_rows_vector[row_idx].remaining_width) {
            placeCellAndReoptimizeRow(sub_rows_vector[row_idx], theCell);
            if(theCell->isPlaced) {
              fixed_count++;
              placed = true;
              omp_unset_lock(&sub_row_locks[row_idx]);
              break;
            }
          }

          omp_unset_lock(&sub_row_locks[row_idx]);
        }
      }
    }

    std::cout << "Abacus: Fixed " << fixed_count << " of "
              << cells_to_fix.size() << " overlapping cells." << std::endl;

    // 更新未放置单元数
    final_unplaced = 0;
    for(const auto& current_cell : cells) {
      if(!current_cell.isFixed && !current_cell.isPlaced) {
        final_unplaced++;
      }
    }
  }

  if(final_unplaced > 0) {
    double final_placement_rate =
        100.0 * (cells.size() - final_unplaced) / cells.size();
    std::cout << "Abacus: WARNING - " << final_unplaced
              << " cells remain unplaced. Final placement rate: "
              << final_placement_rate << "%" << std::endl;
  }
  else {
    std::cout << "Abacus: All cells successfully placed!" << std::endl;
  }
}