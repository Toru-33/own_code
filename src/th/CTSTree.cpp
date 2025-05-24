#include "CTSTree.h"

namespace fzq {
CTSTree::CTSTree()
    : m_pq([](Element& p1, Element& p2) { return p1.p.dis < p2.p.dis; }) {}
void CTSTree::init() {
  double lx = DBL_MAX, ly = DBL_MAX, ux = DBL_MIN, uy = DBL_MIN;
  for(auto& v : m_sinks) {
    lx = std::min(lx, v.x);
    ly = std::min(ly, v.y);
    ux = std::max(ux, v.x);
    uy = std::max(uy, v.y);
  }
  lx = (long long)std::max(lx - 10.0, 0.0);
  ly = (long long)std::max(ly - 10.0, 0.0);
  ux = (long long)ux + 10;
  uy = (long long)uy + 10;

  // lx = ly = 0;
  double area = (ux - lx) * (uy - ly) / m_sinks.size();
  double unit = (long long)sqrt(area) + 1;
  unit *= 2;
  int countx = (ux - lx) / unit + 1;
  int county = (uy - ly) / unit + 1;

  m_lx = lx, m_ly = ly;
  m_ux = countx * unit + lx;
  m_uy = county * unit + ly;

  m_countx = countx;
  m_county = county;

  // std::cout << "sum:" << m_sinks.size() << " divide:" << m_countx << " " <<
  // m_county << std::endl;
  m_width = m_height = unit;

  m_map.resize(m_countx);
  for(auto& m : m_map) {
    m.clear();
    m.resize(m_county);
  }
  m_valids.resize(m_sinks.size());
  std::fill(m_valids.begin(), m_valids.end(), true);
  m_pairs.resize(m_sinks.size());
  std::fill(m_pairs.begin(), m_pairs.end(), Pair());
  m_pq.clear();
  update(false);
}

void CTSTree::update(bool scale) {
  if(scale) {
    m_width = std::min(m_width * 2, m_ux - m_lx);
    m_height = std::min(m_height * 2, m_uy - m_ly);
    m_countx = (m_ux - m_lx) / m_width + 1;
    m_county = (m_uy - m_ly) / m_height + 1;

    int j = 0;
    for(int i = 0; i < m_sinks.size(); ++i) {
      if(m_valids[i]) {
        m_sinks[j] = m_sinks[i];
        m_pairs[j] = m_pairs[i];
        m_valids[j] = true;
        ++j;
      }
    }
    assert(j > 0);
    m_sinks.resize(j);
    m_pairs.resize(j);
    m_valids.resize(j);

    m_map.resize(m_countx);
    for(int i = 0; i < m_countx; ++i) {
      m_map[i].resize(m_county);
      for(int j = 0; j < m_county; ++j) {
        m_map[i][j].reset();
      }
    }
    m_pq.clear();
  }
  // std::cout << "update map ;";
  int i = 0;
  for(auto& v : m_sinks) {
    m_map.at((v.x - m_lx) / m_width)
        .at((v.y - m_ly) / m_height)
        .nodes.emplace_back(i++);
  }
  // std::cout << "update map finished" << std::endl;
  for(int x = 0; x < m_countx; ++x) {
    for(int y = 0; y < m_county; ++y) {
      std::vector< int >*left = nullptr, *lefttop = nullptr, *leftbot = nullptr,
                       *bottom = nullptr;
      if(x > 0) left = &m_map[x - 1][y].nodes;
      if(y > 0) bottom = &m_map[x][y - 1].nodes;
      if(x > 0 && y < m_county - 1) lefttop = &m_map[x - 1][y + 1].nodes;
      if(x > 0 && y > 0) leftbot = &m_map[x - 1][y - 1].nodes;

      auto& g = m_map[x][y].nodes;
      // in-grid
      for(int i = 0; i < g.size(); ++i) {
        int index = g[i];
        for(int j = i + 1; j < g.size(); ++j) {
          m_pairs[index].push(g[j], distance(index, g[j]));
          // pairs[g[j]].push(index, dis);
        }
        if(left != nullptr) {
          for(auto& v : *left) {
            m_pairs[index].push(v, distance(index, v));
          }
        }
        if(lefttop != nullptr) {
          for(auto& v : *lefttop) {
            m_pairs[index].push(v, distance(index, v));
          }
        }
        if(leftbot != nullptr) {
          for(auto& v : *leftbot) {
            m_pairs[index].push(v, distance(index, v));
          }
        }
        if(bottom != nullptr) {
          for(auto& v : *bottom) {
            m_pairs[index].push(v, distance(index, v));
          }
        }
      }
    }
  }

  for(int x = 0; x < m_countx; ++x) {
    for(int y = 0; y < m_county; ++y) {
      int index = getNearGrid(x, y);
      if(index != -1) {
        m_map[x][y].key = m_pq.push(Element{index, m_pairs[index]});
      }
    }
  }
}

int CTSTree::getNearGrid(int x, int y) {
  auto& g = m_map[x][y].nodes;
  if(g.empty()) return -1;

  int index = g[0];
  double p = m_pairs[g[0]].dis;
  for(int i = 1; i < g.size(); ++i) {
    if(m_pairs[g[i]].dis < p) {
      index = g[i];
      p = m_pairs[g[i]].dis;
    }
  }
  return index;
}

std::pair< FPos, FPos > CTSTree::get() {
loop:
  auto v = m_pq.top();
  if(v.p.index == -1) {
    update(true);
    if(m_sinks.size() > 1) {
      goto loop;
    }
    else
      return {{-1, -1, -1}, {}};
  }
  else if(m_valids[v.index] == false) {
    auto& s = m_sinks[v.index];
    int x = (s.x - m_lx) / m_width, y = (s.y - m_ly) / m_height;
    auto& g = m_map[x][y].nodes;
    if(g.empty()) {
      m_pq.pop();
      m_map[x][y].key = -1;
      // std::cout << "pop:" << x << " " << y << ";";
      goto loop;
    }
    for(auto& index : g) {
      if(m_pairs[index].index != -1 &&
         m_valids[m_pairs[index].index] == false) {
        updateWait(index, x, y);
      }
    }
    int index = getNearGrid(x, y);
    assert(index != -1);
    m_pq.update(m_map[x][y].key, Element{index, m_pairs[index]});
    goto loop;
  }
  else if(m_valids[v.p.index] == false) {
    auto& s = m_sinks[v.index];
    int x = (s.x - m_lx) / m_width, y = (s.y - m_ly) / m_height;
    auto& g = m_map[x][y].nodes;

    assert(!g.empty());
    for(auto& index : g) {
      if(m_pairs[index].index != -1 &&
         m_valids[m_pairs[index].index] == false) {
        updateWait(index, x, y);
      }
    }
    // if(m_valids[v.index] == false &&  )
    int index = getNearGrid(x, y);
    m_pq.update(m_map[x][y].key, Element{index, m_pairs[index]});
    goto loop;
  }
  else {
    if(remove(v.index)) {
      auto& s = m_sinks[v.index];
      int x = (s.x - m_lx) / m_width, y = (s.y - m_ly) / m_height;
      m_map[x][y].key = -1;
      m_pq.pop();
      //	std::cout << "pop:" << x << " " << y;
    }

    // std::cout << count << std::endl;
    m_cur = {m_sinks[v.index], m_sinks[v.p.index]};
    assert(v.p.index == m_pairs[v.index].index);
    return {m_sinks[v.index], m_sinks[v.p.index]};
  }
}

void CTSTree::updateWait(int index, int x, int y) {
  auto& g = m_map[x][y].nodes;
  std::vector< int >*left = nullptr, *lefttop = nullptr, *leftbot = nullptr,
                   *bottom = nullptr;
  if(x > 0) left = &m_map[x - 1][y].nodes;
  if(y > 0) bottom = &m_map[x][y - 1].nodes;
  if(x > 0 && y < m_county - 1) lefttop = &m_map[x - 1][y + 1].nodes;
  if(x > 0 && y > 0) leftbot = &m_map[x - 1][y - 1].nodes;

  m_pairs[index].reset();

  int i;
  for(i = 0; i < g.size(); ++i) {
    if(g[i] == index) break;
  }
  for(int j = i + 1; j < g.size(); ++j) {
    m_pairs[index].push(g[j], distance(index, g[j]));
  }

  if(left != nullptr) {
    for(auto& v : *left) {
      m_pairs[index].push(v, distance(index, v));
    }
  }
  if(lefttop != nullptr) {
    for(auto& v : *lefttop) {
      m_pairs[index].push(v, distance(index, v));
    }
  }
  if(leftbot != nullptr) {
    for(auto& v : *leftbot) {
      m_pairs[index].push(v, distance(index, v));
    }
  }
  if(bottom != nullptr) {
    for(auto& v : *bottom) {
      m_pairs[index].push(v, distance(index, v));
    }
  }
}

bool CTSTree::remove(int index) {
  {
    m_valids[index] = false;
    auto& pos = m_sinks[index];
    auto& g = m_map[(pos.x - m_lx) / m_width][(pos.y - m_ly) / m_height].nodes;
    for(auto p = g.begin(); p != g.end(); ++p) {
      if(*p == index) {
        g.erase(p);
        break;
      }
    }
  }
  {
    int ip = m_pairs[index].index;
    assert(ip == m_pq.top().p.index);
    m_valids[ip] = false;
    auto& pos = m_sinks[ip];
    auto& g = m_map[(pos.x - m_lx) / m_width][(pos.y - m_ly) / m_height].nodes;
    for(auto p = g.begin(); p != g.end(); ++p) {
      if(*p == ip) {
        g.erase(p);
        break;
      }
    }
  }
  auto& s = m_sinks[index];
  int x = (s.x - m_lx) / m_width, y = (s.y - m_ly) / m_height;
  auto& g = m_map[x][y].nodes;
  for(int i = 0; i < g.size(); ++i) {
    int index = g[i];
    // assert(m_pairs[index].index != -1);
    if(m_pairs[index].index != -1 && m_valids[m_pairs[index].index] == false) {
      updateWait(index, x, y);
    }
  }
  int minindex = getNearGrid(x, y);
  if(minindex != -1) {
    m_pq.update(m_map[x][y].key, Element{minindex, m_pairs[minindex]});
    return false;
  }
  return true;
}

void CTSTree::add(FPos p) {
  if(!(p.x >= m_lx && p.x < m_ux && p.y >= m_ly && p.y < m_uy)) {
    std::cout << "warning, new node:(" << p.x << "," << p.y << ") exceed rect"
              << "(" << m_lx << "," << m_ux << ")-(" << m_ly << "," << m_uy
              << ")" << std::endl;
    p.x = std::min(std::max(m_lx, p.x), m_ux - 1.0);
    p.y = std::min(std::max(m_ly, p.y), m_uy - 1.0);
  }
  // assert(p.x >= m_lx && p.x < m_ux&& p.y >= m_ly && p.y < m_uy);

  m_sinks.emplace_back(p);
  m_pairs.emplace_back(-1, DBL_MAX);
  m_valids.emplace_back(true);

  int x = (p.x - m_lx) / m_width, y = (p.y - m_ly) / m_height;
  int index = m_sinks.size() - 1;
  assert(x >= 0 && x < m_map.size() && y >= 0 && y < m_map[0].size());
  m_map[x][y].nodes.emplace_back(index);

  {
    std::vector< int >*left = nullptr, *lefttop = nullptr, *leftbot = nullptr,
                     *bottom = nullptr;
    if(x > 0) left = &m_map[x - 1][y].nodes;
    if(y > 0) bottom = &m_map[x][y - 1].nodes;
    if(x > 0 && y < m_county - 1) lefttop = &m_map[x - 1][y + 1].nodes;
    if(x > 0 && y > 0) leftbot = &m_map[x - 1][y - 1].nodes;

    auto& g = m_map[x][y].nodes;

    // 同grid 更新末尾的自己
    for(int i = 0; i < g.size() - 1; ++i) {
      m_pairs[g[i]].push(index, distance(index, g[i]));
    }

    if(left != nullptr) {
      for(auto& v : *left) {
        m_pairs[index].push(v, distance(index, v));
      }
    }
    if(lefttop != nullptr) {
      for(auto& v : *lefttop) {
        m_pairs[index].push(v, distance(index, v));
      }
    }
    if(leftbot != nullptr) {
      for(auto& v : *leftbot) {
        m_pairs[index].push(v, distance(index, v));
      }
    }
    if(bottom != nullptr) {
      for(auto& v : *bottom) {
        m_pairs[index].push(v, distance(index, v));
      }
    }

    if(g.size() == 1 && m_map[x][y].key == -1) {
      // std::cout << "push:" << x << " " << y << "   ";
      m_map[x][y].key = m_pq.push(Element{index, m_pairs[index]});
    }
    else {
      int index = getNearGrid(x, y);
      m_pq.update(m_map[x][y].key, Element{index, m_pairs[index]});
      // std::cout << "upd:" << x << "," <<y<< "   ";
    }
  }
  {
    auto fun = [&](int x, int y) {
      int ok = 0;
      for(auto& v : m_map[x][y].nodes) {
        ok |= m_pairs[v].push(index, distance(index, v));
      }
      if(ok) {
        int index = getNearGrid(x, y);
        if(index != -1) {
          m_pq.update(m_map[x][y].key, Element{index, m_pairs[index]});
        }
        else {
          exit(-1);
        }
      }
    };

    if(x < m_countx - 1) {
      fun(x + 1, y);
    }
    if(y < m_county - 1) {
      fun(x, y + 1);
    }
    if(x < m_countx - 1 && y < m_county - 1) {
      fun(x + 1, y + 1);
    }
    if(x < m_countx - 1 && y > 0) {
      fun(x + 1, y - 1);
    }
  }
}
}  // namespace fzq
