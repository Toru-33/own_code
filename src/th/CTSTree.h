#pragma once

#include <vector>
#include <iostream>
#include <assert.h>
#include <algorithm>
#include <future>
#include <unordered_set>
#include "PriorityQueue.h"
#include <cfloat>
#include <cmath>
namespace fzq {

	struct FPos {
		double x, y;
		int id;
		FPos() {}
		//FPos(double x, double y) :x(x), y(y) {}
		FPos(double x, double y, int id) :x(x), y(y), id(id) {}
	};

	class CTSTree
	{
		struct Pair {
			double dis;
			int index;
			Pair(int index, double dis) :dis(dis), index(index) {}
			Pair() :dis(DBL_MAX), index(-1)
			{

			}
			int push(int index, double dis) {
				if (dis < this->dis) {
					this->index = index;
					this->dis = dis;
					return 1;
				}
				return 0;
			}

			void reset() {
				dis = DBL_MAX;
				index = -1;
			}
		};

		struct Grid {
			std::vector<int> nodes;
			size_t key;

			Grid():key(-1){}
			void reset() {
				key = -1;
				nodes.clear();
			}
		};

		struct Element {
			int index;
			Pair p;
		};
	public:
		CTSTree();
		void init();

		inline std::vector<FPos>& getSinks() {
			return m_sinks;
		}

		std::pair<FPos, FPos> get();
		void add(FPos p);
	private:
		std::vector<FPos> m_sinks;
		std::vector<bool> m_valids;
		std::vector<Pair> m_pairs;//��֤�Ƿ���Ҫ�Ż�map
		std::vector<std::vector<Grid>> m_map;
		MutablePriorityQueue<Element> m_pq;
	//	std::unordered_set<int> m_dirty;

		double m_lx, m_ly, m_ux, m_uy;
		int m_countx, m_county;
		double m_width, m_height;

		void update(bool scale);

		inline double distance(int index1, int index2) {
			FPos p1 = m_sinks[index1], p2 = m_sinks[index2];
			return abs(p1.x - p2.x) + abs(p1.y - p2.y);
		}

		int getNearGrid(int x, int y);

		void updateWait(int index, int x, int y);

		bool remove(int index);

		std::pair<FPos, FPos> m_cur;
	};

}

