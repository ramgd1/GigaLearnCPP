#pragma once
#include "../Framework.h"

namespace GGL {
	struct Timer {
		std::chrono::steady_clock::time_point startTime;

		Timer() {
			Reset();
		}

		// Returns elapsed time in seconds (use steady_clock throughout for Linux/GCC)
		double Elapsed() {
			auto endTime = std::chrono::steady_clock::now();
			std::chrono::duration<double> elapsed = endTime - startTime;
			return elapsed.count();
		}

		void Reset() {
			startTime = std::chrono::steady_clock::now();
		}
	};
}