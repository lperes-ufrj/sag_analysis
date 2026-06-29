#pragma once
#include <vector>

short mode_of(const std::vector<short>& v);
double compute_baseline(const std::vector<short>& adc, int c, int t, short mode);
double noise(const std::vector<short>& adc, short n, double baseline);
double compute_integral(const std::vector<short>& adc, double baseline, int n1, int n2);
void   compute_minmax(const std::vector<short>& adc, double baseline,
                      int n1, int n2, double& vmin, double& vmax);