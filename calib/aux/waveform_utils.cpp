#include "waveform_utils.h"
#include <map>
#include <algorithm>
#include <cmath>

short mode_of(const std::vector<short>& v){
    std::map<short, int> c;

    for (short x : v) c[x]++;
    return std::max_element(c.begin(), c.end(),
    [](const auto& a, const auto& b){
        return a.second < b.second; 
    })->first;
}

double compute_baseline(const std::vector<short>& adc, int c, int t, short mode){
    double sum = 0; int cnt = 0; int i = 0; int sz = adc.size();
    while(i < sz){
        if(adc[i] >= mode-c && adc[i] <= mode+c){sum+=adc[i]; cnt++; i+=t;}
        else i++;
    }
    return cnt > 0 ? sum / cnt : 0.0;
}

double noise(const std::vector<short>& adc, short n, double baseline){
    double sqr_dif = 0; int i = 0; double noise = 0;
    while(i < n){
        double d = (adc[i] - baseline);
        sqr_dif += d*d;
        i++;
    }
    noise = std::sqrt(sqr_dif / n);
    return noise;
}