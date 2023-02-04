//
// Created by mac on 2023/2/1.
//

#include "GameSimulator.h"

bool sig_caught = false;

int main() {
    GA ga{};
    ga.Initialize();
    auto start = high_resolution_clock::now();
    ga.Load("out.bin");
    while(true) {
        ga.Generation();
        auto end = high_resolution_clock::now();
        auto dur = duration_cast<minutes>(end - start);
        if (dur.count() > 50) break;
    }
    ga.Save("out.bin");
    return 0;
}