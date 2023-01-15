//
// Created by mac on 2023/1/14.
//
#include <iostream>
#include "ANN.h"
#include "GameSimulator.h"

#ifdef _MSC_VER
#define export extern "C" __declspec( dllexport )
#else
#define export extern "C"
#endif

export bool Entry() {
    std::cout << "MadPodRacing C++" << std::endl;
    return true;
}

export int TestMethod() {
    std::cout << "Hello, World!" << std::endl;
    ANNUsed ann;
    ann.InitializeSpace({12, 8, 8, 4});
    ann.Randomize();
    ann.Compute();
    for (int i = 0; i < ann.NeuronCount; i++) {
        std::cout << ann.Neurons[i] << ' ';
    }
    std::cout << std::endl;
    return 20;
}

export intptr_t GSCreate(int podsPerSide, int totalLaps) {
    return (std::intptr_t)new GameSimulator(podsPerSide, totalLaps);
}
export void GSSetup(intptr_t sim, Vec* start, int len) {
    auto gameSim = (GameSimulator*)sim;
    gameSim->Setup(std::vector(start, start + len));
}
export bool GSTick(intptr_t sim) {
    return ((GameSimulator*)sim)->Tick();
}
export Pod* GSGetPod(intptr_t sim, int idx) {
    return &((GameSimulator*)sim)->Pods[idx];
}
export Vec* GSGetCP(intptr_t sim, int idx) {
    return &((GameSimulator*)sim)->Checkpoints[idx];
}
