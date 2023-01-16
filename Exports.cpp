//
// Created by mac on 2023/1/14.
//
#include <iostream>
#include "ANN.h"
#include "GameSimulator.h"

#ifdef _MSC_VER
#define export_fn extern "C" __declspec( dllexport )
#else
#define export_fn extern "C"
#endif

export_fn bool Entry() {
    std::cout << "MadPodRacing C++" << std::endl;
    return true;
}

export_fn int TestMethod() {
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

export_fn intptr_t GSCreate(int podsPerSide, int totalLaps) {
    return (std::intptr_t) new GameSimulator(podsPerSide, totalLaps);
}
export_fn void GSSetup(intptr_t sim, Vec *start, int len) {
    auto gameSim = (GameSimulator *) sim;
    gameSim->Setup(std::vector(start, start + len));
}
export_fn bool GSTick(intptr_t sim) {
    return ((GameSimulator *) sim)->Tick();
}
export_fn Pod *GSGetPod(intptr_t sim, int idx) {
    return &((GameSimulator *) sim)->Pods[idx];
}
export_fn Vec *GSGetCP(intptr_t sim, int idx) {
    return &((GameSimulator *) sim)->Checkpoints[idx];
}

export_fn intptr_t GACreate() {
    return (std::intptr_t) new GAUsed;
}

export_fn void GAInitialize(intptr_t ga, int* nodes, int len) {
    std::array<int, ANNUsed::LayersCount> nodesArr{};
    for (int i = 0; i < len; i++) {
        nodesArr[i] = nodes[i];
    }
    ((GAUsed*)ga)->Initialize(nodesArr);
}
export_fn bool GATick(intptr_t ga) {
    return ((GAUsed*)ga)->Tick();
}
export_fn bool GAGeneration(intptr_t ga) {
    return ((GAUsed*)ga)->Generation();
}
export_fn void GAGenerationStart(intptr_t ga) {
    ((GAUsed*)ga)->GenerationStart();
}
export_fn void GAGenerationEnd(intptr_t ga) {
    ((GAUsed*)ga)->GenerationEnd();
}
export_fn Vec* GAGetCheckpoint(intptr_t ga, int idx) {
    return &((GAUsed*)ga)->Checkpoints[idx];
}
export_fn intptr_t GAGetSimulators(intptr_t ga, int idx) {
    return (intptr_t)&((GAUsed*)ga)->Checkpoints[idx];
}