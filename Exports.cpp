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
    ann.InitializeSpace(ANNUsed::DefaultNodes);
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
export_fn void GSSetup(intptr_t sim, intptr_t ga) {
    auto gameSim = (GameSimulator *) sim;
    gameSim->Setup((GAUsed*)ga);
}
export_fn bool GSTick(intptr_t sim) {
    return ((GameSimulator *) sim)->Tick();
}
export_fn Pod *GSGetPod(intptr_t sim, int idx) {
    return &((GameSimulator *) sim)->Pods[idx];
}
export_fn Vec *GSGetCP(intptr_t sim, int idx) {
    return &((GameSimulator *) sim)->GA->Checkpoints[idx];
}
export_fn void GSReset(intptr_t sim, intptr_t ga) {
    auto gameSim = (GameSimulator *) sim;
    gameSim->Reset((GAUsed*)ga);
}
export_fn double GSRecalculateFitness(intptr_t sim) {
    return ((GameSimulator *) sim)->RecalculateFitness();
}
export_fn double GSFitness(intptr_t sim) {
    return ((GameSimulator *) sim)->Fitness();
}
export_fn void GSSetupRandomANN(intptr_t sim) {
    auto& gs = *(GameSimulator*)sim;
    gs.SetANN(std::make_shared<ANNUsed>());
    gs.ANNController->InitializeSpace(ANNUsed::DefaultNodes);
    gs.ANNController->Randomize();
}

export_fn intptr_t GACreate() {
    return (std::intptr_t) new GAUsed();
}
export_fn void GAInitializeDefault(intptr_t ga) {
    ((GAUsed *) ga)->Initialize();
}
export_fn void GAInitialize(intptr_t ga, const int *nodes, int len) {
    std::array<int, ANNUsed::LayersCount> nodesArr{};
    for (int i = 0; i < len; i++) {
        nodesArr[i] = nodes[i];
    }
    ((GAUsed *) ga)->Initialize(nodesArr);
}
export_fn bool GATick(intptr_t ga) {
    return ((GAUsed *) ga)->Tick();
}
export_fn bool GAGeneration(intptr_t ga) {
    return ((GAUsed *) ga)->Generation();
}
export_fn void GAGenerationStart(intptr_t ga) {
    ((GAUsed *) ga)->GenerationStart();
}
export_fn void GAGenerationEnd(intptr_t ga) {
    ((GAUsed *) ga)->GenerationEnd();
}
export_fn Vec *GAGetCheckpoint(intptr_t ga, int idx) {
    return &((GAUsed *) ga)->Checkpoints[idx];
}
export_fn intptr_t GAGetSimulator(intptr_t ga, int idx) {
    return (intptr_t) &((GAUsed *) ga)->Simulators[idx];
}

export_fn bool GASave(intptr_t ga, const char* path) {
    return ((GAUsed*)ga)->Save(std::string(path));
}
export_fn bool GALoad(intptr_t ga, const char* path) {
    return ((GAUsed*)ga)->Load(std::string(path));
}


export_fn double UtilCollisionTime(Vec v1, Vec v2, Vec p1, Vec p2, double r1, double r2) {
    return CollisionTime(v1, v2, p1, p2, r1, r2);
}