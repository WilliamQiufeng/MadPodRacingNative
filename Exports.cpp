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
export_fn int Version() {
    return 2;
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

export_fn intptr_t GSCreate(int totalLaps) {
    return (std::intptr_t) new GameSimulator(totalLaps);
}
export_fn void GSSetup(intptr_t sim, intptr_t ga) {
    auto gameSim = (GameSimulator *) sim;
    gameSim->Setup((GAUsed *) ga);
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
    gameSim->Reset((GAUsed *) ga);
}
export_fn bool GSRun(intptr_t sim, int ann1Idx, int ann2Idx, bool record) {
    auto gs = (GameSimulator *) sim;
    return gs->Run(gs->GA->ANNs[ann1Idx], gs->GA->ANNs[ann2Idx], record);
}
export_fn bool GSRunPtr(intptr_t sim, intptr_t ann1, intptr_t ann2, bool record) {
    auto gs = (GameSimulator *) sim;
    std::shared_ptr<ANNUsed> ann1Ptr, ann2Ptr;
    ann1Ptr.reset((ANNUsed *) ann1);
    ann2Ptr.reset((ANNUsed *) ann2);
    return gs->Run(ann1Ptr, ann2Ptr, record);
}
export_fn void GSCalculateFitness(intptr_t sim) {
    ((GameSimulator *) sim)->CalculateFitness();
}
export_fn double GSFitness1(intptr_t sim) {
    return ((GameSimulator *) sim)->Fitness1;
}
export_fn double GSFitness2(intptr_t sim) {
    return ((GameSimulator *) sim)->Fitness2;
}
export_fn int GSSnapshotCount(std::intptr_t sim) {
    return ((GameSimulator *) sim)->Snapshots.size();
}
export_fn intptr_t GSSnapshot(intptr_t sim, int idx) {
    return (std::intptr_t) &((GameSimulator *) sim)->Snapshots[idx];
}
export_fn void GSSetupRandomANN(intptr_t sim) {
    auto& gs = *(GameSimulator *) sim;
    gs.SetANN(std::make_shared<ANNUsed>(), std::make_shared<ANNUsed>());
    gs.ANN1->InitializeSpace(ANNUsed::DefaultNodes);
    gs.ANN1->Randomize();
    gs.ANN2->InitializeSpace(ANNUsed::DefaultNodes);
    gs.ANN2->Randomize();
}
export_fn int GSCPCount(std::intptr_t sim) {
    return ((GameSimulator *) sim)->GA->CheckpointSize;
}
export_fn Pod *SnapshotGetPod(std::intptr_t snapshot, int idx) {
    return &((Snapshot *) snapshot)->Pods[idx];
}
export_fn int SnapshotGetTick(std::intptr_t snapshot) {
    return ((Snapshot *) snapshot)->CurrentTick;
}
export_fn fitness_t SnapshotGetFitness1(std::intptr_t snapshot) {
    return ((Snapshot *) snapshot)->Fitness1;
}

export_fn fitness_t SnapshotGetFitness2(std::intptr_t snapshot) {
    return ((Snapshot *) snapshot)->Fitness2;
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
export_fn int GACPCount(std::intptr_t ga) {
    return ((GAUsed *) ga)->CheckpointSize;
}
export_fn intptr_t GAGetANN(std::intptr_t ga, int idx) {
    return (intptr_t) &(((GAUsed *) ga)->ANNs[idx]);
}
export_fn intptr_t GAGetSimulator(intptr_t ga) {
    return (intptr_t) &((GAUsed *) ga)->Simulator;
}
export_fn int GAPopulation() {
    return GAUsed::PopulationCount;
}
export_fn bool GASave(intptr_t ga, const char *path) {
    return ((GAUsed *) ga)->Save(std::string(path));
}
export_fn bool GALoad(intptr_t ga, const char *path) {
    return ((GAUsed *) ga)->Load(std::string(path));
}
export_fn bool GASavePlain(intptr_t ga, const char *path) {
    return ((GAUsed *) ga)->SavePlain(std::string(path));
}
export_fn bool GALoadPlain(intptr_t ga, const char *path) {
    return ((GAUsed *) ga)->LoadPlain(std::string(path));
}
export_fn bool ANNWriteCode(intptr_t ann, const char *path) {
    return ((ANNUsed *) ann)->WriteCode(std::string(path));
}

export_fn double UtilCollisionTime(Vec v1, Vec v2, Vec p1, Vec p2, double r1, double r2) {
    return CollisionTime(v1, v2, p1, p2, r1, r2);
}