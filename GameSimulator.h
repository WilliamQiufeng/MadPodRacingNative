//
// Created by mac on 2022/12/25.
//

#ifndef MADPODRACING_GAMESIMULATOR_H
#define MADPODRACING_GAMESIMULATOR_H

#include <vector>
#include "Utils.h"
#include "ANN.h"

struct PodEncodeInfo {
    float x, y, vx, vy, m;
    void Write(ANNUsed& ann, int& currentNeuron);
};
struct PodDecodeInfo {
    float Angle, Thrust;
};

struct Pod {
    Vec Position, LastPosition;
    Vec Velocity;
    double Facing = 0;
    Vec TargetPosition;
    int Thrust;
    bool Boost, Boosted;
    int ShieldCD;
    bool IsEnemy;
    // Next checkpoint
    int NextCheckpointIndex;
    int Mass;
    int Lap;
    int NonCPTicks = 0;
    bool IsOut = false;
    bool Finished = false;
    bool IsCollided = false;
    constexpr const static float NormalMass = 1, ShieldMass = 10;
    constexpr const static int Radius = 400, RadiusSq = Radius * Radius;
    constexpr const static int Diameter = Radius * 2, DiameterSq = Diameter * Diameter;
    PodEncodeInfo Encode();
    void UpdateVelocity();
    double CheckCollision(const Pod& other) const;
};
class GameSimulator {
public:
    std::unique_ptr<Pod[]> Pods;
    std::vector<Vec> Checkpoints;
    std::shared_ptr<ANNUsed> ANNController;
    int PodsPerSide, TotalLaps;
    constexpr const static int CPRadius = 600, CPRadiusSq = CPRadius * CPRadius;
    constexpr static const Vec FieldSize{16000, 8000};
    GameSimulator(int podsPerSide, int totalLaps);
    GameSimulator();
    void SetANN(std::shared_ptr<ANNUsed> ann);
    void Setup(std::vector<Vec> checkpoints);
    void UpdatePodAI(int podIndex);
    bool Tick();
};
class GA {
    constexpr static const int Population = 50;
    std::array<ANNUsed, Population> Residents;
    std::array<GameSimulator, Population> Simulators;
    std::vector<Vec> Checkpoints;

    std::random_device RandomDevice;
    std::mt19937 RNG;
    std::uniform_real_distribution<int> DistributionX;
    std::uniform_real_distribution<int> DistributionY;

    GA();

    void Initialize();
    void Tick();
};

#endif //MADPODRACING_GAMESIMULATOR_H
