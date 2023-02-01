//
// Created by mac on 2022/12/25.
//

#ifndef MADPODRACING_GAMESIMULATOR_H
#define MADPODRACING_GAMESIMULATOR_H

#include <vector>
#include "Utils.h"
#include "ANN.h"

void WriteNeuron(ANNUsed& ann, int& currentNeuron, float val);

struct PodEncodeInfo {
    float r, angle, vr, vAngle;

    void Write(ANNUsed& ann, int& currentNeuron);
};
struct SelfPodEncodeInfo {
    float vr, vAngle;
    void Write(ANNUsed& ann, int& currentNeuron);
};

void WriteCheckpoint(ANNUsed& ann, int& currentNeuron, Vec& cpPos, Vec& podPos);

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

    PodEncodeInfo Encode(Pod& relativeTo);
    SelfPodEncodeInfo EncodeSelf();

    void UpdateVelocity();

    double CheckCollision(const Pod& other) const;

    bool IsEnabled() const;
};

class GameSimulator {
public:
    std::shared_ptr<Pod[]> Pods;
    std::vector<Vec> Checkpoints;
    std::shared_ptr<ANNUsed> ANNController;
    int PodsPerSide, TotalLaps, CurrentTick = 1;
    double CalculatedFitness = -1;
    constexpr const static int CPRadius = 600, CPRadiusSq = CPRadius * CPRadius;
    constexpr const static int CPDiameter = CPRadius * 2, CPDiameterSq = CPDiameter * CPDiameter;
    constexpr static const Vec FieldSize{16000, 8000};
    static float FieldDiagonalLength;

    GameSimulator(int podsPerSide, int totalLaps);

    GameSimulator();

    void SetANN(std::shared_ptr<ANNUsed> ann);

    void Setup(std::vector<Vec> checkpoints);

    void MoveAndCollide();

    void UpdatePodAI(int podIndex);

    bool Tick();

    void Reset(std::vector<Vec> cp);

    double Fitness();

    double RecalculateFitness();

    static bool Compare(GameSimulator& a, GameSimulator& b);
};

template<int Population = 50>
class GA {
public:
    std::array<GameSimulator, Population> Simulators;
    std::array<double, Population> SelectionWeights;
    std::vector<Vec> Checkpoints;
    std::array<int, ANNUsed::LayersCount> Nodes;
    constexpr static const int PopulationCount = Population;
    constexpr static const int ChildrenCount = Population / 2;
    constexpr static const float CrossoverProbability = 0.5f;
    constexpr static const float MutateProbability = 0.06f;
private:
    std::random_device RandomDevice;
    std::mt19937 RNG;
    std::uniform_real_distribution<int> DistributionX;
    std::uniform_real_distribution<int> DistributionY;
    std::uniform_real_distribution<int> DistributionCPCount;
    std::discrete_distribution<int> DistributionSelection;
public:
    GA() : RNG(RandomDevice()), DistributionX(0, GameSimulator::FieldSize.x),
           DistributionY(0, GameSimulator::FieldSize.y),
           DistributionCPCount(3, 5) {

    }

    void RandomizeCheckpoints() {
        Checkpoints.clear();
        Checkpoints.resize(DistributionCPCount(RNG));
        for (int i = 0; i < Checkpoints.size(); i++) {
            while (true) {
                int x = DistributionX(RNG);
                int y = DistributionY(RNG);
                Checkpoints[i] = {x, y};
                bool rethrow = false;
                for (int j = 0; j < i; j++) {
                    if ((Checkpoints[i] - Checkpoints[j]).SqDist() <= GameSimulator::CPDiameterSq) {
                        rethrow = true;
                        break;
                    }
                }
                if (!rethrow) break;
            }
        }
    }

    void Initialize(std::array<int, ANNUsed::LayersCount> nodes) {
        Nodes = nodes;
        RandomizeCheckpoints();
        for (int i = 0; i < Population; i++) {
//            Residents[i].InitializeSpace(nodes);
//            Residents[i].Randomize();
            Simulators[i].SetANN(std::make_shared<ANNUsed>());
            Simulators[i].ANNController->InitializeSpace(Nodes);
            Simulators[i].ANNController->Randomize();
            Simulators[i].Setup(Checkpoints);
        }
    }

    bool Tick() {
        bool cont = false;
        for (int i = 0; i < Population; i++) {
            cont |= Simulators[i].Tick();
        }
        return cont;
    };
    void GenerationStart() {
        RandomizeCheckpoints();
        for (int i = 0; i < Population; i++) {
            Simulators[i].Reset(Checkpoints);
        }
    }
    void GenerationEnd() {
        SortByFitness();
        ProduceOffsprings();
    }

    bool Generation() {
        GenerationStart();
        while (Tick()) {

        }
        GenerationEnd();
        return true;
    }

    void ProduceOffsprings() {
        std::array<GameSimulator, ChildrenCount> offsprings;
        for (GameSimulator& gs: offsprings) {
            gs.SetANN(std::make_shared<ANNUsed>());
            gs.ANNController->InitializeSpace(Nodes);
            int father = DistributionSelection(RNG);
            int mother = DistributionSelection(RNG);
            Simulators[father].ANNController->Mate(*Simulators[mother].ANNController, *gs.ANNController,
                                                   CrossoverProbability, MutateProbability);
        }
        for (int i = 0; i < ChildrenCount; i++) {
            Simulators[Population - i - 1] = offsprings[i];
        }
    }

    void SortByFitness() {
        std::sort(Simulators.rbegin(), Simulators.rend(), GameSimulator::Compare);
        for (int i = 0; i < Population; i++) {
            SelectionWeights[i] = Simulators[i].Fitness();
        }
        DistributionSelection = std::discrete_distribution(SelectionWeights.begin(), SelectionWeights.end());
    }
};

typedef GA<> GAUsed;

#endif //MADPODRACING_GAMESIMULATOR_H
