//
// Created by mac on 2022/12/25.
//

#ifndef MADPODRACING_GAMESIMULATOR_H
#define MADPODRACING_GAMESIMULATOR_H

#include <vector>
#include <fstream>
#include <chrono>
#include "Utils.h"
#include "ANN.h"
using namespace std::chrono;

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
template<int Population = 1000>
class GA;
typedef GA<> GAUsed;

class GameSimulator {
public:
    std::shared_ptr<Pod[]> Pods;
    GAUsed* GA;
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

    void Setup(GAUsed* ga);

    void MoveAndCollide();

    void UpdatePodAI(int podIndex);

    bool Tick();

    void Reset(GAUsed* ga);

    double Fitness();

    double RecalculateFitness();

    static bool Compare(GameSimulator& a, GameSimulator& b);
};

template<int Population>
class GA {
public:
    std::array<GameSimulator, Population> Simulators;
    std::array<double, Population> SelectionWeights;
    std::shared_ptr<Vec[]> Checkpoints;
    std::shared_ptr<Vec[]> CPDiffWithBefore;
    std::shared_ptr<double[]> CPDistWithBefore;
    int CheckpointSize;
    std::array<int, ANNUsed::LayersCount> Nodes;
    int GenerationCount = 0;
    constexpr static const int PodsPerSide = 2;
    constexpr static const int Laps = 5;
    constexpr static const int PopulationCount = Population;
    constexpr static const int ChildrenCount = Population / 3 * 2;
    constexpr static const float CrossoverProbability = 0.5f;
    constexpr static const float MutateProbability = 0.06f;
private:
    std::random_device RandomDevice;
    std::mt19937 RNG;
    std::uniform_int_distribution<int> DistributionX{0, GameSimulator::FieldSize.x};
    std::uniform_int_distribution<int> DistributionY{0, GameSimulator::FieldSize.y};
    std::uniform_int_distribution<int> DistributionCPCount{3, 5};
    std::discrete_distribution<int> DistributionSelection;
public:
    GA() : RNG(RandomDevice()){

    }

    void RandomizeCheckpoints() {
        CheckpointSize = DistributionCPCount(RNG);
        Checkpoints.reset(new Vec[CheckpointSize]);
        CPDiffWithBefore.reset(new Vec[CheckpointSize]);
        CPDistWithBefore.reset(new double[CheckpointSize]);
        for (int i = 0; i < CheckpointSize; i++) {
            int tries = 0;
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
                if (!rethrow || ++tries > 10) break;
            }
        }
        for (int i = 0; i < CheckpointSize; i++) {
            auto nextIdx = (i + 1) % CheckpointSize;
            CPDiffWithBefore[nextIdx] = Checkpoints[nextIdx] - Checkpoints[i];
            CPDistWithBefore[nextIdx] = CPDiffWithBefore[nextIdx].Abs();
        }
    }

    void Initialize(std::array<int, ANNUsed::LayersCount> nodes = ANNUsed::DefaultNodes) {
        Nodes = nodes;
        RandomizeCheckpoints();
        for (int i = 0; i < Population; i++) {
//            Residents[i].InitializeSpace(nodes);
//            Residents[i].Randomize();
            Simulators[i] = GameSimulator(PodsPerSide, Laps);
            Simulators[i].SetANN(std::make_shared<ANNUsed>());
            Simulators[i].ANNController->InitializeSpace(Nodes);
            Simulators[i].ANNController->Randomize();
            Simulators[i].Setup(this);
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
            Simulators[i].Reset(this);
        }
        GenerationCount++;
    }
    void GenerationEnd() {
        SortByFitness();
        ProduceOffsprings();
    }

    bool Generation() {
        auto start = high_resolution_clock::now();
        GenerationStart();
        while (Tick()) {

        }
        GenerationEnd();
        auto end = high_resolution_clock::now();
        auto durationNs = duration_cast<microseconds>(end - start);
        std::cout << durationNs.count() << std::endl;
        std::cout << "Generation " << GenerationCount << ": \n";
        for (int i = 0; i < 3; i++) {
            std::cout << i << ": " << Simulators[i].Fitness() << std::endl;
        }
        return true;
    }

    void ProduceOffsprings() {
        std::array<GameSimulator, ChildrenCount> offsprings;
        for (GameSimulator& gs: offsprings) {
            gs = GameSimulator(PodsPerSide, Laps);
            gs.GA = this;
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
        for (auto& sim : Simulators) {
            sim.RecalculateFitness();
        }
        std::sort(Simulators.rbegin(), Simulators.rend(), GameSimulator::Compare);
        for (int i = 0; i < Population; i++) {
            SelectionWeights[i] = Simulators[i].Fitness();
        }
        DistributionSelection = std::discrete_distribution(SelectionWeights.begin(), SelectionWeights.end());
    }

    void Write(std::ostream& os) {
        for (auto& sim : Simulators) {
            sim.ANNController->Write(os);
        }
    }
    void Read(std::istream& is) {
        for (auto& sim : Simulators) {
            sim.ANNController->Read(is);
        }
    }
    bool Save(std::string path) {
        std::ofstream os(path, std::ios::binary);
        Write(os);
        os.close();
        return os.good();
    }
    bool Load(std::string path) {
        std::ifstream is(path, std::ios::binary);
        Read(is);
        is.close();
        return is.good();
    }
};


#endif //MADPODRACING_GAMESIMULATOR_H
