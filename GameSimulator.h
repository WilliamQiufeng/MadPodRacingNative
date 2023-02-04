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

    void Write(ANNUsed& ann, int& currentNeuron) const;
};

constexpr static const PodEncodeInfo FarawayPodEncodeInfo{1, 0, 0, 0};

struct SelfPodEncodeInfo {
    float vr, vAngle;

    void Write(ANNUsed& ann, int& currentNeuron) const;
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
    int CPPassed = 0;
    constexpr const static float NormalMass = 1, ShieldMass = 10;
    constexpr const static int Radius = 400, RadiusSq = Radius * Radius;
    constexpr const static int Diameter = Radius * 2, DiameterSq = Diameter * Diameter;

    PodEncodeInfo Encode(Pod& relativeTo) const;

    [[nodiscard]] SelfPodEncodeInfo EncodeSelf() const;

    void UpdateVelocity();

    [[nodiscard]] double CheckCollision(const Pod& other) const;

    [[nodiscard]] bool IsEnabled() const;
};

void WriteCheckpoint(ANNUsed& ann, int& currentNeuron, Vec& cpPos, Pod& pod);

template<int Population = 512>
class GA;

typedef GA<> GAUsed;
struct Snapshot;


class GameSimulator {
public:
    std::shared_ptr<Pod[]> Pods;
    GAUsed *GA = nullptr;
    std::shared_ptr<ANNUsed> ANN1, ANN2;
    int TotalLaps = 3, CurrentTick = 1;
    constexpr const static int PodsPerSide = 2;
    constexpr const static int PodCount = PodsPerSide * 2;
    double Fitness1, Fitness2;
    constexpr const static int CPRadius = 600, CPRadiusSq = CPRadius * CPRadius;
    constexpr const static int CPDiameter = CPRadius * 2, CPDiameterSq = CPDiameter * CPDiameter;
    constexpr static const Vec FieldSize{16000, 8000};
    static float FieldDiagonalLength;
    bool ANN1Won, Finished;
    std::vector<Snapshot> Snapshots;

    explicit GameSimulator(int totalLaps);

    GameSimulator();

    void SetANN(std::shared_ptr<ANNUsed> ann1, std::shared_ptr<ANNUsed> ann2);

    void Setup(GAUsed *ga);

    void MoveAndCollide() const;

    void UpdatePodAI(int podIndex) const;

    bool Tick();

    bool Run(std::shared_ptr<ANNUsed> ann1, std::shared_ptr<ANNUsed> ann2, bool record = false);

    void Reset(GAUsed *ga);

    double Fitness(double& out, int offset) const;

    void CalculateFitness();


};

struct Snapshot {
public:
    std::array<Pod, GameSimulator::PodCount> Pods;
    int CurrentTick;
    double Fitness1, Fitness2;

    explicit Snapshot(GameSimulator& simulator);
};

template<int Population>
class GA {
public:
    std::array<std::shared_ptr<ANNUsed>, Population> ANNs;
    GameSimulator Simulator;
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
    constexpr static const int SelectionWeightBias = 50;
private:
    std::random_device RandomDevice;
    std::mt19937 RNG;
    std::uniform_int_distribution<int> DistributionX{GameSimulator::FieldSize.x};
    std::uniform_int_distribution<int> DistributionY{GameSimulator::FieldSize.y};
    std::uniform_int_distribution<int> DistributionCPCount{5};
    std::discrete_distribution<int> DistributionSelection;
public:
    GA() : RNG(RandomDevice()) {
        for (int i = 0; i < Population; i++) {
            SelectionWeights[i] = Population - i + SelectionWeightBias;
        }
        DistributionSelection = std::discrete_distribution(SelectionWeights.begin(), SelectionWeights.end());
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
        for (auto& ann: ANNs) {
            ann = std::make_shared<ANNUsed>();
            ann->InitializeSpace(Nodes);
            ann->Randomize();
        }
        Simulator = GameSimulator(Laps);
        Simulator.Setup(this);
    }


    bool Tick() {
        bool cont = false;
        cont |= Simulator.Tick();
        return cont;
    };

    void GenerationStart() {
        RandomizeCheckpoints();
        Simulator.Reset(this);
        GenerationCount++;
    }

    void GenerationEnd() {
        ProduceOffsprings();
    }

    bool Generation() {
        auto start = high_resolution_clock::now();
        GenerationStart();
        for (int i = 0; i < 2; i++) {
            TournamentSort();
        }
        GenerationEnd();
        auto end = high_resolution_clock::now();
        auto durationNs = duration_cast<microseconds>(end - start);
        std::cout << durationNs.count() << std::endl;
        std::cout << "Generation " << GenerationCount << ", " << CheckpointSize << " checkpoints: \n";
        for (int i = 0; i < 3; i++) {
            int against = DistributionSelection(RNG);
            Simulator.Run(ANNs[i], ANNs[against]);
            Simulator.CalculateFitness();
            std::cout << i << " against " << against << ": " << Simulator.Fitness1 << " to " << Simulator.Fitness2;
            for (int j = 0; j < PodsPerSide * 2; j++) {
                std::cout << " " << Simulator.Pods[j].CPPassed;
            }
            std::cout << " done in " << Simulator.CurrentTick << " ticks, ";
            std::cout << (Simulator.ANN1Won ? "Won" : "Lost") << std::endl;
        }
        return true;
    }

    void TournamentSort() {
        auto separation = Population / 2;
        while (separation) {
            for (int i = 0; i < separation; i++) {
                // former lost. swap with latter.
                if (!Simulator.Run(ANNs[i], ANNs[i + separation])) {
                    std::swap(ANNs[i], ANNs[i + separation]);
                    std::cout << "L";
                } else {
                    std::cout << "W";
                }
            }
            separation /= 2;
        }
        std::cout << std::endl;
    }

    void ProduceOffsprings() {
        std::array<std::shared_ptr<ANNUsed>, ChildrenCount> offsprings;
        for (auto& newANN: offsprings) {
            int father = DistributionSelection(RNG);
            int mother = DistributionSelection(RNG);
            newANN = std::make_shared<ANNUsed>();
            newANN->InitializeSpace(Nodes);
            ANNs[father]->Mate(*ANNs[mother], *newANN, MutateProbability);
        }
        for (int i = 0; i < ChildrenCount; i++) {
            ANNs[Population - i - 1] = offsprings[i];
        }
    }

    void Write(std::ostream& os) {
        for (auto& ann: ANNs) {
            ann->Write(os);
        }
    }

    void Read(std::istream& is) {
        for (auto& ann: ANNs) {
            ann->Read(is);
        }
    }

    bool Save(const std::string& path) {
        std::ofstream os(path, std::ios::binary);
        Write(os);
        os.close();
        return os.good();
    }

    bool Load(const std::string& path) {
        std::ifstream is(path, std::ios::binary);
        Read(is);
        is.close();
        return is.good();
    }
};


#endif //MADPODRACING_GAMESIMULATOR_H
