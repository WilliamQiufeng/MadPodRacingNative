//
// Created by mac on 2022/12/25.
//

#ifndef MADPODRACING_GAMESIMULATOR_H
#define MADPODRACING_GAMESIMULATOR_H

#include <vector>
#include <fstream>
#include <chrono>
#include <map>
#include "Utils.h"
#include "ANN.h"

using namespace std::chrono;


constexpr static const int PodsPerSide = 2;
constexpr static const int Laps = 5;

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

typedef int fitness_t;
enum PodType {
    RunnerPod,
    DefenderPod
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
    PodType Type;

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

template<int Population = 128>
class GA;

typedef GA<> GAUsed;
struct Snapshot;


class GameSimulator {
public:
    std::array<Pod, PodsPerSide * 2> Pods;
    GAUsed *GA = nullptr;
    ANNUsed::Pointer ANN1, ANN2;
    int TotalLaps = 3, CurrentTick = 1;
    constexpr const static int PodsPerSide = 2;
    constexpr const static int PodCount = PodsPerSide * 2;
    fitness_t Fitness1, Fitness2;
    constexpr const static int CPRadius = 600, CPRadiusSq = CPRadius * CPRadius;
    constexpr const static int CPDiameter = CPRadius * 2, CPDiameterSq = CPDiameter * CPDiameter;
    constexpr static const Vec FieldSize{16000, 8000};
    static float FieldDiagonalLength;
    bool ANN1Won, GameFinished, AllCheckpointsCompleted;
    std::vector<Snapshot> Snapshots;

    explicit GameSimulator(int totalLaps);

    GameSimulator();

    void SetANN(ANNUsed::Pointer ann1, ANNUsed::Pointer ann2);

    void Setup(GAUsed *ga);

    void MoveAndCollide();

    void UpdatePodAI(int podIndex);

    bool Tick();

    bool Run(ANNUsed::Pointer ann1, ANNUsed::Pointer ann2, bool record = false);

    bool RunForCompletion(ANNUsed::Pointer ann1, ANNUsed::Pointer ann2, bool record = false);

    void Reset(GAUsed *ga);

    fitness_t Fitness(fitness_t& out, int offset) const;

    fitness_t CompetitiveFitness(fitness_t& out, int offset) const;

    void CalculateFitness();

    void CalculateCompetitiveFitness();

    double Accuracy();


};

struct Snapshot {
public:
    std::array<Pod, GameSimulator::PodCount> Pods;
    int CurrentTick;
    fitness_t Fitness1, Fitness2;

    explicit Snapshot(GameSimulator& simulator);
};

template<int Population>
class ANNPopulation {
    struct RandomSelectionResult {
        int Index;
        ANNUsed::Pointer& ANN;
    };
public:
    std::array<ANNUsed::Pointer, Population> Storage;
    std::map<ANNUsed::Pointer, double> FitnessBuffer;
    std::discrete_distribution<int> DistributionSelection;
    std::mt19937 RNG;
    std::array<int, Population> SelectionWeights;
    std::array<int, ANNUsed::LayersCount> Nodes;
    constexpr static const int SelectionWeightBias = Population / 3;
    constexpr static const int ChildrenCount = Population / 4;
    constexpr static const float CrossoverProbability = 0.5f;
    constexpr static const float MutateProbability = 0.06f;

    ANNPopulation(decltype(RNG)& rng) : RNG(rng) {

        for (int i = 0; i < Population; i++) {
            SelectionWeights[i] = Population - i + SelectionWeightBias;
        }
        DistributionSelection = std::discrete_distribution(SelectionWeights.begin(), SelectionWeights.end());
    }

    void Initialize(std::array<int, ANNUsed::LayersCount> nodes) {
        Nodes = nodes;
        for (auto& ann: Storage) {
            ann = std::make_shared<ANNUsed>();
            ann->InitializeSpace(nodes);
            ann->Randomize();
        }
    }

    RandomSelectionResult SelectRandom() {
        int idx = DistributionSelection(RNG);
        auto& ann = Storage[idx];
        return {idx, ann};
    }

    void ProduceOffsprings() {
        std::array<ANNUsed::Pointer, ChildrenCount> offsprings;
        for (auto& newANN: offsprings) {
            auto [fatherIdx, father] = SelectRandom();
            auto [motherIdx, mother] = SelectRandom();
            newANN = std::make_shared<ANNUsed>();
            newANN->InitializeSpace(Nodes);
            father->Mate(*mother, *newANN, MutateProbability);
        }
        for (int i = 0; i < ChildrenCount; i++) {
            Storage[Population - i - 1] = offsprings[i];
        }
    }
};


template<int Population>
class GA {
public:
    ANNPopulation<Population> RunnerANNs;
    ANNPopulation<Population> DefenderANNs;
    GameSimulator Simulator;
    std::vector<Vec> Checkpoints;
    std::vector<Vec> CPDiffWithBefore;
    std::vector<double> CPDistWithBefore;
    int CheckpointSize;
    int GenerationCount = 0;
    int LastRoundCompletionCount = 0;
    std::map<ANNUsed::Pointer, double> FitnessBuffer;
    double Accuracy = 0;
    double MaxAccuracy = 0;
    constexpr static const int PopulationCount = Population;
    constexpr static const int NeedCompletionPopulation = Population / 3;
private:
    std::random_device RandomDevice;
    std::mt19937 RNG;
    std::uniform_int_distribution<int> DistributionX{0, GameSimulator::FieldSize.x / 4};
    std::uniform_int_distribution<int> DistributionY{0, GameSimulator::FieldSize.y / 4};
    std::uniform_int_distribution<int> DistributionCPCount{3, 5};
    std::ofstream StatsCsvFile;
    bool LogStats = false;
public:
    GA() : RNG(RandomDevice()), RunnerANNs(RNG), DefenderANNs(RNG) {
    }

    void RandomizeCheckpoints() {
        CheckpointSize = DistributionCPCount(RNG);
        Checkpoints.resize(CheckpointSize);
        CPDiffWithBefore.resize(CheckpointSize);
        CPDistWithBefore.resize(CheckpointSize);
        int curX = GameSimulator::FieldSize.x / 2;
        int curY = GameSimulator::FieldSize.y / 2;
        curX += DistributionX(RNG);
        curY += DistributionY(RNG);
        for (int i = 0; i < CheckpointSize; i++) {
            int tries = 0;
            while (true) {
                int x = DistributionX(RNG);
                int y = DistributionY(RNG);
                Checkpoints[i] = {curX + x, curY + y};
                bool rethrow = false;
                for (int j = 0; j < i; j++) {
                    if ((Checkpoints[i] - Checkpoints[j]).SqDist() <= GameSimulator::CPDiameterSq) {
                        rethrow = true;
                        break;
                    }
                }
                if (x < 0 || y < 0 || x > GameSimulator::FieldSize.x || y > GameSimulator::FieldSize.y) rethrow = true;
                else if (!rethrow || ++tries > 10) break;
            }
            curX = Checkpoints[i].x;
            curY = Checkpoints[i].y;
        }
        for (int i = 0; i < CheckpointSize; i++) {
            auto nextIdx = (i + 1) % CheckpointSize;
            CPDiffWithBefore[nextIdx] = Checkpoints[nextIdx] - Checkpoints[i];
            CPDistWithBefore[nextIdx] = CPDiffWithBefore[nextIdx].Abs();
        }
    }

    void Initialize(const std::array<int, ANNUsed::LayersCount>& nodes = ANNUsed::DefaultNodes) {
        RandomizeCheckpoints();
        RunnerANNs.Initialize(nodes);
        DefenderANNs.Initialize(nodes);
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
        LastRoundCompletionCount = 0;
    }

    void GenerationEnd() {
        RunnerANNs.ProduceOffsprings();
        DefenderANNs.ProduceOffsprings();
    }

    bool Generation() {
        auto start = high_resolution_clock::now();
        Accuracy = MaxAccuracy = 0;
        std::cout << "Last round completed: " << LastRoundCompletionCount << ", needed " << NeedCompletionPopulation
                  << std::endl;
        auto forCompletion = LastRoundCompletionCount < NeedCompletionPopulation;
//        forCompletion = true;
        GenerationStart();
//        for (int i = 0; i < 3; i++)
        if (forCompletion)
            SortByFitness();
        else
            TournamentSort();
        GenerationEnd();
        auto end = high_resolution_clock::now();
        auto durationNs = duration_cast<microseconds>(end - start);
        std::cout << durationNs.count() << std::endl;
        std::cout << "Generation " << GenerationCount << ", " << CheckpointSize << " checkpoints: \n";
        for (int i = 0; i < 3; i++) {
            auto [opponentIdx, opponent] = RunnerANNs.SelectRandom();
            if (forCompletion) {
                Simulator.RunForCompletion(RunnerANNs.Storage[i], opponent);
            } else {
                Simulator.Run(RunnerANNs.Storage[i], opponent);
                Simulator.CalculateFitness();
            }
            std::cout << i << " against " << opponentIdx << ": " << Simulator.Fitness1 << " to " << Simulator.Fitness2;
            for (int j = 0; j < PodsPerSide * 2; j++) {
                std::cout << " " << Simulator.Pods[j].CPPassed;
            }
            std::cout << " done in " << Simulator.CurrentTick << " ticks, ";
            std::cout << (Simulator.ANN1Won ? "Won" : "Lost");
            std::cout << std::endl;
        }
        std::cout << "Accuracy = " << Accuracy * 100 << "%" << std::endl;
        std::cout << "Max accuracy = " << MaxAccuracy * 100 << "%" << std::endl;
        if (LogStats) {
            StatsCsvFile << GenerationCount << "," << CheckpointSize << "," << Accuracy << "," << MaxAccuracy << ","
                         << LastRoundCompletionCount << std::endl;
        }
        return true;
    }

    bool Compare(ANNUsed::Pointer& a1, ANNUsed::Pointer& a2) {
        return FitnessBuffer[a1] > FitnessBuffer[a2];
    }

    void SortByFitness() {
        FitnessBuffer.clear();
        for (int i = 0; i < Population; i++) {
            auto& runnerANN = RunnerANNs.Storage[i];
            auto& defenderANN = DefenderANNs.Storage[i];
            Simulator.RunForCompletion(runnerANN, defenderANN);
            if (Simulator.AllCheckpointsCompleted) LastRoundCompletionCount++;
            RunnerANNs.FitnessBuffer[runnerANN] = Simulator.Fitness1;
            DefenderANNs.FitnessBuffer[defenderANN] = Simulator.Fitness2;
            auto acc = Simulator.Accuracy();
            Accuracy += acc;
            MaxAccuracy = std::max(MaxAccuracy, acc);
        }
        Accuracy /= Population;
        std::sort(RunnerANNs.Storage.begin(), RunnerANNs.Storage.end(),
                  [&](ANNUsed::Pointer& a1, ANNUsed::Pointer& a2) {
                      return Compare(a1, a2);
                  });
        std::sort(DefenderANNs.Storage.begin(), DefenderANNs.Storage.end(),
                  [&](ANNUsed::Pointer& a1, ANNUsed::Pointer& a2) {
                      return Compare(a1, a2);
                  });
    }

    void TournamentSort() {
        for (int round = 0; round < 4; round++) {
            auto separation = Population / 2;
            auto thisRoundCompletionCount = 0;
            int count = 0;
            double accuracySum = 0;
            while (separation) {
                for (int i = 0; i < separation; i++) {
                    // former lost. swap with latter.
                    auto ann1Won = Simulator.Run(RunnerANNs.Storage[i], RunnerANNs.Storage[i + separation]);
                    if (!ann1Won) {
                        std::swap(RunnerANNs.Storage[i], RunnerANNs.Storage[i + separation]);
//                        std::cout << "L";
                    } else {
//                        std::cout << "W";
                    }
                    count++;
                    auto acc = Simulator.Accuracy();
                    accuracySum += acc;
                    MaxAccuracy = std::max(MaxAccuracy, acc);
                    if (Simulator.AllCheckpointsCompleted) thisRoundCompletionCount++;
                }
                separation /= 2;
            }
//            std::cout << std::endl;
            LastRoundCompletionCount = std::max(thisRoundCompletionCount, LastRoundCompletionCount);
            Accuracy = std::max(Accuracy, accuracySum / count);
        }
    }


    // TODO
    void Write(std::ostream& os) {
        for (auto& ann: RunnerANNs.Storage) {
            ann->Write(os);
        }
    }

    void Read(std::istream& is) {
        for (auto& ann: RunnerANNs.Storage) {
            ann->Read(is);
        }
    }

    void WritePlain(std::ostream& os) {
        for (auto& ann: RunnerANNs.Storage) {
            ann->WritePlain(os);
        }
    }

    void ReadPlain(std::istream& is) {
        for (auto& ann: RunnerANNs.Storage) {
            ann->ReadPlain(is);
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

    bool SavePlain(const std::string& path) {
        std::ofstream os(path);
        WritePlain(os);
        os.close();
        return os.good();
    }

    bool LoadPlain(const std::string& path) {
        std::ifstream is(path);
        ReadPlain(is);
        is.close();
        return is.good();
    }

    void SetupStatsCsvFile(const std::string& path) {
        StatsCsvFile.open(path);
        LogStats = true;
        StatsCsvFile << "Generation,Checkpoints,AvgAcc,MaxAcc,Completed\n";
    }

    ~GA() {
        StatsCsvFile.close();
    }
};


#endif //MADPODRACING_GAMESIMULATOR_H
