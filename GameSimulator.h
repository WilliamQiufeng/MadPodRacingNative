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
constexpr static const int TotalPods = PodsPerSide * 2;
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

typedef double fitness_t;
enum class PodRole {
    Unknown,
    RunnerPod,
    DefenderPod,
    PIDPod,
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
    int CollisionCount = 0;
    int EnemyCollisionCount = 0;

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

template<int Population = 256>
class GA;

typedef GA<> GAUsed;
struct Snapshot;


struct Controller {
    ANNUsed::Pointer ANN;
    PodRole Role;
    fitness_t Fitness;
    /**
     * How much [0..1] of all the laps has the pod travelled
     */
    double Progress;
};

template<int Population>
using ControllerStorage = std::array<Controller, Population>;
template<int Population>
using RefControllerStorage = std::array<Controller *, Population>;

template<int Population>
class ANNPopulation {
    struct RandomSelectionResult {
        int Index;
        Controller& Controller;
    };
public:
    ControllerStorage<Population> Storage;
    std::map<ANNUsed::Pointer, double> FitnessBuffer;
    std::discrete_distribution<int> DistributionSelection;
    std::random_device RandomDevice;
    std::mt19937 RNG;
    std::array<int, Population> SelectionWeights;
    std::array<int, ANNUsed::LayersCount> Nodes;
    constexpr static const int SelectionWeightBias = Population / 3;
    constexpr static const int ChildrenCount = Population / 4;
    constexpr static const float CrossoverProbability = 0.5f;
    constexpr static const float MutateProbability = 0.06f;

    ANNPopulation() : RNG(RandomDevice()) {

        for (int i = 0; i < Population; i++) {
            SelectionWeights[i] = Population - i + SelectionWeightBias;
        }
        DistributionSelection = std::discrete_distribution(SelectionWeights.begin(), SelectionWeights.end());
    }

    void Initialize(std::array<int, ANNUsed::LayersCount> nodes, PodRole assignedRole) {
        Nodes = nodes;
        for (auto& controller: Storage) {
            controller.ANN = std::make_shared<ANNUsed>();
            controller.ANN->InitializeSpace(nodes);
            controller.ANN->Randomize();
            controller.Role = assignedRole;
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
            father.ANN->Mate(*mother.ANN, *newANN, MutateProbability);
        }
        for (int i = 0; i < ChildrenCount; i++) {
            Storage[Population - i - 1].ANN = offsprings[i];
        }
    }

    void SortByFitness() {
        std::sort(Storage.begin(), Storage.end(), [&](Controller& c1, Controller& c2) {
            return c1.Fitness > c2.Fitness;
        });
    }
};

class GameSimulator {
    fitness_t RunnerFitness(int podIndex);

    fitness_t DefenderFitness(int podIndex);

public:
    std::array<Pod, TotalPods> Pods;
    GAUsed *GA = nullptr;
    std::array<Controller, TotalPods> Controllers;
    int TotalLaps = 3, CurrentTick = 1;
    constexpr static const std::array<PodRole, 4> DefaultPodRoles = {
            PodRole::RunnerPod, PodRole::DefenderPod,
            PodRole::RunnerPod, PodRole::DefenderPod
    };
    constexpr const static int CPRadius = 600, CPRadiusSq = CPRadius * CPRadius;
    constexpr const static int CPDiameter = CPRadius * 2, CPDiameterSq = CPDiameter * CPDiameter;
    constexpr static const Vec FieldSize{16000, 8000};
    static float FieldDiagonalLength;
    bool ANN1Won, GameFinished, AllCheckpointsCompleted;
    std::vector<Snapshot> Snapshots;

    explicit GameSimulator(int totalLaps);

    GameSimulator();

    void SetANN(ControllerStorage<TotalPods> anns);


    void Dump(std::ostream& os);

    /**
     * Sets or resets pods and GA
     * @param ga GA
     */
    void Setup(GAUsed *ga);

    void MoveAndCollide();

    void UpdatePodAI(int podIndex);

    bool Tick();

    bool Run(ControllerStorage<TotalPods> anns, bool record = false);

    bool RunForCompletion(ControllerStorage<TotalPods> anns, bool record = false);

    void SyncToStorage(RefControllerStorage<TotalPods> controllers);

    void SyncMaxToStorage(int source1, int source2, Controller& syncTarget);;

    void Reset(GAUsed *ga);

    fitness_t Fitness(int podIndex, bool forceRunnerFitness = false);

    void CalculateProgress();

    [[nodiscard]] fitness_t SideAverageFitness(int offset) const;

    [[nodiscard]] fitness_t MaxFitnessByRole(PodRole role) const;

    void CalculateFitness(bool forceRunnerFitness = false);

    double Accuracy();

};

struct Snapshot {
public:
    std::array<Pod, TotalPods> Pods;
    int CurrentTick;

    explicit Snapshot(GameSimulator& simulator);
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
    double Accuracy = 0;
    double MaxAccuracy = 0;
    double MinAccuracy = 0;
    bool LastRoundIsCompletion = true;
    constexpr static const int PopulationCount = Population;
    constexpr static const int NeedTournamentPopulation = Population / 4;
    constexpr static const int NeedCompletionPopulation = Population / 6;
private:
    std::random_device RandomDevice;
    std::mt19937 RNG;
    std::uniform_int_distribution<int> DistributionX{0, GameSimulator::FieldSize.x / 4};
    std::uniform_int_distribution<int> DistributionY{0, GameSimulator::FieldSize.y / 4};
    std::uniform_int_distribution<int> DistributionCPCount{3, 5};
    std::uniform_int_distribution<> BooleanDistribution{0, 1};
    std::ofstream StatsCsvFile;
    bool LogStats = false;
public:
    GA() : RNG(RandomDevice()) {
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
        RunnerANNs.Initialize(nodes, PodRole::RunnerPod);
        DefenderANNs.Initialize(nodes, PodRole::DefenderPod);
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
        Accuracy = 0;
        MaxAccuracy = std::numeric_limits<decltype(MaxAccuracy)>::min();
        MinAccuracy = std::numeric_limits<decltype(MaxAccuracy)>::max();
        auto forCompletion = LastRoundIsCompletion ?
                             LastRoundCompletionCount < NeedTournamentPopulation :
                             LastRoundCompletionCount < NeedCompletionPopulation;
        LastRoundIsCompletion = forCompletion;
        std::cout << "Last round completed: " << LastRoundCompletionCount
                  << (forCompletion ? " Completion" : " Tournament")
                  << std::endl;

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
            auto proponentRunner = RunnerANNs.Storage[i];
            auto proponentDefender = DefenderANNs.Storage[i];
            auto [opponentRunnerIdx, opponentRunner] = RunnerANNs.SelectRandom();
            auto [opponentDefenderIdx, opponentDefender] = DefenderANNs.SelectRandom();
            if (forCompletion) {
                Simulator.RunForCompletion({proponentRunner, proponentDefender, opponentRunner, opponentDefender});
            } else {
                Simulator.Run({proponentRunner, proponentDefender, opponentRunner, opponentDefender});
            }
            std::cout << i << " against " << std::setfill('0') << std::setw(3) << opponentRunnerIdx << ": ";
            Simulator.Dump(std::cout);
            for (int j = 0; j < TotalPods; j++) {
                std::cout << " " << Simulator.Pods[j].CPPassed;
            }
            std::cout << " done in " << Simulator.CurrentTick << " ticks, ";
            std::cout << (Simulator.ANN1Won ? "Won" : "Lost");
            std::cout << std::endl;
        }
        std::cout << "Accuracy = " << Accuracy * 100 << "%" << std::endl;
        std::cout << "Max accuracy = " << MaxAccuracy * 100 << "%" << std::endl;
        if (LogStats) {
            StatsCsvFile << GenerationCount << "," << CheckpointSize << "," << Accuracy << "," << MinAccuracy << ","
                         << MaxAccuracy << ","
                         << LastRoundCompletionCount
                         << ((int) forCompletion) << std::endl;
        }
        return true;
    }

    void SortByFitness() {
        for (int i = 0; i < Population; i++) {
            auto& runnerController = RunnerANNs.Storage[i];
            auto& defenderController = DefenderANNs.Storage[i];
            Simulator.RunForCompletion({runnerController, defenderController, runnerController, defenderController});
            Simulator.SyncMaxToStorage(0, PodsPerSide, runnerController);
            Simulator.SyncMaxToStorage(1, PodsPerSide + 1, defenderController);
            if (Simulator.AllCheckpointsCompleted) LastRoundCompletionCount++;
            auto acc = Simulator.Accuracy();
            Accuracy += acc;
            MaxAccuracy = std::max(MaxAccuracy, acc);
            MinAccuracy = std::min(MinAccuracy, acc);
        }
        Accuracy /= Population;
        RunnerANNs.SortByFitness();
        DefenderANNs.SortByFitness();
    }

    void TournamentSort() {
        // TODO
        for (int round = 0; round < 4; round++) {
            auto separation = Population / 2;
            auto thisRoundCompletionCount = 0;
            int count = 0;
            double accuracySum = 0;
            while (separation) {
                for (int i = 0; i < separation; i++) {
                    // former lost. swap with latter.
                    auto proponent1 = &RunnerANNs.Storage[i];
                    auto proponent2 = &DefenderANNs.Storage[i];
                    auto opponent1 = &RunnerANNs.Storage[i + separation];
                    auto opponent2 = &DefenderANNs.Storage[i + separation];
                    if (BooleanDistribution(RNG)) {
                        std::swap(proponent1, proponent2);
                    }
                    if (BooleanDistribution(RNG)) {
                        std::swap(opponent1, opponent2);
                    }
                    auto ann1Won = Simulator.Run(
                            {
                                    *proponent1, *proponent2,
                                    *opponent1, *opponent2
                            });
                    Simulator.SyncToStorage(
                            {
                                    proponent1, proponent2,
                                    opponent1, opponent2
                            });
                    if (!ann1Won) {
                        std::swap(RunnerANNs.Storage[i], RunnerANNs.Storage[i + separation]);
                        std::swap(DefenderANNs.Storage[i], DefenderANNs.Storage[i + separation]);
//                        std::cout << "L";
                    } else {
//                        std::cout << "W";
                    }
                    count++;
                    auto acc = Simulator.Accuracy();
                    accuracySum += acc;
                    MaxAccuracy = std::max(MaxAccuracy, acc);
                    MinAccuracy = std::max(MinAccuracy, acc);
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
        for (auto& controller: RunnerANNs.Storage) {
            controller.ANN->Write(os);
        }
        for (auto& controller: DefenderANNs.Storage) {
            controller.ANN->Write(os);
        }
    }

    void Read(std::istream& is) {
        for (auto& controller: RunnerANNs.Storage) {
            controller.ANN->Read(is);
        }
        for (auto& controller: DefenderANNs.Storage) {
            controller.ANN->Read(is);
        }
    }

    void WritePlain(std::ostream& os) {
        for (auto& controller: RunnerANNs.Storage) {
            controller.ANN->WritePlain(os);
        }
        for (auto& controller: DefenderANNs.Storage) {
            controller.ANN->WritePlain(os);
        }
    }

    void ReadPlain(std::istream& is) {
        for (auto& controller: RunnerANNs.Storage) {
            controller.ANN->ReadPlain(is);
        }
        for (auto& controller: DefenderANNs.Storage) {
            controller.ANN->ReadPlain(is);
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
        StatsCsvFile << "Generation,Checkpoints,AvgAcc,MinAcc,MaxAcc,Completed,ForCompletion\n";
    }

    ~GA() {
        StatsCsvFile.close();
    }
};


#endif //MADPODRACING_GAMESIMULATOR_H
