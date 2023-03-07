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

template<int Population = 256>
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
    bool ANN1Won, GameFinished, AllCheckpointsCompleted;
    std::vector<Snapshot> Snapshots;

    explicit GameSimulator(int totalLaps);

    GameSimulator();

    void SetANN(std::shared_ptr<ANNUsed> ann1, std::shared_ptr<ANNUsed> ann2);

    void Setup(GAUsed *ga);

    void MoveAndCollide() const;

    void UpdatePodAI(int podIndex) const;

    bool Tick();

    bool Run(std::shared_ptr<ANNUsed> ann1, std::shared_ptr<ANNUsed> ann2, bool record = false);

    bool RunForCompletion(std::shared_ptr<ANNUsed> ann1, std::shared_ptr<ANNUsed> ann2, bool record = false);

    void Reset(GAUsed *ga);

    double Fitness(double& out, int offset) const;

    double SpeedlessFitness(double& out, int offset) const;

    void CalculateFitness();

    void CalculateSpeedlessFitness();

    double Accuracy();


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
    int LastRoundCompletionCount = 0;
    std::map<std::shared_ptr<ANNUsed>, double> FitnessBuffer;
    double Accuracy = 0;
    double MaxAccuracy = 0;
    constexpr static const int PodsPerSide = 2;
    constexpr static const int Laps = 5;
    constexpr static const int PopulationCount = Population;
    constexpr static const int ChildrenCount = Population / 4;
    constexpr static const float CrossoverProbability = 0.5f;
    constexpr static const float MutateProbability = 0.06f;
    constexpr static const int SelectionWeightBias = 1;
    constexpr static const int NeedCompletionPopulation = 10;
private:
    std::random_device RandomDevice;
    std::mt19937 RNG;
    std::uniform_int_distribution<int> DistributionX{0, GameSimulator::FieldSize.x};
    std::uniform_int_distribution<int> DistributionY{0, GameSimulator::FieldSize.y};
    std::uniform_int_distribution<int> DistributionCPCount{3, 5};
    std::discrete_distribution<int> DistributionSelection;
    std::ofstream StatsCsvFile;
    bool LogStats = false;
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
        LastRoundCompletionCount = 0;
    }

    void GenerationEnd() {
        ProduceOffsprings();
    }

    bool Generation() {
        auto start = high_resolution_clock::now();
        Accuracy = MaxAccuracy = 0;
        std::cout << "Last round completed: " << LastRoundCompletionCount << ", needed " << NeedCompletionPopulation
                  << std::endl;
        auto forCompletion = LastRoundCompletionCount < NeedCompletionPopulation;
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
            int against = DistributionSelection(RNG);
            if (forCompletion) {
                Simulator.RunForCompletion(ANNs[i], ANNs[against]);
            } else {
                Simulator.Run(ANNs[i], ANNs[against]);
                Simulator.CalculateFitness();
            }
            if (std::isnan(Simulator.Fitness1)) { ;
            }
            std::cout << i << " against " << against << ": " << Simulator.Fitness1 << " to " << Simulator.Fitness2;
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

    bool Compare(std::shared_ptr<ANNUsed>& a1, std::shared_ptr<ANNUsed>& a2) {
        return FitnessBuffer[a1] > FitnessBuffer[a2];
    }

    void SortByFitness() {
        FitnessBuffer.clear();
        for (auto& ann: ANNs) {
            Simulator.RunForCompletion(ann, ann);
            if (Simulator.AllCheckpointsCompleted) LastRoundCompletionCount++;
            FitnessBuffer[ann] = std::max(Simulator.Fitness1, Simulator.Fitness2);
            auto acc = Simulator.Accuracy();
            Accuracy += acc;
            MaxAccuracy = std::max(MaxAccuracy, acc);
        }
        Accuracy /= ANNs.size();
        std::sort(ANNs.begin(), ANNs.end(), [&](std::shared_ptr<ANNUsed>& a1, std::shared_ptr<ANNUsed>& a2) {
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
                    auto ann1Won = Simulator.Run(ANNs[i], ANNs[i + separation]);
                    if (!ann1Won) {
                        std::swap(ANNs[i], ANNs[i + separation]);
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

    void WritePlain(std::ostream& os) {
        for (auto& ann: ANNs) {
            ann->WritePlain(os);
        }
    }

    void ReadPlain(std::istream& is) {
        for (auto& ann: ANNs) {
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
