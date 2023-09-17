//
// Created by mac on 2022/12/25.
//

#include "GameSimulator.h"

#include <utility>

GameSimulator::GameSimulator(int totalLaps) : TotalLaps(totalLaps) {

}

void GameSimulator::Setup(GAUsed *ga) {
    GA = ga;
    for (int i = 0; i < TotalPods; i++) {
        Vec pos = GA->Checkpoints[0] + UnitRight.Rotate(M_PI / PodsPerSide * i) * Pod::Diameter * 1.1;
        Pods[i] = Pod{
                pos,
                pos,
                {0, 0},
                0,
                {0, 0},
                100,
                false,
                false,
                0,
                i >= PodsPerSide,
                1,
                1,
                0
        };
    }
}

void GameSimulator::UpdatePodAI(int podIndex) {
    int currentNeuron = 0;
    Pod& currentPod = Pods[podIndex];
    ANNUsed& ann = *Controllers[podIndex].ANN;
    currentPod.EncodeSelf().Write(ann, currentNeuron);
    for (int i = 0; i < TotalPods; i++) {
        // If enemy, start filling enemy data first
        int j = podIndex < PodsPerSide ? i : (i + PodsPerSide) % TotalPods;
        if (j == podIndex) continue;
        Pod& pod = Pods[j];
        if (pod.IsEnabled())
            pod.Encode(currentPod).Write(ann, currentNeuron);
        else
            FarawayPodEncodeInfo.Write(ann, currentNeuron);
    }
    WriteCheckpoint(ann, currentNeuron, GA->Checkpoints[currentPod.NextCheckpointIndex], currentPod);
    WriteCheckpoint(ann, currentNeuron, GA->Checkpoints[(currentPod.NextCheckpointIndex + 1) % GA->CheckpointSize],
                    currentPod);
    ann.Compute();
    float angle = ann.Neurons[ann.NeuronCount - 2];
    float thrust = ann.Neurons[ann.NeuronCount - 1];
    angle = angle * M_PI;
    bool boost = -0.5 <= thrust && thrust < 0;
    bool shield = thrust < -0.5;
    thrust = std::clamp<float>(thrust * 100, 0, 100);
    if (currentPod.ShieldCD == 0) {
        if (shield) {
            currentPod.ShieldCD = 3;
            currentPod.Mass = Pod::ShieldMass;
        } else if (boost && !currentPod.Boosted) {
            currentPod.Boost = currentPod.Boosted = true;
        }
    }
    currentPod.Thrust = thrust;
    currentPod.TargetPosition = currentPod.Position + UnitRight.Rotate(currentPod.Facing + angle) * 1000;
}

void Pod::UpdateVelocity() {
    // Update position
    auto idealDirection = TargetPosition - Position;
    auto rotateRad = Bound<double>(ClampRadian(idealDirection.Angle() - Facing), DegToRad(-18), DegToRad(18));
    auto thrust = Thrust;
    if (Boost) {
        thrust = Boosted ? 100 : 650;
        Boosted = true;
    }
    if (ShieldCD) {
        thrust = 0;
        ShieldCD--;
    }
    Mass = ShieldCD ? Pod::ShieldMass : Pod::NormalMass;
    Facing += rotateRad;
    Velocity += UnitRight.Rotate(Facing) * thrust;
}


double Pod::CheckCollision(const Pod& other) const {
    return CollisionTime(Velocity, other.Velocity, Position, other.Position, Radius, Radius);
}

void GameSimulator::MoveAndCollide() {
    const double maxTime = 1e9;
    double minNextCldTime = maxTime;
    int mctI, mctJ; // if min exists, the two collided object index
    int lastI = -1, lastJ = -1;
    double remainTime = 1;
    // O(N^2?)
    while (remainTime > 0) {
        for (int i = 0; i < TotalPods; i++) {
            auto& pod = Pods[i];
            if (pod.IsOut) continue; // Skip disabled
            for (int j = i + 1; j < TotalPods; j++) {
                auto& anotherPod = Pods[j];
                if (pod.IsOut) continue;
                if (i == lastI && j == lastJ) continue;
                auto collideTime = pod.CheckCollision(anotherPod);
                if (collideTime <= 0 || collideTime > remainTime) continue;
                if (collideTime < minNextCldTime) {
                    mctI = i;
                    mctJ = j;
                    minNextCldTime = collideTime;
                }
                pod.CollisionCount++;
                anotherPod.CollisionCount++;
                if ((i / PodsPerSide) != (j / PodsPerSide)) {
                    pod.EnemyCollisionCount++;
                    anotherPod.EnemyCollisionCount++;
                }
            }
        }
        if (minNextCldTime > remainTime) { // No more collisions
            for (int i = 0; i < TotalPods; i++) {
                auto& pod = Pods[i];
                if (pod.IsOut) continue;
                pod.Position += pod.Velocity * remainTime;
            }
            break;
        }
        for (int i = 0; i < TotalPods; i++) {
            auto& pod = Pods[i];
            if (pod.IsOut) continue;
            pod.Position += pod.Velocity * minNextCldTime;
        }
        auto& pi = Pods[mctI], & pj = Pods[mctJ];
        auto [v1, v2] = ElasticCollision(pi.Mass, pj.Mass, pi.Velocity, pj.Velocity, pi.Position, pj.Position);
        pi.Velocity = v1;
        pj.Velocity = v2;
        remainTime -= minNextCldTime;
        minNextCldTime = maxTime;
        lastI = mctI;
        lastJ = mctJ;
    }
}


bool GameSimulator::Tick() {
    if (GameFinished) return false;
    for (int i = 0; i < TotalPods; i++) {
        UpdatePodAI(i);
    }
    for (int i = 0; i < TotalPods; i++) {
        auto& pod = Pods[i];
        int companionPodIdx = (i < PodsPerSide ? 0 : PodsPerSide) + !(i % 2);
        Pod& companionPod = Pods[companionPodIdx];
        // 100+ turns not reaching a checkpoint: dead
        if (pod.IsOut) continue;
        // Check checkpoint collision
        if ((pod.Position - GA->Checkpoints[pod.NextCheckpointIndex]).SqDist() <= CPRadiusSq) {
            pod.NextCheckpointIndex++;
            pod.CPPassed++;
            if (pod.NextCheckpointIndex >= GA->CheckpointSize) {
                pod.NextCheckpointIndex = 0;
                pod.Lap++;
            }
            // Goes back to first checkpoint after all laps
            if (pod.Lap >= TotalLaps && pod.NextCheckpointIndex != 0) {
                AllCheckpointsCompleted = true;
            }
            pod.NonCPTicks = 0;
        } else {
            pod.NonCPTicks++;
            if (pod.NonCPTicks >= 100) {
                pod.IsOut = true;
                if (companionPod.IsOut) { // Two pods both out
                    ANN1Won = pod.IsEnemy;
                    GameFinished = true;
                    return false;
                }
                continue;
            }
        }
        pod.UpdateVelocity();
        pod.IsCollided = false;
    }
    MoveAndCollide();
    // Finishing work
    for (int i = 0; i < TotalPods; i++) {
        auto& pod = Pods[i];
        pod.Velocity *= 0.85;
        pod.Position = pod.Position.Floor();
    }
    // If all pods are out or one pod has finished: return false
    // Else return true
    bool allOut = true;
    int finished1 = 0, finished2 = 0;
    for (int i = 0; i < TotalPods; i++) {
        auto& pod = Pods[i];
        if (pod.Finished) {
            (pod.IsEnemy ? finished2 : finished1)++;
        }
        if (!pod.IsOut) allOut = false;
    }
    if (allOut) return false;
    if (CurrentTick > MaxTick) { // Some have finished
//        std::cout << "Finish: " << finished1 << " to " << finished2 << std::endl;
        ANN1Won = finished1 >= finished2;
        GameFinished = true;
        return false;
    }
    CurrentTick++;
    return true;
}

void GameSimulator::SetANN(ControllerStorage<TotalPods> anns) {
    Controllers = anns;
}

void GameSimulator::CalculateProgress() {
    auto totalCheckpointCount = Laps * GA->CheckpointSize;
    for (int i = 0; i < TotalPods; i++) {
        auto& controller = Controllers[i];
        auto& pod = Pods[i];
        controller.Progress = 0;
        controller.Progress += pod.CPPassed;
        if (!pod.Finished) {
            auto posDiff = GA->Checkpoints[pod.NextCheckpointIndex] - pod.Position;
            auto checkpointDist = GA->CPDistWithBefore[pod.NextCheckpointIndex];
            controller.Progress += std::clamp<double>(checkpointDist / posDiff.Abs(), 0, 1);
        }
        controller.Progress /= totalCheckpointCount;
    }
}

fitness_t GameSimulator::RunnerFitness(int podIndex) {
    auto& controller = Controllers[podIndex];
    auto& pod = Pods[podIndex];
    auto totalCheckpoints = Laps * GA->CheckpointSize;
    auto maximumFitnessPossible = 2000 + 100;
    controller.Fitness = controller.Progress * 1000;
    if (pod.Boosted) controller.Fitness += 100;
    // Maximum = Laps * CheckpointCount * 1000 + 3000 + 200
    controller.Fitness /= maximumFitnessPossible;
    return controller.Fitness;
}

fitness_t GameSimulator::DefenderFitness(int podIndex) {
    auto& controller = Controllers[podIndex];
    auto& pod = Pods[podIndex];
    double blockingEffectiveness = 0;
    double enemyMaxProgress = 0;
    double selfMaxProgress = 0;
    // Average collision per enemy. 1/10 for each collision
    double collisionRating = 0;
    for (int i = 0; i < TotalPods; i++) {
        if (i / PodsPerSide == podIndex / PodsPerSide) {
            selfMaxProgress = std::max(selfMaxProgress, Controllers[i].Progress);
            continue;
        }
        enemyMaxProgress = std::max(enemyMaxProgress, Controllers[i].Progress);
    }
    collisionRating += 0.025 * pod.EnemyCollisionCount;
    blockingEffectiveness = (selfMaxProgress - enemyMaxProgress) * 0.5;
    controller.Fitness = (collisionRating + blockingEffectiveness) / 2;
    return controller.Fitness;
}

fitness_t GameSimulator::Fitness(int podIndex, bool forceRunnerFitness) {
    if (forceRunnerFitness)
        return RunnerFitness(podIndex);
    switch (Controllers[podIndex].Role) {
        case PodRole::RunnerPod:
            return RunnerFitness(podIndex);
        case PodRole::DefenderPod:
            return DefenderFitness(podIndex);
        case PodRole::Unknown:
        case PodRole::PIDPod:
            return 0;
    }
}

fitness_t GameSimulator::SideAverageFitness(int offset) const {
    return (Controllers[offset].Fitness + Controllers[offset + 1].Fitness) / 2;
}

fitness_t GameSimulator::MaxFitnessByRole(PodRole role) const {
    fitness_t maxFitness = std::numeric_limits<fitness_t>::min();
    for (const auto& controller: Controllers) {
        if (controller.Role == role && controller.Fitness > maxFitness) maxFitness = controller.Fitness;
    }
    return maxFitness;
}

void GameSimulator::CalculateFitness(bool forceRunnerFitness) {
    CalculateProgress();
    for (int i = 0; i < TotalPods; i++) {
        Fitness(i, forceRunnerFitness);
    }
}

double GameSimulator::Accuracy() {
    return std::max(SideAverageFitness(0), SideAverageFitness(PodsPerSide));
}

bool GameSimulator::Run(ControllerStorage<TotalPods> anns, bool record) {
    Reset(GA);
    SetANN(anns);
    if (record) Snapshots.clear();
    while (Tick()) {
        if (record) {
            Snapshots.emplace_back(*this);
        }
    }
    CalculateFitness();
    return ANN1Won;
}

bool GameSimulator::RunForCompletion(ControllerStorage<TotalPods> anns, bool record) {
    Reset(GA);
    SetANN(anns);
    if (record) Snapshots.clear();
    while (Tick()) {
        if (record) {
            Snapshots.emplace_back(*this);
        }
    }
    CalculateFitness(true);
    ANN1Won = SideAverageFitness(0) > SideAverageFitness(PodsPerSide) || Pods[0].Finished || Pods[1].Finished;
    return ANN1Won;
}

void GameSimulator::Reset(GAUsed *ga) {
    Setup(ga);
    CurrentTick = 1;
    ANN1Won = false;
    GameFinished = false;
    AllCheckpointsCompleted = false;
}

void GameSimulator::Dump(std::ostream& os) {
    for (int i = 0; i < TotalPods; i++) {
        auto& pod = Pods[i];
        auto& controller = Controllers[i];
        os << "Pod " << i << "(" << (int) controller.Role << "): Progress "
           << std::setfill(' ') << std::setw(6) << (controller.Progress * 100) << "%, "
           << std::setfill(' ') << std::setw(7) << (controller.Fitness * 100) << "%; ";
    }
}

GameSimulator::GameSimulator() = default;

void PodEncodeInfo::Write(ANNUsed& ann, int& currentNeuron) const {
    WriteNeuron(ann, currentNeuron, r);
    WriteNeuron(ann, currentNeuron, angle);
    WriteNeuron(ann, currentNeuron, vr);
    WriteNeuron(ann, currentNeuron, vAngle);
}

PodEncodeInfo Pod::Encode(Pod& relativeTo) const {
    auto posDiff = Position - relativeTo.Position;
    PodEncodeInfo res{};
    res.r = posDiff.Abs() / GameSimulator::FieldDiagonalLength;
    res.angle = (posDiff.Angle() - relativeTo.Facing) / M_PI;
    auto relVel = Velocity - relativeTo.Velocity;
    res.vr = relVel.Abs() / 1000;
    res.vAngle = (relVel.Angle() - relativeTo.Facing) / M_PI;
    return res;
}

bool Pod::IsEnabled() const {
    return !IsOut;
}

SelfPodEncodeInfo Pod::EncodeSelf() const {
    SelfPodEncodeInfo res{};
    res.vr = Velocity.Abs() / 1000;
    res.vAngle = Velocity.Angle() / M_PI;
    return res;
}

void WriteNeuron(ANNUsed& ann, int& currentNeuron, float val) {
    ann.Neurons[currentNeuron++] = val;
}

void WriteCheckpoint(ANNUsed& ann, int& currentNeuron, Vec& cpPos, Pod& pod) {
    auto posDiff = cpPos - pod.Position;
    auto normalized = Vec{posDiff.x / GameSimulator::FieldSize.x, posDiff.y / GameSimulator::FieldSize.y};
    WriteNeuron(ann, currentNeuron, normalized.Abs());
    WriteNeuron(ann, currentNeuron, posDiff.Angle() - pod.Facing);
}

void SelfPodEncodeInfo::Write(ANNUsed& ann, int& currentNeuron) const {
    WriteNeuron(ann, currentNeuron, vr);
    WriteNeuron(ann, currentNeuron, vAngle);
}

float GameSimulator::FieldDiagonalLength = FieldSize.Abs();

void GameSimulator::SyncToStorage(RefControllerStorage<TotalPods> controllers) {
    for (int i = 0; i < Controllers.size(); i++) {
        controllers[i]->Fitness = Controllers[i].Fitness;
        controllers[i]->Progress = Controllers[i].Progress;
    }
}

void GameSimulator::SyncMaxToStorage(int source1, int source2, Controller& syncTarget) {
    syncTarget.Fitness = std::max(Controllers[source1].Fitness, Controllers[source2].Fitness);
    syncTarget.Progress = std::max(Controllers[source1].Progress, Controllers[source2].Progress);
}

Snapshot::Snapshot(GameSimulator& simulator) {
    CurrentTick = simulator.CurrentTick;
    simulator.CalculateFitness();
    for (int i = 0; i < TotalPods; i++) {
        Pods[i] = simulator.Pods[i];
    }
}
