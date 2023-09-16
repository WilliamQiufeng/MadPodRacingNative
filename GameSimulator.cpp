//
// Created by mac on 2022/12/25.
//

#include "GameSimulator.h"

#include <utility>

GameSimulator::GameSimulator(int totalLaps) : TotalLaps(totalLaps) {

}

void GameSimulator::Setup(GAUsed *ga) {
    GA = ga;
    for (int i = 0; i < PodCount; i++) {
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
                0};
    }
}

void GameSimulator::UpdatePodAI(int podIndex) {
    int currentNeuron = 0;
    Pod& currentPod = Pods[podIndex];
    ANNUsed& ann = *(podIndex < PodsPerSide ? ANN1 : ANN2);
    currentPod.EncodeSelf().Write(ann, currentNeuron);
    for (int i = 0; i < PodCount; i++) {
        // If enemy, start filling enemy data first
        int j = podIndex < PodsPerSide ? i : (i + PodsPerSide) % PodCount;
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
        for (int i = 0; i < PodCount; i++) {
            auto& pod = Pods[i];
            if (!pod.IsEnabled()) continue;
            for (int j = i + 1; j < PodCount; j++) {
                auto& anotherPod = Pods[j];
                if (!pod.IsEnabled()) continue;
                if (i == lastI && j == lastJ) continue;
                auto collideTime = pod.CheckCollision(anotherPod);
                if (collideTime <= 0 || collideTime > remainTime) continue;
                if (collideTime < minNextCldTime) {
                    mctI = i;
                    mctJ = j;
                    minNextCldTime = collideTime;
                }
            }
        }
        if (minNextCldTime > remainTime) { // No more collisions
            for (int i = 0; i < PodCount; i++) {
                auto& pod = Pods[i];
                if (!pod.IsEnabled()) continue;
                pod.Position += pod.Velocity * remainTime;
            }
            break;
        }
        for (int i = 0; i < PodCount; i++) {
            auto& pod = Pods[i];
            if (!pod.IsEnabled()) continue;
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
    for (int i = 0; i < PodCount; i++) {
        UpdatePodAI(i);
    }
    for (int i = 0; i < PodCount; i++) {
        auto& pod = Pods[i];
        int companionPodIdx = (i < PodsPerSide ? 0 : PodsPerSide) + !(i % 2);
        Pod& companionPod = Pods[companionPodIdx];
        // 100+ turns not reaching a checkpoint: dead
        if (!pod.IsEnabled()) continue;
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
                pod.Finished = true;
                // First one finished wins.
                ANN1Won = !pod.IsEnemy;
                GameFinished = true;
                AllCheckpointsCompleted = true;
                return false;
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
    for (int i = 0; i < PodCount; i++) {
        auto& pod = Pods[i];
        pod.Velocity *= 0.85;
        pod.Position = pod.Position.Floor();
    }
    // If all pods are out or one pod has finished: return false
    // Else return true
    bool allDisabled = true, allOut = true;
    int finished1 = 0, finished2 = 0;
    for (int i = 0; i < PodCount; i++) {
        auto& pod = Pods[i];
        if (pod.Finished) {
            (pod.IsEnemy ? finished2 : finished1)++;
        } else if (pod.IsEnabled()) allDisabled = false;
        if (!pod.IsOut) allOut = false;
    }
    if (!allDisabled) CurrentTick++;
    else if (!allOut) { // Some have finished
//        std::cout << "Finish: " << finished1 << " to " << finished2 << std::endl;
        ANN1Won = finished1 >= finished2;
        GameFinished = true;
        return false;
    } else { // All out
        CalculateFitness();
        ANN1Won = Fitness1 > Fitness2;
        GameFinished = true;
        return false;
    }
    return true;
}

void GameSimulator::SetANN(ANNUsed::Pointer ann1, ANNUsed::Pointer ann2) {
//    ANN1 = std::move(ann1);
//    ANN2 = std::move(ann2);
    ANN1 = std::move(ann1);
    ANN2 = std::move(ann2);
}

fitness_t GameSimulator::Fitness(fitness_t& out, int offset) const {
    out = 0;
    for (int i = 0; i < PodsPerSide; i++) {
        auto& pod = Pods[offset + i];
        out += pod.CPPassed * 1000 - CurrentTick;
        if (!pod.Finished) {
            auto posDiff = GA->Checkpoints[pod.NextCheckpointIndex] - pod.Position;
            auto checkpointDist = GA->CPDistWithBefore[pod.NextCheckpointIndex];
            out += std::clamp<fitness_t>(checkpointDist / posDiff.Abs() * 1000, 0, 1000);
        }
        if (pod.IsOut) continue;
        if (pod.Finished) out += 3000; // ?
        if (pod.Boosted) out += 200;
    }
    return out;
}

fitness_t GameSimulator::CompetitiveFitness(fitness_t& out, int offset) const {
    out = 0;
    for (int i = 0; i < PodsPerSide; i++) {
        auto& pod = Pods[offset + i];
        out += pod.CPPassed * 100;
        if (!pod.Finished) {
            auto posDiff = GA->Checkpoints[pod.NextCheckpointIndex] - pod.Position;
            out += std::clamp<fitness_t>(GA->CPDistWithBefore[pod.NextCheckpointIndex] / posDiff.Abs(), 0, 30);
        }
    }
    return out;
}

void GameSimulator::CalculateFitness() {
    Fitness(Fitness1, 0);
    Fitness(Fitness2, PodsPerSide);
}

void GameSimulator::CalculateCompetitiveFitness() {
    CompetitiveFitness(Fitness1, 0);
    CompetitiveFitness(Fitness2, PodsPerSide);
}

double GameSimulator::Accuracy() {
    CalculateFitness();
    auto maxFitness = PodsPerSide * 25000;
    return std::max(Fitness1, Fitness2) / (double) maxFitness;
}

bool GameSimulator::Run(ANNUsed::Pointer ann1, ANNUsed::Pointer ann2, bool record) {
    Reset(GA);
    SetANN(std::move(ann1), std::move(ann2));
    if (record) Snapshots.clear();
    while (Tick()) {
        if (record) {
            Snapshots.emplace_back(*this);
        }
    }
    return ANN1Won;
}

bool GameSimulator::RunForCompletion(ANNUsed::Pointer ann1, ANNUsed::Pointer ann2, bool record) {
    Reset(GA);
    SetANN(std::move(ann1), std::move(ann2));
    if (record) Snapshots.clear();
    while (Tick()) {
        if (record) {
            Snapshots.emplace_back(*this);
        }
    }
    CalculateFitness();
    ANN1Won = Fitness1 > Fitness2 || Pods[0].Finished || Pods[1].Finished;
    return ANN1Won;
}

void GameSimulator::Reset(GAUsed *ga) {
    Setup(ga);
    CurrentTick = 1;
    Fitness1 = Fitness2 = -1;
    ANN1Won = false;
    GameFinished = false;
    AllCheckpointsCompleted = false;
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
    return !IsOut && !Finished;
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

Snapshot::Snapshot(GameSimulator& simulator) {
    CurrentTick = simulator.CurrentTick;
    simulator.CalculateFitness();
    Fitness1 = simulator.Fitness1;
    Fitness2 = simulator.Fitness2;
    for (int i = 0; i < GameSimulator::PodCount; i++) {
        Pods[i] = simulator.Pods[i];
    }
}
