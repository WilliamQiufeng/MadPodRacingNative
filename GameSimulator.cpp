//
// Created by mac on 2022/12/25.
//

#include "GameSimulator.h"

#include <utility>

GameSimulator::GameSimulator(int podsPerSide, int totalLaps = 3) : PodsPerSide(podsPerSide), TotalLaps(totalLaps) {

}

void GameSimulator::Setup(GAUsed* ga) {
    GA = ga;
    Pods.reset(new Pod[PodsPerSide * 2]);
    for (int i = 0; i < PodsPerSide * 2; i++) {
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
                i < PodsPerSide,
                1,
                1,
                0};
    }
}

void GameSimulator::UpdatePodAI(int podIndex) {
    int currentNeuron = 0;
    Pod& currentPod = Pods[podIndex];
    currentPod.EncodeSelf().Write(*ANNController, currentNeuron);
    for (int i = 0; i < PodsPerSide * 2; i++) {
        // If enemy, start filling enemy data first
        int j = podIndex < PodsPerSide ? i : (i + PodsPerSide) % (PodsPerSide * 2);
        if (j == podIndex) continue;
        Pod& pod = Pods[j];
        pod.Encode(currentPod).Write(*ANNController, currentNeuron);
    }
    WriteCheckpoint(*ANNController, currentNeuron, GA->Checkpoints[currentPod.NextCheckpointIndex], currentPod.Position);
    ANNController->Compute();
    float angle = ANNController->Neurons[ANNController->NeuronCount - 2];
    float thrust = ANNController->Neurons[ANNController->NeuronCount - 1];
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
    currentPod.TargetPosition = currentPod.Position + UnitRight.Rotate(angle) * 1000;
}

void Pod::UpdateVelocity() {
    // Update position
    auto idealDirection = TargetPosition - Position;
    auto rotateDegree = Bound<double>(ClampRadian(idealDirection.Angle() - Facing), DegToRad(-18), DegToRad(18));
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
    Facing += rotateDegree;
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
        for (int i = 0; i < PodsPerSide * 2; i++) {
            auto& pod = Pods[i];
            if (!pod.IsEnabled()) continue;
            for (int j = i + 1; j < PodsPerSide * 2; j++) {
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
            for (int i = 0; i < PodsPerSide * 2; i++) {
                auto& pod = Pods[i];
                if (!pod.IsEnabled()) continue;
                pod.Position += pod.Velocity * remainTime;
            }
            break;
        }
        for (int i = 0; i < PodsPerSide * 2; i++) {
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
    for (int i = 0; i < PodsPerSide * 2; i++) {
        UpdatePodAI(i);
    }
    for (int i = 0; i < PodsPerSide * 2; i++) {
        auto& pod = Pods[i];
        // 100+ turns not reaching a checkpoint: dead
        if (!pod.IsEnabled()) continue;
        // Check checkpoint collision
        if ((pod.Position - GA->Checkpoints[pod.NextCheckpointIndex]).SqDist() <= CPRadiusSq) {
            pod.NextCheckpointIndex++;
            if (pod.NextCheckpointIndex >= GA->CheckpointSize) {
                pod.NextCheckpointIndex = 0;
                pod.Lap++;
            }
            // Goes back to first checkpoint after all laps
            if (pod.Lap >= TotalLaps && pod.NextCheckpointIndex != 0) {
                pod.Finished = true;
                continue;
            }
            pod.NonCPTicks = 0;
        } else {
            pod.NonCPTicks++;
            if (pod.NonCPTicks >= 100) {
                pod.IsOut = true;
                continue;
            }
        }
        pod.UpdateVelocity();
        pod.IsCollided = false;
    }
    MoveAndCollide();
    // Finishing work
    for (int i = 0; i < PodsPerSide * 2; i++) {
        auto& pod = Pods[i];
        pod.Velocity *= 0.85;
        pod.Position = pod.Position.Floor();
    }
    // If all pods are out or one pod has finished: return false
    // Else return true
    bool allOut = true;
    for (int i = 0; i < PodsPerSide * 2; i++) {
        auto& pod = Pods[i];
        if (pod.Finished) return false;
        if (!pod.IsOut) allOut = false;
    }
    if (!allOut) CurrentTick++;
    return !allOut;
}

void GameSimulator::SetANN(std::shared_ptr<ANNUsed> ann) {
    ANNController = std::move(ann);
}

double GameSimulator::Fitness() {
    if (CalculatedFitness != -1) return CalculatedFitness;
    CalculatedFitness = 0;
    for (int i = 0; i < PodsPerSide * 2; i++) {
        auto& pod = Pods[i];
        CalculatedFitness += (pod.Lap * GA->CheckpointSize + pod.NextCheckpointIndex) / (double) CurrentTick * 100;
        if (!pod.Finished) {

            auto posDiff = GA->Checkpoints[pod.NextCheckpointIndex] - pod.Position;
            CalculatedFitness += std::clamp<double>(GA->CPDistWithBefore[pod.NextCheckpointIndex] / posDiff.Abs(), 0, 3);
        }
        if (pod.IsOut) continue;
        if (pod.Finished) CalculatedFitness += 10; // ?
        if (pod.Boosted) CalculatedFitness += 1;
    }
    return CalculatedFitness;
}

bool GameSimulator::Compare(GameSimulator& a, GameSimulator& b) {
    return a.Fitness() < b.Fitness();
}

void GameSimulator::Reset(GAUsed* ga) {
    Setup(ga);
    CurrentTick = 1;
    CalculatedFitness = -1;
}

double GameSimulator::RecalculateFitness() {
    CalculatedFitness = -1;
    return Fitness();
}

GameSimulator::GameSimulator() = default;

void PodEncodeInfo::Write(ANNUsed& ann, int& currentNeuron) {
    WriteNeuron(ann, currentNeuron, r);
    WriteNeuron(ann, currentNeuron, angle);
    WriteNeuron(ann, currentNeuron, vr);
    WriteNeuron(ann, currentNeuron, vAngle);
}

PodEncodeInfo Pod::Encode(Pod& relativeTo) {
    auto posDiff = Position - relativeTo.Position;
    PodEncodeInfo res{};
    res.r = posDiff.Abs() / GameSimulator::FieldDiagonalLength;
    res.angle = posDiff.Angle() / M_PI;
    res.vr = relativeTo.Velocity.Abs() / 1000;
    res.vAngle = relativeTo.Velocity.Angle() / M_PI;
    return res;
}

bool Pod::IsEnabled() const {
    return !IsOut && !Finished;
}

SelfPodEncodeInfo Pod::EncodeSelf() {
    SelfPodEncodeInfo res{};
    res.vr = Velocity.Abs() / 1000;
    res.vAngle = Velocity.Angle() / M_PI;
    return res;
}

void WriteNeuron(ANNUsed& ann, int& currentNeuron, float val) {
    ann.Neurons[currentNeuron++] = val;
}

void WriteCheckpoint(ANNUsed& ann, int& currentNeuron, Vec& cpPos, Vec& podPos) {
    auto posDiff = cpPos - podPos;
    auto normalized = Vec{posDiff.x / GameSimulator::FieldSize.x, posDiff.y / GameSimulator::FieldSize.y};
    WriteNeuron(ann, currentNeuron, normalized.x);
    WriteNeuron(ann, currentNeuron, normalized.y);
}

void SelfPodEncodeInfo::Write(ANNUsed& ann, int& currentNeuron) {
    WriteNeuron(ann, currentNeuron, vr);
    WriteNeuron(ann, currentNeuron, vAngle);
}

float GameSimulator::FieldDiagonalLength = FieldSize.Abs();
