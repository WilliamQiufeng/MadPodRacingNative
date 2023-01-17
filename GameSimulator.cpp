//
// Created by mac on 2022/12/25.
//

#include "GameSimulator.h"

#include <utility>

GameSimulator::GameSimulator(int podsPerSide, int totalLaps = 3) : PodsPerSide(podsPerSide), TotalLaps(totalLaps) {

}

void GameSimulator::Setup(std::vector<Vec> checkpoints) {
    Checkpoints = std::move(checkpoints);
    Pods.reset(new Pod[PodsPerSide * 2]);
    for (int i = 0; i < PodsPerSide * 2; i++) {
        Vec pos = Checkpoints[0] + UnitRight.Rotate(M_PI / PodsPerSide * i) * Pod::Diameter * 1.1;
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
    for (int i = 0; i < PodsPerSide * 2; i++) {
        // If enemy, start filling enemy data first
        int j = podIndex < PodsPerSide ? i : (i + PodsPerSide) % (PodsPerSide * 2);
        if (j == podIndex) continue;
        currentPod.Encode().Write(*ANNController, currentNeuron);
    }
    ANNController->Compute();
    float angle = ANNController->Neurons[ANNController->NeuronCount - 2];
    float thrust = ANNController->Neurons[ANNController->NeuronCount - 1];
    angle = angle * DegToRad(20); // Scale to [-20..20]
    bool boost = 0.5 <= thrust && thrust < 0;
    bool shield = thrust < 0.5;
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
    auto dp = Position - other.Position;
    auto dv = Velocity - other.Velocity;
//    auto minTime = -dp.Dot(dv) / dv.Dot(dv);
//    auto minSqDist = (dv * minTime + dp).SqDist();
//    if (minTime < 0 || minSqDist > DiameterSq) return -1;
//    return minTime;
    for (int ti = 0; ti <= 10; ti++) {
        auto t = 0.1 * ti;
        if ((dp + dv * t).SqDist() <= DiameterSq) {
            double l = 0, r = t, mid = r / 2;
            while (abs(r - l) > 1e-4) {
                if ((dp + dv * mid).SqDist() < DiameterSq) {
                    r = mid;
                } else {
                    l = mid;
                }
                mid = (l + r) / 2;
            }
            return mid;
        }
    }
    return -1;
}


bool GameSimulator::Tick() {
    for (int i = 0; i < PodsPerSide * 2; i++) {
        auto& pod = Pods[i];
        // 100+ turns not reaching a checkpoint: dead
        if (!pod.IsEnabled()) continue;
        // Check checkpoint collision
        if ((pod.Position - Checkpoints[pod.NextCheckpointIndex]).SqDist() <= CPRadiusSq) {
            pod.NextCheckpointIndex++;
            if (pod.NextCheckpointIndex >= Checkpoints.size()) {
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
    for (int i = 0; i < PodsPerSide * 2; i++) {
        auto& pod = Pods[i];
        if (pod.IsOut || pod.Finished || pod.IsCollided) continue;
        // O(N^2)
        for (int j = 0; j < PodsPerSide * 2; j++) {
            if (i == j) continue; // Not colliding itself obviously
            auto& anotherPod = Pods[j];
            if (!anotherPod.IsEnabled()) continue;
            auto collideTime = pod.CheckCollision(anotherPod);
            if (collideTime <= 0 || collideTime > 1) continue;
            // Before collision
            pod.Position += pod.Velocity * collideTime;
            anotherPod.Position += anotherPod.Velocity * collideTime;
            // Collision happens: change velocities
            auto [v1, v2] = ElasticCollision(pod.Mass, anotherPod.Mass, pod.Velocity, anotherPod.Velocity);
            pod.Velocity = MinImpulse(120, pod.Mass, v1);
            anotherPod.Velocity = MinImpulse(120, anotherPod.Mass, v2);
            // After collision
            pod.Position += pod.Velocity * (1 - collideTime);
            anotherPod.Position += anotherPod.Velocity * (1 - collideTime);
            pod.IsCollided = anotherPod.IsCollided = true;
            break;
        }
        // No collision happens
        if (!pod.IsCollided)
            pod.Position += pod.Velocity;
    }
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
        CalculatedFitness += (pod.Lap * Checkpoints.size() + pod.NextCheckpointIndex) / (double) CurrentTick;
        if (pod.IsOut) continue;
        if (pod.Finished) CalculatedFitness += 0.02; // ?
        if (pod.Boosted) CalculatedFitness += 0.01;
    }
    return CalculatedFitness;
}

bool GameSimulator::Compare(GameSimulator& a, GameSimulator& b) {
    return a.Fitness() < b.Fitness();
}

void GameSimulator::Reset(std::vector<Vec> cp) {
    Setup(cp);
    CurrentTick = 1;
    CalculatedFitness = -1;
}

double GameSimulator::RecalculateFitness() {
    CalculatedFitness = -1;
    return Fitness();
}

GameSimulator::GameSimulator() = default;

void PodEncodeInfo::Write(ANNUsed& ann, int& currentNeuron) {
    ann.Neurons[currentNeuron++] = x;
    ann.Neurons[currentNeuron++] = y;
    ann.Neurons[currentNeuron++] = vx;
    ann.Neurons[currentNeuron++] = vy;
    ann.Neurons[currentNeuron++] = m;
}

PodEncodeInfo Pod::Encode() {
    PodEncodeInfo res{};
    res.x = Position.x / GameSimulator::FieldSize.x;
    res.y = Position.y / GameSimulator::FieldSize.y;
    res.vx = Velocity.x / GameSimulator::FieldSize.x;
    res.vy = Velocity.y / GameSimulator::FieldSize.y;
    res.m = Mass / ShieldMass;
    return res;
}

bool Pod::IsEnabled() const {
    return !IsOut && !Finished;
}
