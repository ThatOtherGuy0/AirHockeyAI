import org.deeplearning4j.rl4j.learning.configuration.QLearningConfiguration;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense;
import org.deeplearning4j.rl4j.network.configuration.DQNDenseNetworkConfiguration;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.nd4j.linalg.learning.config.Adam;

import processing.core.PApplet;

public class MainSketch extends PApplet {

    private AILogic learningEnvironment;
    private QLearningDiscreteDense<AILogic.AgentState> qLearningTrainer;
    private DQNPolicy<AILogic.AgentState> trainedPolicy;
    private boolean isTraining = true;

    public static void main(String[] args) {
        PApplet.main("MainSketch");
    }

    @Override
    public void settings() {
        size(600, 600);
    }

    @Override
    public void setup() {
        frameRate(60);
        
        // 1. Create the RL Environment
        learningEnvironment = new AILogic(this);

        // 2. Configure Neural Network Structure
        DQNDenseNetworkConfiguration neuralNetConfig = DQNDenseNetworkConfiguration.builder()
                .l2(0.001)
                .updater(new Adam(0.005))
                .numHiddenNodes(32)
                .numLayers(2)
                .build();

        // 3. Configure RL Hyperparameters
        QLearningConfiguration rlTrainingConfig = QLearningConfiguration.builder()
                .seed(123L)
                .maxEpochStep(300)
                .maxStep(15000)      // Total learning steps
                .expRepMaxSize(5000)
                .batchSize(32)
                .targetDqnUpdateFreq(100)
                .updateStart(50)
                .rewardFactor(1.0)
                .gamma(0.95)
                .errorClamp(1.0)
                .minEpsilon(0.05f)   // Minimum random exploration rate
                .epsilonNbStep(3000)
                .doubleDQN(true)
                .build();

        // 4. Create the Q-Learning Trainer
        qLearningTrainer = new QLearningDiscreteDense<>(learningEnvironment, neuralNetConfig, rlTrainingConfig);

        // Run training in background thread so the Processing window opens smoothly
        new Thread(() -> {
            qLearningTrainer.train();
            trainedPolicy = qLearningTrainer.getPolicy();
            isTraining = false;
        }).start();
    }

    @Override
    public void draw() {
        background(30);

        // Once trained, feed current state into the trained neural policy to choose the best action
        if (!isTraining && trainedPolicy != null) {
            AILogic.AgentState currentState = learningEnvironment.reset();
            Observation currentObservation = new Observation(currentState.getData());
            int chosenAction = trainedPolicy.nextAction(currentObservation);
            learningEnvironment.step(chosenAction);
        }

        // Target guide line from agent to mouse
        stroke(100);
        line(learningEnvironment.position.x, learningEnvironment.position.y, mouseX, mouseY);

        // Draw Agent Triangle
        pushMatrix();
        translate(learningEnvironment.position.x, learningEnvironment.position.y);
        rotate(learningEnvironment.currentAngle);

        fill(0, 200, 255);
        stroke(255);
        triangle(20, 0, -10, -10, -10, 10);
        popMatrix();

        // Display Status Text
        fill(255);
        textSize(14);
        text(isTraining ? "Status: Training AI..." : "Status: Trained! Following mouse", 20, 30);
    }
}