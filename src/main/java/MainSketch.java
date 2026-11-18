import org.deeplearning4j.rl4j.learning.configuration.QLearningConfiguration;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense;
import org.deeplearning4j.rl4j.network.configuration.DQNDenseNetworkConfiguration;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.nd4j.linalg.learning.config.Adam;

import processing.core.PApplet;

public class MainSketch extends PApplet {

    private AILogic learningEnvironment;//this holds the environment the AI learns in
    private QLearningDiscreteDense<AILogic.AgentState> qLearningTrainer;//this is the trainer that teaches the AI
    private DQNPolicy<AILogic.AgentState> trainedPolicy;//this stores the AI brain once it is done training
    private boolean isTraining = true;//this is the boolean that controls whether or not the AI is still training

    public static void main(String[] args) {
        PApplet.main("MainSketch");
    }

    @Override
    public void settings() {
        size(600, 600);//this sets the window size
    }

    @Override
    public void setup() {
        frameRate(60);
        
        learningEnvironment = new AILogic(this);//this creates the learning environment

        DQNDenseNetworkConfiguration neuralNetConfig = DQNDenseNetworkConfiguration.builder()//this whole chunk sets up the neural network structure
                .l2(0.001)
                .updater(new Adam(0.005))
                .numHiddenNodes(32)
                .numLayers(2)
                .build();

        QLearningConfiguration rlTrainingConfig = QLearningConfiguration.builder()//this whole chunk configures the training settings
                .seed(123L)
                .maxEpochStep(300)
                .maxStep(15000)//this sets the total learning steps
                .expRepMaxSize(5000)
                .batchSize(32)
                .targetDqnUpdateFreq(100)
                .updateStart(50)
                .rewardFactor(1.0)
                .gamma(0.95)
                .errorClamp(1.0)
                .minEpsilon(0.05f)//this sets the minimum random exploration rate
                .epsilonNbStep(3000)
                .doubleDQN(true)
                .build();

        qLearningTrainer = new QLearningDiscreteDense<>(learningEnvironment, neuralNetConfig, rlTrainingConfig);//this creates the Q-Learning trainer

        //this runs the training in the background so the window does not freeze
        new Thread(() -> {
            qLearningTrainer.train();//this actually starts the AI training
            trainedPolicy = qLearningTrainer.getPolicy();//this grabs the finished brain
            isTraining = false;//this tells the program training is done
        }).start();
    }

    @Override
    public void draw() {
        background(30);//this sets the color of the background

        if (!isTraining && trainedPolicy != null) {//this makes the AI move on its own once training is finished
            AILogic.AgentState currentState = learningEnvironment.reset();//this gets the current state of the agent
            Observation currentObservation = new Observation(currentState.getData());//this turns the state into something the AI can read
            int chosenAction = trainedPolicy.nextAction(currentObservation);//this picks the best move
            learningEnvironment.step(chosenAction);//this makes the agent actually do the move
        }

        stroke(100);
        line(learningEnvironment.position.x, learningEnvironment.position.y, mouseX, mouseY);//this draws the guide line from the agent to the mouse

        //this is the end of the AI logic, further code draws the agent
        pushMatrix();
        translate(learningEnvironment.position.x, learningEnvironment.position.y);//this moves the drawing origin to the agent
        rotate(learningEnvironment.currentAngle);//this rotates the agent to face the right direction

        fill(0, 200, 255);
        stroke(255);
        triangle(20, 0, -10, -10, -10, 10);//THIS IS THE AGENT
        popMatrix();

        fill(255);
        textSize(14);
        text(isTraining ? "Status: Training AI..." : "Status: Trained! Following mouse", 20, 30);//this displays whether the AI is training or ready
    }
}
