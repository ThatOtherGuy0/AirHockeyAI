import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.space.ArrayObservationSpace;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import processing.core.PApplet;
import processing.core.PVector;

public class AILogic implements MDP<AILogic.AgentState, Integer, DiscreteSpace> {

    // Custom state container holding neural network input data
    public static class AgentState implements Encodable {
        private final INDArray stateArray;

        public AgentState(double[] rawInputs) {
            this.stateArray = Nd4j.create(rawInputs);
        }

        public AgentState(INDArray existingArray) {
            this.stateArray = existingArray.dup();
        }

        @Override
        public double[] toArray() {
            return stateArray.toDoubleVector();
        }

        @Override
        public INDArray getData() {
            return stateArray;
        }

        @Override
        public boolean isSkipped() {
            return false;
        }

        @Override
        public Encodable dup() {
            return new AgentState(stateArray);
        }
    }

    private final PApplet sketch;
    private final DiscreteSpace actionSpace = new DiscreteSpace(3); // 0: Turn Left, 1: Turn Right, 2: Stay Still
    private final ObservationSpace<AgentState> observationSpace = new ArrayObservationSpace<>(new int[]{2});

    // Agent Spatial Properties
    public PVector position;
    public float currentAngle = 0; // Angle facing in radians
    private final float turnSpeed = 0.1f; // Radians turned per step
    private int currentStepCount = 0;

    public AILogic(PApplet sketch) {
        this.sketch = sketch;
        this.position = new PVector(sketch.width / 2f, sketch.height / 2f);
    }

    @Override
    public ObservationSpace<AgentState> getObservationSpace() {
        return observationSpace;
    }

    @Override
    public DiscreteSpace getActionSpace() {
        return actionSpace;
    }

    @Override
    public AgentState reset() {
        currentStepCount = 0;
        currentAngle = sketch.random(PApplet.TWO_PI);
        return getCurrentObservation();
    }

    @Override
    public StepReply<AgentState> step(Integer action) {
        currentStepCount++;

        // Apply action chosen by AI
        if (action == 0) {
            currentAngle -= turnSpeed; // Turn Left
        } else if (action == 1) {
            currentAngle += turnSpeed; // Turn Right
        }
        
        // Wrap angle within [-PI, PI] radians
        currentAngle = PApplet.atan2(PApplet.sin(currentAngle), PApplet.cos(currentAngle));

        // Calculate difference between current facing angle and mouse target angle
        PVector vectorToMouse = new PVector(sketch.mouseX - position.x, sketch.mouseY - position.y);
        float targetAngle = vectorToMouse.heading();
        float angleDifference = PApplet.atan2(PApplet.sin(targetAngle - currentAngle), PApplet.cos(targetAngle - currentAngle));

        // Reward: Higher positive reward when angle error is near zero
        double reward = 1.0 - (Math.abs(angleDifference) / Math.PI);

        // End training episode after 300 steps
        boolean episodeFinished = currentStepCount >= 300;

        return new StepReply<>(getCurrentObservation(), reward, episodeFinished, null);
    }

    private AgentState getCurrentObservation() {
        PVector vectorToMouse = new PVector(sketch.mouseX - position.x, sketch.mouseY - position.y);
        float targetAngle = vectorToMouse.heading();
        float angleDifference = PApplet.atan2(PApplet.sin(targetAngle - currentAngle), PApplet.cos(targetAngle - currentAngle));
        
        // Scale values between -1 and 1 for neural network stability
        double normalizedAngleError = angleDifference / Math.PI;
        double normalizedDistance = Math.min(vectorToMouse.mag() / sketch.width, 1.0);

        return new AgentState(new double[]{normalizedAngleError, normalizedDistance});
    }

    @Override
    public boolean isDone() {
        return currentStepCount >= 300;
    }

    @Override
    public MDP<AgentState, Integer, DiscreteSpace> newInstance() {
        return new AILogic(sketch);
    }

    @Override
    public void close() {}
}