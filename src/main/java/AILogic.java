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

    //this whole class packages the agent data so the neural network can read it
    public static class AgentState implements Encodable {
        private final INDArray stateArray;//this holds the actual numbers going into the AI

        public AgentState(double[] rawInputs) {
            this.stateArray = Nd4j.create(rawInputs);//this converts the regular array into an INDArray
        }

        public AgentState(INDArray existingArray) {
            this.stateArray = existingArray.dup();//this copies the array so it doesnt get messed up
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
    //this is the end of the AgentState class, further code controls the AI logic

    private final PApplet sketch;//this gives access to the main Processing sketch
    private final DiscreteSpace actionSpace = new DiscreteSpace(3);//this sets the 3 possible moves: 0 is turn left, 1 is turn right, 2 is stay still
    private final ObservationSpace<AgentState> observationSpace = new ArrayObservationSpace<>(new int[]{2});//this tells the AI it is getting 2 input numbers

    public PVector position;//this controls the X and Y position of the agent
    public float currentAngle = 0;//this controls the angle the agent is facing in radians
    private final float turnSpeed = 0.1f;//this sets how fast the agent turns each step
    private int currentStepCount = 0;//this keeps track of how many steps have passed in the round

    public AILogic(PApplet sketch) {
        this.sketch = sketch;
        this.position = new PVector(sketch.width / 2f, sketch.height / 2f);//this puts the agent in the middle of the screen
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
    public AgentState reset() {//this resets everything for a new round
        currentStepCount = 0;
        currentAngle = sketch.random(PApplet.TWO_PI);//this starts the agent facing a random direction
        return getCurrentObservation();
    }

    @Override
    public StepReply<AgentState> step(Integer action) {
        currentStepCount++;

        if (action == 0) {//this makes the agent turn left
            currentAngle -= turnSpeed;
        } else if (action == 1) {//this makes the agent turn right
            currentAngle += turnSpeed;
        }
        
        currentAngle = PApplet.atan2(PApplet.sin(currentAngle), PApplet.cos(currentAngle));//this keeps the angle wrapped between -PI and PI so it doesnt spin out infinitely

        PVector vectorToMouse = new PVector(sketch.mouseX - position.x, sketch.mouseY - position.y);//this gets the direction from the agent to the mouse
        float targetAngle = vectorToMouse.heading();//this finds the exact angle to the mouse
        float angleDifference = PApplet.atan2(PApplet.sin(targetAngle - currentAngle), PApplet.cos(targetAngle - currentAngle));//this calculates how far off the agent is from pointing at the mouse

        double reward = 1.0 - (Math.abs(angleDifference) / Math.PI);//THIS IS THE REWARD, closer to pointing at the mouse gives a higher score

        boolean episodeFinished = currentStepCount >= 300;//this ends the training round after 300 steps

        return new StepReply<>(getCurrentObservation(), reward, episodeFinished, null);//this sends the new state, reward, and done status back to the trainer
    }

    private AgentState getCurrentObservation() {
        PVector vectorToMouse = new PVector(sketch.mouseX - position.x, sketch.mouseY - position.y);
        float targetAngle = vectorToMouse.heading();
        float angleDifference = PApplet.atan2(PApplet.sin(targetAngle - currentAngle), PApplet.cos(targetAngle - currentAngle));
        
        double normalizedAngleError = angleDifference / Math.PI;//this squishes the angle error between -1 and 1 so the neural network learns better
        double normalizedDistance = Math.min(vectorToMouse.mag() / sketch.width, 1.0);//this also squishes the distance between 0 and 1

        return new AgentState(new double[]{normalizedAngleError, normalizedDistance});//this bundles the inputs and sends them to the AI
    }

    @Override
    public boolean isDone() {
        return currentStepCount >= 300;//checks if the round is over
    }

    @Override
    public MDP<AgentState, Integer, DiscreteSpace> newInstance() {
        return new AILogic(sketch);//this creates a fresh copy of the environment
    }

    @Override
    public void close() {}
}
