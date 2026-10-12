import processing.core.PApplet;

public class MainSketch extends PApplet {

    public static void main(String[] args) {
        PApplet.main("MainSketch");
    }

    @Override
    public void settings() {
        size(AILogic.tatas, 500);
    }

    @Override
    public void setup() {
        println("Processing launched and reading AILogic.tatas: " + AILogic.tatas);
    }

    @Override
    public void draw() {
        background(30);
    }
}