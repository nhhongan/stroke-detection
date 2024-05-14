package Model;

import weka.core.Copyable;
import weka.core.Instance;
import weka.core.Instances;
import java.util.Random;

public class TrainTestSplit {
    public Instances train;
    public Instances test;
    public Instances data;
    public double ratio;
    public Random random;
    public int numAttributes;
    public int targetIndex;

    public TrainTestSplit(Instances data, double ratio, Random random) {
        this.data = new Instances(data);
        this.ratio = ratio;
        this.train = new Instances(data, 0);
        this.test = new Instances(data, 0);
        this.random = random;
        this.data.randomize(this.random);
        this.numAttributes = this.data.numAttributes();
        this.targetIndex = this.numAttributes - 1;

        int trainSize = (int) Math.round(data.numInstances() * ratio);
        for (int i = 0; i < trainSize; i++) {
            this.train.add(this.data.instance(i));
        }
        this.train.setClassIndex(this.targetIndex);
        for (int i = trainSize; i < data.numInstances(); i++) {
            this.test.add(this.data.instance(i));
        }
        this.test.setClassIndex(this.targetIndex);
    }
}
