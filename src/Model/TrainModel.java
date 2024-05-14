package Model;

import weka.core.Instance;
import weka.core.Instances;

import java.util.Random;


public class TrainModel {
    public Instances train;
    public Instances test;
    public WekaModel model;

    public TrainModel(WekaModel model, Instances train, Instances test, Random random) {
        this.train = train;
        this.test = test;
        this.model = model;
    }

    public void trainModel() throws Exception {
        model.buildClassifier(train);
    }

    public double[] makePredictions() throws Exception {
        double[] predictions = new double[test.numInstances()];

        for (int i = 0; i < test.numInstances(); i++) {
            Instance instance = test.instance(i);
            predictions[i] = model.classifyInstance(instance);
        }
        return predictions;
    }

    public double calculateAccuracy(double[] predictions) {
        double correct = 0;

        for (int i = 0; i < test.numInstances(); i++) {
            if (test.instance(i).classValue() == predictions[i]) {
                correct++;
            }
        }

        return correct / test.numInstances();
    }

}
