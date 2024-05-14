
import Data_processing.DataProcess;
import Data_processing.Smote;
import Model.*;
import Evaluation.KCrossVal;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.classifiers.trees.RandomTree;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

public class Main {
    public static void main(String[] args) {
        try {
            Instances data = DataProcess.readData("dataset/healthcare-dataset-stroke-non_missing_data.arff");
            data.setClassIndex(11);
            data = DataProcess.normalize(data);

            StrokeDetection(data);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void StrokeDetection(Instances data) throws Exception {
        double train_size = 0.7;
        Random random = new Random(507);
        TrainTestSplit trainTestSplit = new TrainTestSplit(data, train_size, random);
        // String outputFilePath = "healthcare-dataset-stroke-smote_randomize_train_data.arff";
        Instances train = trainTestSplit.train;
        Instances test = trainTestSplit.test;
        System.out.println(train.numInstances());

        // Count each class in training data
        double class0Count = train.attributeStats(11).nominalCounts[0]; 
        double class1Count = train.attributeStats(11).nominalCounts[1];
        System.out.println("Number of instances for class 0: " + class0Count);
        System.out.println("Number of instances for class 1: " + class1Count);

        // Apply SMOTE
        HashMap<String, Integer> params = new HashMap<>();
        params.put("0", (int) (class0Count));
        params.put("1", (int) (class1Count*Math.floor(class0Count/class1Count)));
        int K = 5;
        String distanceMetric = "Euclidean";
        Smote smote = new Smote(params, K, distanceMetric, random);
        Instances balancedTrain = smote.apply(train);
        System.out.println("\nAfter SMOTE:");
        System.out.println("Number of instances after SMOTE: " + balancedTrain.numInstances());
        System.out.println("Number of attributes: " + balancedTrain.numAttributes());

        // ArffSaver saver = new ArffSaver();
        // saver.setInstances(balancedTrain);
        // saver.setFile(new File(outputFilePath));
        // saver.writeBatch();
        // System.out.println("Smote ARFF file saved: " + outputFilePath);

        double class0Smote = balancedTrain.attributeStats(11).nominalCounts[0]; 
        double class1Smote = balancedTrain.attributeStats(11).nominalCounts[1];
        System.out.println("Number of instances for class 0: " + class0Smote);
        System.out.println("Number of instances for class 1: " + class1Smote);

        balancedTrain.randomize(random);

        RandomForest randomForest = new RandomForest();
        NaiveBayes naiveBayes = new NaiveBayes();
        J48 j48 = new J48();

        ArrayList<Classifier> classifiers = new ArrayList<>();
        classifiers.add(randomForest);
        classifiers.add(naiveBayes);
        classifiers.add(j48);

        ArrayList<WekaModel> models = new ArrayList<>();
        for (Classifier classifier : classifiers) {
            WekaModel model = new WekaModel(AbstractClassifier.makeCopy(classifier));
            models.add(model);
        }

        for (int i = 0; i < models.size(); i++) {
            WekaModel model = models.get(i);
            System.out.println("Training data using " + model.modelName());
            KCrossVal kCrossVal = new KCrossVal(model, balancedTrain, 10);
            kCrossVal.k_folds_validation();
            System.out.println("---------------------------------");
        }
    }
}