
import Data_processing.DataProcess;
import Data_processing.Pre_process_apriori;
import Data_processing.Smote;
import Model.*;
import Evaluation.*;
import Data_processing.Pre_process_apriori;
import weka.associations.AssociationRule;
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
import weka.associations.Apriori;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
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
        Apriori apriori = new Apriori();

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
             Evaluate evaluate = new Evaluate(model, balancedTrain, test);
             evaluate.execute();
            KCrossVal kCrossVal = new KCrossVal(model, balancedTrain, 10);
            kCrossVal.execute();
            String filePath = model.modelName() + ".model";
            kCrossVal.saveModel(filePath);
            System.out.println("---------------------------------");
        }

        apriori.setLowerBoundMinSupport(0.1);
        apriori.setMinMetric(0.5);
        apriori.buildAssociations(Pre_process_apriori.preprocess(balancedTrain));
        System.out.println(apriori);
        System.out.println("---------------------------------");
        EvaluateApriori evaluateApriori = new EvaluateApriori(Pre_process_apriori.preprocess(balancedTrain));
        evaluateApriori.execute();


        // Test models with test dataset
        // ArrayList<String> model_names = new ArrayList<>(List.of("RandomForest", "NaiveBayes", "J48"));
        // for (int i = 0; i < model_names.size(); i++){
        //     String filePath = model_names.get(i) + ".model";
        //     WekaModel model = (WekaModel) weka.core.SerializationHelper.read(filePath);
        //     System.out.println("Testing data using " + model.modelName());
        //     Random rand = new Random(1);
        //     int randomNumber = rand.nextInt(test.numInstances() - 1 + 1) + 1;
        //     double actualValue = test.instance(randomNumber).classValue();
        //     Instance inst = test.instance(randomNumber);
        //     double pred = model.classifyInstance(inst);
        //     System.out.println("Actual value: " + actualValue + " --- Predicted value: " + pred);
        // }
    }
}