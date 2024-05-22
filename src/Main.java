
import Data_processing.DataProcess;
import Data_processing.DatasetFilter;
import Data_processing.DatasetSplitter;
import Data_processing.Pre_process_apriori;
import Data_processing.Smote;
import Data_processing.Undersampler;
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
import weka.core.converters.ConverterUtils.DataSource;
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
        // Initial dataset
        System.out.println("Original dataset size: " + new DataSource(data).getDataSet().numInstances());
        double class0Count = data.attributeStats(11).nominalCounts[0]; 
        double class1Count = data.attributeStats(11).nominalCounts[1];
        System.out.println("Number of instances for class 0: " + class0Count);
        System.out.println("Number of instances for class 1: " + class1Count);

        // Undersampling
        Random random = new Random(507);
        double samplePercentage = 12.55794971; // ~13% for undersampling
        Undersampler undersampler = new Undersampler(data, samplePercentage);
        Instances undersampledDataset = undersampler.undersample();
        System.out.println("Undersampled dataset size: " + undersampledDataset.numInstances());
        double class0Count2 = undersampledDataset.attributeStats(11).nominalCounts[0]; 
        double class1Count2 = undersampledDataset.attributeStats(11).nominalCounts[1];
        System.out.println("Number of instances for class 0: " + class0Count2);
        System.out.println("Number of instances for class 1: " + class1Count2);
        DatasetFilter filter = new DatasetFilter(undersampledDataset, 0);
        Instances class_0_data = filter.filterClass();
        System.out.println("Class 0 dataset size: " + class_0_data.numInstances());
        double class0Count4 = class_0_data.attributeStats(11).nominalCounts[0]; 
        double class1Count4 = class_0_data.attributeStats(11).nominalCounts[1];
        System.out.println("Number of instances for class 0: " + class0Count4);
        System.out.println("Number of instances for class 1: " + class1Count4);

        // Apply SMOTE - Oversampling
        HashMap<String, Integer> params = new HashMap<>();
        params.put("0", (int) (class0Count4));
        params.put("1", (int) (class1Count2*Math.floor(class0Count4/class1Count2)*0.697));
        int K = 5;
        String distanceMetric = "Euclidean";
        Smote smote = new Smote(params, K, distanceMetric, random);
        Instances smoteData = smote.apply(data);
        System.out.println("Smote Dataset size: " + smoteData.numInstances());
        double class0Count3 = smoteData.attributeStats(11).nominalCounts[0]; 
        double class1Count3 = smoteData.attributeStats(11).nominalCounts[1];
        System.out.println("Number of instances for class 0: " + class0Count3);
        System.out.println("Number of instances for class 1: " + class1Count3);
        System.out.println("Number of attributes: " + smoteData.numAttributes());
        DatasetFilter filter2 = new DatasetFilter(smoteData, 1);
        Instances class_1_data = filter2.filterClass();
        System.out.println("Class 1 dataset size: " + class_1_data.numInstances());
        double class0Count5 = class_1_data.attributeStats(11).nominalCounts[0]; 
        double class1Count5 = class_1_data.attributeStats(11).nominalCounts[1];
        System.out.println("Number of instances for class 0: " + class0Count5);
        System.out.println("Number of instances for class 1: " + class1Count5);


        Instances balancedData = DataProcess.combineDatasets(class_0_data, class_1_data);
        System.out.println("Balanced dataset size: " + balancedData.numInstances());
        double class0Count6 = balancedData.attributeStats(11).nominalCounts[0]; 
        double class1Count6 = balancedData.attributeStats(11).nominalCounts[1];
        System.out.println("Number of instances for class 0: " + class0Count6);
        System.out.println("Number of instances for class 1: " + class1Count6);


        // Train test split
        double trainPercentage = 0.8; // 80% for train, 20% for test

        // Split dataset into train and test datasets
        DatasetSplitter splitter = new DatasetSplitter();
        Instances[] splitDatasets = splitter.splitDataset(balancedData, trainPercentage);
        Instances trainDataset = splitDatasets[0];
        Instances testDataset = splitDatasets[1];

        //  Print the sizes of train and test datasets
        System.out.println("Train dataset size: " + trainDataset.numInstances());
        double class0Count7 = trainDataset.attributeStats(11).nominalCounts[0]; 
        double class1Count7 = trainDataset.attributeStats(11).nominalCounts[1];
        System.out.println("Number of instances for class 0: " + class0Count7);
        System.out.println("Number of instances for class 1: " + class1Count7);
        System.out.println("Test dataset size: " + testDataset.numInstances());
        double class0Count8 = testDataset.attributeStats(11).nominalCounts[0]; 
        double class1Count8 = testDataset.attributeStats(11).nominalCounts[1];
        System.out.println("Number of instances for class 0: " + class0Count8);
        System.out.println("Number of instances for class 1: " + class1Count8);

        // ArffSaver saver = new ArffSaver();
        // saver.setInstances(balancedTrain);
        // saver.setFile(new File(outputFilePath));
        // saver.writeBatch();
        // System.out.println("Smote ARFF file saved: " + outputFilePath);

        trainDataset.randomize(random);

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
            Evaluate evaluate = new Evaluate(model, trainDataset, testDataset);
            evaluate.execute();
            KCrossVal kCrossVal = new KCrossVal(model, trainDataset, 10);
            kCrossVal.execute();
            String filePath = model.modelName() + ".model";
            kCrossVal.saveModel(filePath);
            System.out.println("---------------------------------");
        }

        apriori.setLowerBoundMinSupport(0.1);
        apriori.setMinMetric(0.5);
        apriori.buildAssociations(Pre_process_apriori.preprocess(trainDataset));
        System.out.println(apriori);
        System.out.println("---------------------------------");
        EvaluateApriori evaluateApriori = new EvaluateApriori(Pre_process_apriori.preprocess(trainDataset));
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