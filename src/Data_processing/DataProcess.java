package Data_processing;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Attribute;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.ArffLoader;
import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import static java.lang.Math.max;


public class DataProcess {
    public DataProcess(){}
    
    public static Instances readData(String filePath) throws Exception {
        
        if (filePath.endsWith(".csv")) {
          String arffFilePath = convertCSVtoARFF(filePath);
                return readARFF(arffFilePath);
        } else if (filePath.endsWith(".arff")) {
                return readARFF(filePath);
        } else {
                throw new IllegalArgumentException("Unsupported file format. Please provide a CSV or ARFF file.");
        }
    }
      
    private static String convertCSVtoARFF(String csvFilePath) throws Exception {

        String arffFilePath = csvFilePath.replace(".csv", ".arff");
      
        CSVLoader loader = new CSVLoader();
        loader.setFile(new File(csvFilePath));
        Instances data = loader.getDataSet();
      
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(arffFilePath));
        saver.writeBatch();
      
        return arffFilePath;
    }

    public static Instances readARFF(String filePath) throws Exception {
        ArffLoader loader = new ArffLoader();
        loader.setFile(new File(filePath));
        return loader.getDataSet();
    }


    private static Instances convertStringToNumericAndHandleMissingValues(Instances data, String attributeName) {
    // Find the index of the attribute
        int attributeIndex = data.attribute(attributeName).index();

        // Collect numeric values and count missing values occurrences
        List<Double> nonMissingValues = new ArrayList<>();
        int naCount = 0;
        for (int i = 0; i < data.numInstances(); i++) {
            Instance instance = data.instance(i);
            if (!instance.isMissing(attributeIndex)) {
                try {
                    double value = Double.parseDouble(instance.stringValue(attributeIndex));
                    nonMissingValues.add(value);
                } catch (NumberFormatException e) {
                    // Handle N/A values
                    if (!instance.stringValue(attributeIndex).equalsIgnoreCase("N/A")) {
                        System.err.println("Non-numeric value found in " + attributeName + " attribute: " + instance.stringValue(attributeIndex));
                    } else {
                        naCount++;
                    }
                }
            }
        }

        // Calculate median of non-missing numeric values
        double median = calculateMedian(nonMissingValues);

        // Replace "N/A" values with median
        for (int i = 0; i < data.numInstances(); i++) {
            Instance instance = data.instance(i);
            if (instance.stringValue(attributeIndex).equalsIgnoreCase("N/A")) {
                instance.setValue(attributeIndex, median);
            }
        }

        // Set attribute to numeric
        Attribute attribute = new Attribute(attributeName);
        Instances newData = new Instances(data);
        newData.deleteAttributeAt(attributeIndex);
        newData.insertAttributeAt(new Attribute(attributeName), attributeIndex);

        System.out.println("Replaced " + naCount + " N/A values in " + attributeName + " with median: " + median);

        return data;
    }

    private static double calculateMedian(List<Double> values) {
        double[] sortedValues = values.stream().mapToDouble(Double::doubleValue).sorted().toArray();
        int middle = sortedValues.length / 2;
        if (sortedValues.length % 2 == 0) {
            return (sortedValues[middle - 1] + sortedValues[middle]) / 2;
        } else {
            return sortedValues[middle];
        }
    }

    public static Instances applySmoteOversampling(Instances data, HashMap<String, Integer> sampling_strategy, int K, String distance_metric) {
        Random rand = new Random();
        Smote smote = new Smote(sampling_strategy, K, distance_metric, rand);
        return smote.apply(data);
    }

    public static Instances normalize(Instances data) {
        for (int i = 0 ; i < data.numAttributes() ; i++) {
            if (data.classIndex() == i) {
                continue;
            }
            if (data.attribute(i).isNumeric()) {
                double mx = 0.0, mn = 1e9;

                for (int j = 0 ; j < data.numInstances() ; j++) {
                    mx = max(mx, data.instance(j).value(i));
                    mn = max(mn, data.instance(j).value(i));
                }

                for (int j = 0 ; j < data.numInstances() ; j++) {
                    double val = data.instance(j).value(i);
                    data.instance(j).setValue(i, val / mx);
//                    data.instance(j).setValue(i, (val - mn/ (mx - mn)));
                }
            }
        }

        return data;
    }

    public static void analyze_data(Instances data) {
        HashMap<String, Integer> _count = new HashMap<>();
        for (Instance instance:data) {
            String label = instance.stringValue(data.classIndex());
            if (_count.containsKey(label)) {
                _count.put(label, _count.get(label) + 1);
            } else {
                _count.put(label, 1);
            }
        }
        System.out.println(_count);
    } 
    public static Instances combineDatasets(Instances dataset1, Instances dataset2) {
        // Check if datasets are compatible
        if (!dataset1.equalHeaders(dataset2)) {
            throw new IllegalArgumentException("Datasets have different headers and cannot be combined.");
        }

        // Create a new Instances object to hold the combined data
        Instances combinedDataset = new Instances(dataset1);

        // Add all instances from the second dataset to the combined dataset
        for (int i = 0; i < dataset2.numInstances(); i++) {
            combinedDataset.add(dataset2.instance(i));
        }

        return combinedDataset;
    }
}