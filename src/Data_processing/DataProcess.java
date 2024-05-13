package Data_processing;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.ArffLoader;
import java.io.File;

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

        public static void main(String[] args) throws Exception {
          
          String filePath = "dataset/healthcare-dataset-stroke-data.csv"; 
    
          Instances data = DataProcess.readData(filePath);
      
          System.out.println("Number of instances: " + data.numInstances());
          System.out.println("Number of attributes: " + data.numAttributes());
      
      }
      
}