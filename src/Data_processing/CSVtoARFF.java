package Data_processing;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import java.io.File;


public class CSVtoARFF {
    public CSVtoARFF(){}
    
    public void convert(String csv, String arff) throws Exception {
        // load CSV
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(csv));
        Instances data = loader.getDataSet();
        
        // save ARFF
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(arff));
        saver.writeBatch();

        System.out.println("CSV converted to ARFF successfully!");
    }
    
    public static void main(String[] args) throws Exception {
        CSVtoARFF converter = new CSVtoARFF();
        converter.convert("dataset/healthcare-dataset-stroke-data.csv", "dataset/data.arff");
    }

    
}
