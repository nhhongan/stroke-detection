package Data_processing;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Resample;

public class Undersampler {
    private Instances dataset;
    private double samplePercentage;

    public Undersampler(Instances data, double samplePercentage) {
        this.dataset = data;
        this.samplePercentage = samplePercentage;
    }

    public Instances undersample() throws Exception {
        // Set class index (target variable)
        if (dataset.classIndex() == -1) {
            dataset.setClassIndex(dataset.numAttributes() - 1);
        }

        // Create Resample filter
        Resample resample = new Resample();
        resample.setInputFormat(dataset);

        // Set options for undersampling
        String[] options = new String[6];
        options[0] = "-S"; // Random seed
        options[1] = "1";  // Seed value
        options[2] = "-Z"; // Percentage of dataset to sample
        options[3] = String.valueOf(samplePercentage); // Percentage of the original dataset
        options[4] = "-no-replacement"; // Ensure no replacement
        options[5] = ""; // Placeholder for the flag
        // options[5] = "-B"; // Bias factor for undersampling
        // options[6] = "1.0"; // Bias value (default)

        resample.setOptions(options);

        // Apply filter
        Instances undersampledDataset = Filter.useFilter(dataset, resample);
        return undersampledDataset;
    }
}

