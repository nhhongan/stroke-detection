package Data_processing;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveWithValues;

public class DatasetFilter {
    private Instances dataset;
    private int classValue;
    private int delta;

    public DatasetFilter (Instances dataset, int classValue){
        this.dataset = dataset;
        this.classValue = classValue;
    }
    public Instances filterClass() throws Exception {
        // Set class index (target variable)
        if (dataset.classIndex() == -1) {
            dataset.setClassIndex(dataset.numAttributes() - 1);
        }

        if (classValue == 1) {
            delta = 0;
        } else {
            delta = 2;
        }

        // Create RemoveWithValues filter
        RemoveWithValues removeWithValues = new RemoveWithValues();
        String[] options = new String[4];
        options[0] = "-C"; // Specifies the attribute index
        options[1] = "last"; // The class attribute
        options[2] = "-L"; // Specify the nominal label to keep
        options[3] = String.valueOf(classValue + delta); // Keep class 0 (adjust based on your dataset if class 0 is not the first class)

        removeWithValues.setOptions(options);
        removeWithValues.setInputFormat(dataset);

        // Apply filter
        Instances filteredDataset = Filter.useFilter(dataset, removeWithValues);
        return filteredDataset;
    }
}
