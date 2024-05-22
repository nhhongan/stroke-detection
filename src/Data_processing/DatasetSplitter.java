package Data_processing;

import weka.core.Instances;
import weka.core.Instance;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class DatasetSplitter {
    public Instances[] splitDataset(Instances dataset, double trainPercentage) {
        // Map to store instances for each class
        Map<Double, List<Instance>> classInstances = new HashMap<>();

        // Separate instances based on class
        for (Instance instance : dataset) {
            double classValue = instance.classValue();
            if (!classInstances.containsKey(classValue)) {
                classInstances.put(classValue, new ArrayList<>());
            }
            classInstances.get(classValue).add(instance);
        }

        // Shuffle instances within each class
        for (List<Instance> instances : classInstances.values()) {
            Collections.shuffle(instances);
        }

        // Determine number of instances for train and test datasets
        int numInstances = dataset.numInstances();
        int numTrainInstances = (int) (numInstances * trainPercentage);
        int numTestInstances = numInstances - numTrainInstances;

        // Create train and test datasets
        Instances trainDataset = new Instances(dataset, numTrainInstances);
        Instances testDataset = new Instances(dataset, numTestInstances);

        // Add instances to train and test datasets
        for (List<Instance> instances : classInstances.values()) {
            int numTrainInstancesPerClass = (int) (trainPercentage * instances.size());
            for (int i = 0; i < numTrainInstancesPerClass; i++) {
                trainDataset.add(instances.get(i));
            }
            for (int i = numTrainInstancesPerClass; i < instances.size(); i++) {
                testDataset.add(instances.get(i));
            }
        }

        return new Instances[]{trainDataset, testDataset};
    }
}

