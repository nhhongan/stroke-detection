package Data_processing;

import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.Filter;
import weka.core.Instances;
import weka.core.Attribute;
import weka.filters.unsupervised.attribute.Remove;

import java.util.Random;

public class Pre_process_apriori{
    public static Instances preprocess(Instances dataset) throws Exception{
        String[] options = new String[4];
        String[] opt = new String[]{"-R","12"};
        options [0] = "-B"; options[1] = "4";
        options[2] = "-R"; options[3] = "3,9,10";
        //Apply discretization:
        Discretize discretize = new Discretize();
        Remove remove = new Remove();
        discretize.setOptions(options);
        discretize.setInputFormat(dataset);
        Instances newData = Filter.useFilter(dataset, discretize);
        remove.setOptions(opt);
        remove.setInputFormat(newData);
        Instances newData2 = Filter.useFilter(newData,remove);
        // Remove "id" attribute ( because it is numeric and don't affect to the result)
        Attribute idAttribute = newData2.attribute("id");
        if (idAttribute != null) {
            newData2.deleteAttributeAt(idAttribute.index());
        }

//        for (int i = 0; i < newData2.numInstances(); i++) {
//            System.out.println(newData2.instance(i));
//        }


        return newData2;
    }

}