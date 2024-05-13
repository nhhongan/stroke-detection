package Data_processing;

import weka.core.Instances;

public class Pre_process_base implements Pre_process_interface {

    public Instances apply(Instances data) throws Exception {
        return data;
    }
}

