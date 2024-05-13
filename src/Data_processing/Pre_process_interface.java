package Data_processing;

import weka.core.Instances;

public interface Pre_process_interface {
    public abstract Instances apply(Instances data) throws Exception;
}
