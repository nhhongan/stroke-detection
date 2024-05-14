package Evaluation;

import Model.WekaModel;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class KCrossVal {
    public WekaModel model;
    public Instances data;
    public int folds;
    public double avg_f1 = 0;

    public KCrossVal(WekaModel model, Instances data, int folds){
        this.model = model;
        this.data = data;
        this.folds = folds;
    }

    public void execute() throws Exception {
        if (data.classAttribute().isNominal()){
            data.stratify(folds);
        }
        for (int i = 0; i < folds; i++){
            Evaluation eval = new Evaluation(data);
            Instances train = data.trainCV(folds, i);
            Instances test = data.testCV(folds, i);
            this.model.buildClassifier(train);
            eval.evaluateModel(model.classifier, test);
            this.avg_f1 += eval.fMeasure(1);
        }
        this.avg_f1 = this.avg_f1/folds;
        System.out.println("Average F1 score: " + avg_f1);
    }

    public void saveModel(String filePath) throws Exception {
        weka.core.SerializationHelper.write(filePath, this.model);
    }
}
